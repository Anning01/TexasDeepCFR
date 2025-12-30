import random
from collections import deque

import numpy as np
import pokerkit_adapter as pokers
import torch
import torch.nn.functional as F
import torch.optim as optim

from core.model import PokerNetwork, encode_state, VERBOSE
from game_logger import log_game_error
from settings import STRICT_CHECKING


class PrioritizedMemory:
    """带有优先经验回放的增强型记忆缓冲器。"""

    def __init__(self, capacity, alpha=0.6):
        """
        初始化带有优先经验回放的记忆缓冲器。

        参数:
            capacity: 可存储的最大经验数量
            alpha: 控制优先级使用程度 (0 = 无优先级, 1 = 完全优先级)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        self._max_priority = 1.0  # Initial max priority for new experiences

    def add(self, experience, priority=None):
        """
        将新经验及其优先级添加到记忆中。

        参数:
            experience: 经验元组 (state, opponent_features, action_type, bet_size, regret)
            priority: 可选的显式优先级值 (如果为None则默认为最大优先级)
        """
        if priority is None:
            priority = self._max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority**self.alpha)
        else:
            # Replace the oldest entry
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority**self.alpha

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        根据经验的优先级采样一批经验。

        参数:
            batch_size: 要采样的经验数量
            beta: 控制重要性采样修正 (0 = 无修正, 1 = 完全修正)
                 应在训练过程中从~0.4退火到1

        返回:
            (samples, indices, importance_sampling_weights) 的元组
        """
        if len(self.buffer) < batch_size:
            # If we don't have enough samples, return all with equal weights
            return self.buffer, list(range(len(self.buffer))), np.ones(len(self.buffer))

        # Convert priorities to probabilities
        total_priority = sum(self.priorities)
        probabilities = np.array([p / total_priority for p in self.priorities])
        # 修复浮点精度问题，确保概率总和为1
        probabilities = probabilities / probabilities.sum()

        # Sample indices based on priorities
        indices = np.random.choice(
            len(self.buffer), batch_size, p=probabilities, replace=False
        )
        samples = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        weights = []
        for idx in indices:
            # P(i) = p_i^α / sum_k p_k^α
            # weight = (1/N * 1/P(i))^β = (N*P(i))^-β
            sample_prob = self.priorities[idx] / total_priority
            weight = (len(self.buffer) * sample_prob) ** -beta
            weights.append(weight)

        # Normalize weights to have maximum weight = 1
        # This ensures we only scale down updates, never up
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priority(self, index, priority):
        """
        更新经验的优先级。

        参数:
            index: 要更新的经验的索引
            priority: 新的优先级值 (在alpha调整之前)
        """
        # Clip priority to be positive
        priority = max(1e-8, priority)

        # Keep track of max priority for new experience initialization
        self._max_priority = max(self._max_priority, priority)

        # Store alpha-adjusted priority
        self.priorities[index] = priority**self.alpha

    def __len__(self):
        """返回记忆的当前大小。"""
        return len(self.buffer)

    def get_memory_stats(self):
        """获取当前记忆缓冲器的统计信息。"""
        if not self.priorities:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "size": 0}

        raw_priorities = [p ** (1 / self.alpha) for p in self.priorities]
        return {
            "min": min(raw_priorities),
            "max": max(raw_priorities),
            "mean": sum(raw_priorities) / len(raw_priorities),
            "median": sorted(raw_priorities)[len(raw_priorities) // 2],
            "size": len(self.buffer),
        }


class DeepCFRAgent:
    def __init__(self, player_id=0, num_players=6, memory_size=300000, device="cpu"):
        self.player_id = player_id
        self.num_players = num_players
        self.device = device

        # 定义动作类型 (弃牌, 跟注, 加注)
        self.num_actions = 3

        # 根据状态编码计算输入大小
        input_size = (
            52 + 52 + 5 + 1 + num_players + num_players + num_players * 4 + 1 + 4 + 5
        )

        # 创建带有下注大小的优势网络
        self.advantage_net = PokerNetwork(
            input_size=input_size, hidden_size=256, num_actions=self.num_actions
        ).to(device)

        # 优势网络学习率 - 1e-4 是 Deep CFR 论文的推荐值
        self.optimizer = optim.Adam(
            self.advantage_net.parameters(), lr=1e-4, weight_decay=1e-5
        )

        # 创建优先记忆缓冲器
        self.advantage_memory = PrioritizedMemory(memory_size)

        # 策略网络
        self.strategy_net = PokerNetwork(
            input_size=input_size, hidden_size=256, num_actions=self.num_actions
        ).to(device)
        # 策略网络学习率略低于优势网络，因为它需要更稳定
        self.strategy_optimizer = optim.Adam(
            self.strategy_net.parameters(), lr=5e-5, weight_decay=1e-5
        )
        self.strategy_memory = deque(maxlen=memory_size)

        # 用于保存统计信息
        self.iteration_count = 0

        # 遗憾归一化跟踪器
        self.max_regret_seen = 1.0

        # 下注大小范围 (作为底池的倍数)
        self.min_bet_size = 0.1
        self.max_bet_size = 3.0

    def action_type_to_pokers_action(
        self, action_type, state, bet_size_multiplier=None
    ):
        """
        将动作类型和可选的下注大小转换为Pokers动作。
        (带有浮点数保护的优化加注逻辑)
        """
        # 访问VERBOSE，假设它是全局设置的或可访问的（例如，如果是实例属性则为self.verbose）
        # 在此示例中，我假设VERBOSE是从model.py或utils导入的全局变量
        # 如果VERBOSE是self.verbose这样的实例变量，请使用该变量。
        # from src.core.model import VERBOSE # 确保此导入在文件顶部或VERBOSE在其他作用域中

        try:
            if action_type == 0:  # 弃牌
                if pokers.ActionEnum.Fold in state.legal_actions:
                    return pokers.Action(pokers.ActionEnum.Fold)
                # 弃牌的备选逻辑
                if pokers.ActionEnum.Check in state.legal_actions:
                    return pokers.Action(pokers.ActionEnum.Check)
                if pokers.ActionEnum.Call in state.legal_actions:
                    return pokers.Action(pokers.ActionEnum.Call)
                if VERBOSE:
                    print(
                        f"DeepCFRAgent 警告: 选择了弃牌但没有其他合法备选。无论如何返回弃牌。"
                    )
                return pokers.Action(pokers.ActionEnum.Fold)  # 最后手段

            elif action_type == 1:  # 跟注
                if pokers.ActionEnum.Check in state.legal_actions:
                    return pokers.Action(pokers.ActionEnum.Check)
                elif pokers.ActionEnum.Call in state.legal_actions:
                    return pokers.Action(pokers.ActionEnum.Call)
                # 跟注的备选逻辑
                if pokers.ActionEnum.Fold in state.legal_actions:
                    return pokers.Action(pokers.ActionEnum.Fold)
                if VERBOSE:
                    print(
                        f"DeepCFRAgent 警告: 选择了跟注但两者都不合法，也不是弃牌。无论如何返回跟注。"
                    )
                return pokers.Action(pokers.ActionEnum.Check)  # 最后手段

            elif action_type == 2:  # 加注
                if pokers.ActionEnum.Raise not in state.legal_actions:
                    # 如果加注不合法，则回退
                    if VERBOSE:
                        print(
                            f"DeepCFRAgent 信息: 选择了加注 (类型2)，但加注不在合法动作中。正在回退。"
                        )
                    if pokers.ActionEnum.Call in state.legal_actions:
                        return pokers.Action(pokers.ActionEnum.Call)
                    if pokers.ActionEnum.Check in state.legal_actions:
                        return pokers.Action(pokers.ActionEnum.Check)
                    return pokers.Action(pokers.ActionEnum.Fold)

                player_state = state.players_state[state.current_player]
                current_bet = player_state.bet_chips  # 玩家本轮已投入底池的筹码
                available_stake = player_state.stake  # 玩家剩余的筹码

                call_amount = max(
                    0.0, state.min_bet - current_bet
                )  # 需要跟注的额外筹码

                min_raise_increment = 1.0
                if (
                    hasattr(state, "bb")
                    and state.bb is not None
                    and float(state.bb) > 0
                ):
                    min_raise_increment = max(1.0, float(state.bb))
                elif (
                    state.min_bet > 0
                ):  # 如果没有BB，则使用min_bet（如果它意味着加注大小）
                    # 如果BB定义不明确，这部分是有点启发式的。
                    # 想法是加注应该有一定的意义。
                    # 如果上次下注是X，min_raise_increment通常是X。
                    # 为了简单起见，我们将坚持使用小的固定最小值或BB。
                    # 更稳健的方法可能涉及查看之前的加注金额。
                    min_raise_increment = max(
                        1.0,
                        (
                            state.min_bet - current_bet
                            if state.min_bet > current_bet
                            else 1.0
                        ),
                    )

                # --- 初始检查：玩家能否进行任何有效的加注？ ---
                # 有效的加注意味着跟注，然后至少增加min_raise_increment。
                if available_stake < call_amount + min_raise_increment:
                    if VERBOSE:
                        print(
                            f"DeepCFRAgent 信息: 选择了加注 (类型2)，但无法进行有效的最小加注增量。 "
                            f"可用筹码({available_stake:.2f}) < 跟注金额({call_amount:.2f}) + 最小增量({min_raise_increment:.2f})。正在切换到跟注。"
                        )
                    if pokers.ActionEnum.Call in state.legal_actions:
                        return pokers.Action(pokers.ActionEnum.Call)
                    else:
                        if VERBOSE:
                            print(
                                f"DeepCFRAgent 警告: 无法跟注（加注检查失败后不合法），回退到弃牌。"
                            )
                        return pokers.Action(pokers.ActionEnum.Fold)
                # --- 初始检查结束 ---

                remaining_stake_after_call = available_stake - call_amount

                # 从网络的bet_size_multiplier获取目标额外加注
                pot_size = max(1.0, state.pot)

                if bet_size_multiplier is None:
                    bet_size_multiplier = 1.0  # 未提供时的默认值
                else:
                    bet_size_multiplier = float(bet_size_multiplier)
                    # 可选: self.adjust_bet_size(state, bet_size_multiplier)（如果使用）

                bet_size_multiplier = max(
                    self.min_bet_size, min(self.max_bet_size, bet_size_multiplier)
                )
                network_desired_additional_raise = pot_size * bet_size_multiplier

                # 根据网络和游戏规则确定chosen_additional_amount
                chosen_additional_amount = network_desired_additional_raise
                # 限制1: 不能超过全下（call后的剩余筹码）
                chosen_additional_amount = min(
                    chosen_additional_amount, remaining_stake_after_call
                )
                # 限制2: 必须至少增加min_raise_increment
                chosen_additional_amount = max(
                    chosen_additional_amount, min_raise_increment
                )

                # 保护措施: 如果由于限制（例如min_raise_increment > remaining_stake_after_call，
                # 如果初始检查正确，这种情况不应该发生），确保它不大于remaining_stake_after_call。
                # 如果min_raise_increment强制要求，这会使其成为全下。
                if chosen_additional_amount > remaining_stake_after_call:
                    chosen_additional_amount = remaining_stake_after_call

                # --- 开始：浮点数保护 ---
                total_chips_player_would_commit_this_turn = (
                    call_amount + chosen_additional_amount
                )
                epsilon = 0.00001  # 浮点数比较的容差

                if (
                    total_chips_player_would_commit_this_turn
                    > available_stake + epsilon
                ):
                    if VERBOSE:
                        print(
                            f"DeepCFRAgent 信息: action_type_to_pokers_action中的浮点数保护已触发。"
                        )
                        print(
                            f"  初始chosen_additional_amount: {chosen_additional_amount:.6f}"
                        )
                        print(
                            f"  总投入 ({total_chips_player_would_commit_this_turn:.6f}) > 可用筹码 ({available_stake:.6f})"
                        )

                    chosen_additional_amount = available_stake - call_amount
                    chosen_additional_amount = max(
                        0.0, chosen_additional_amount
                    )  # 确保不为负

                    if VERBOSE:
                        print(
                            f"  调整后的chosen_additional_amount: {chosen_additional_amount:.6f}"
                        )
                        print(
                            f"  新的总投入: {(call_amount + chosen_additional_amount):.6f}"
                        )
                # --- 结束：浮点数保护 ---

                # 确保在所有调整后chosen_additional_amount不为负
                chosen_additional_amount = max(0.0, chosen_additional_amount)

                if VERBOSE:
                    print(f"--- DeepCFRAgent 加注计算 (最终返回前) ---")
                    print(f"  玩家ID: {state.current_player}, 阶段: {state.stage}")
                    print(
                        f"  可用筹码: {available_stake:.6f}, 当前底池下注: {current_bet:.6f}"
                    )
                    print(
                        f"  状态最小下注(跟注): {state.min_bet:.6f}, 底池大小: {state.pot:.6f}"
                    )
                    print(f"  计算出的跟注金额: {call_amount:.6f}")
                    print(f"  最小加注增量: {min_raise_increment:.6f}")
                    print(f"  跟注后的剩余筹码: {remaining_stake_after_call:.6f}")
                    print(
                        f"  下注大小乘数 (来自网络, 原始): {float(bet_size_multiplier) if bet_size_multiplier is not None else 'N/A'}, (使用, 限制后): {bet_size_multiplier:.6f}"
                    )
                    print(
                        f"  网络期望的额外加注 (底池 * 乘数): {network_desired_additional_raise:.6f}"
                    )
                    print(
                        f"  选择的额外加注金额 (浮点数保护前): {network_desired_additional_raise:.6f} -> 规则限制为 -> {chosen_additional_amount + epsilon if total_chips_player_would_commit_this_turn > available_stake + epsilon else chosen_additional_amount:.6f}"
                    )  # 如果触发了浮点数保护，则显示保护前的值
                    print(
                        f"  最终选择的额外加注金额 (浮点数保护后): {chosen_additional_amount:.6f}"
                    )
                    _total_chips_this_action = call_amount + chosen_additional_amount
                    print(
                        f"  此动作的总筹码 (跟注 + 额外): {_total_chips_this_action:.6f}"
                    )
                    _is_exact_all_in = (
                        abs(_total_chips_this_action - available_stake) < epsilon
                    )
                    print(f"  这是精确的全下吗 (保护后)? {_is_exact_all_in}")
                    if _is_exact_all_in:
                        print(
                            f"    全下差异: {(_total_chips_this_action - available_stake):.10f}"
                        )
                    print(
                        f"--------------------------------------------------------------------"
                    )

                return pokers.Action(pokers.ActionEnum.Raise, chosen_additional_amount)

            else:  # 如果action_type是0、1或2，不应到达此处
                if VERBOSE:
                    print(f"DeepCFRAgent 错误: 未知动作类型: {action_type}")
                if pokers.ActionEnum.Call in state.legal_actions:
                    return pokers.Action(pokers.ActionEnum.Call)
                if pokers.ActionEnum.Check in state.legal_actions:
                    return pokers.Action(pokers.ActionEnum.Check)
                return pokers.Action(pokers.ActionEnum.Fold)

        except Exception as e:
            # 确保VERBOSE在此处可访问或处理其缺失
            try:
                is_verbose = VERBOSE
            except NameError:
                is_verbose = False  # 如果未在此作用域中定义VERBOSE，则使用默认值

            if is_verbose:  # 或者如果是实例属性则为self.verbose
                print(
                    f"DeepCFRAgent 在action_type_to_pokers_action中的严重错误: 玩家 {self.player_id} 的动作类型 {action_type}: {e}"
                )
                print(
                    f"  状态: current_player={state.current_player}, stage={state.stage}, legal_actions={state.legal_actions}"
                )
                if hasattr(state, "players_state") and self.player_id < len(
                    state.players_state
                ):
                    print(
                        f"  玩家 {self.player_id} 筹码: {state.players_state[self.player_id].stake}, 下注: {state.players_state[self.player_id].bet_chips}"
                    )
                else:
                    print(f"  无法访问玩家 {self.player_id} 的状态。")
                import traceback

                traceback.print_exc()

            # 回退到安全动作
            if hasattr(state, "legal_actions"):
                if pokers.ActionEnum.Call in state.legal_actions:
                    return pokers.Action(pokers.ActionEnum.Call)
                if pokers.ActionEnum.Check in state.legal_actions:
                    return pokers.Action(pokers.ActionEnum.Check)
                if pokers.ActionEnum.Fold in state.legal_actions:
                    return pokers.Action(pokers.ActionEnum.Fold)

            # 如果state.legal_actions甚至不可用或为空，则作为最后的手段
            return pokers.Action(pokers.ActionEnum.Fold)

    def adjust_bet_size(self, state, base_multiplier):
        """
        根据游戏状态动态调整下注大小乘数。

        参数:
            state: 当前扑克游戏状态
            base_multiplier: 来自模型的基础下注大小乘数

        返回:
            调整后的下注大小乘数
        """
        # 默认调整因子
        adjustment = 1.0

        # 根据游戏阶段调整
        if int(state.stage) >= 2:  # 转牌或河牌
            adjustment *= 1.2  # 在后期街道增加下注

        # 根据底池大小相对于初始筹码的比例调整
        initial_stake = state.players_state[0].stake + state.players_state[0].bet_chips
        pot_ratio = state.pot / initial_stake
        if pot_ratio > 0.5:  # 大底池
            adjustment *= 1.1  # 在大底池中下注更大
        elif pot_ratio < 0.1:  # 小底池
            adjustment *= 0.9  # 在小底池中下注更小

        # 根据位置调整（在后面位置更激进）
        btn_distance = (state.current_player - state.button) % len(state.players_state)
        if btn_distance <= 1:  # 庄位或 cutoff
            adjustment *= 1.15  # 在后面位置更激进
        elif btn_distance >= 4:  # 前面位置
            adjustment *= 0.9  # 在前面位置更保守

        # 根据活跃玩家数量调整（玩家越少，下注越大）
        active_players = sum(1 for p in state.players_state if p.active)
        if active_players <= 2:
            adjustment *= 1.2  # 一对一游戏时下注更大
        elif active_players >= 5:
            adjustment *= 0.9  # 多人游戏时下注更小

        # 将调整应用于基础乘数
        adjusted_multiplier = base_multiplier * adjustment

        # 确保我们保持在范围内
        return max(self.min_bet_size, min(self.max_bet_size, adjusted_multiplier))

    def get_legal_action_types(self, state):
        """获取当前状态的合法动作类型。"""
        legal_action_types = []

        # 如果可以 Check（免费看牌），不应该 Fold
        # 这避免了 PokerKit 的警告和潜在的状态错误
        can_check = pokers.ActionEnum.Check in state.legal_actions
        can_fold = pokers.ActionEnum.Fold in state.legal_actions

        # 检查每种动作类型
        if can_fold and not can_check:
            # 只有在不能 Check 时才允许 Fold
            legal_action_types.append(0)

        if (
            pokers.ActionEnum.Check in state.legal_actions
            or pokers.ActionEnum.Call in state.legal_actions
        ):
            legal_action_types.append(1)

        if pokers.ActionEnum.Raise in state.legal_actions:
            legal_action_types.append(2)

        return legal_action_types

    def cfr_traverse(self, state, iteration, random_agents, depth=0):
        """
        使用带有连续下注大小的外部采样MCCFR遍历游戏树。

        参数:
            iteration: 当前训练迭代
            random_agents: 对手智能体列表
            depth: 当前递归深度

        返回:
            当前玩家的期望值
        """
        # 添加递归深度保护
        max_depth = 1000
        if depth > max_depth:
            if VERBOSE:
                print(f"警告: 达到最大递归深度 ({max_depth})。返回零值。")
            return 0

        if state.final_state:
            # 返回训练智能体的收益
            return state.players_state[self.player_id].reward

        current_player = state.current_player

        # 如果是训练智能体的回合
        if current_player == self.player_id:
            legal_action_types = self.get_legal_action_types(state)

            if not legal_action_types:
                if VERBOSE:
                    print(
                        f"警告: 在深度 {depth} 处未找到玩家 {current_player} 的合法动作"
                    )
                return 0

            # 编码基础状态
            state_tensor = torch.FloatTensor(encode_state(state, self.player_id)).to(
                self.device
            )

            # 从网络获取优势值和下注大小预测
            with torch.no_grad():
                advantages, bet_size_pred = self.advantage_net(
                    state_tensor.unsqueeze(0)
                )
                advantages = advantages[0].cpu().numpy()
                bet_size_multiplier = bet_size_pred[0][0].item()

            # 使用遗憾匹配计算动作类型的策略
            advantages_masked = np.zeros(self.num_actions)
            for a in legal_action_types:
                advantages_masked[a] = max(advantages[a], 0)

            # 根据策略选择动作
            if sum(advantages_masked) > 0:
                strategy = advantages_masked / sum(advantages_masked)
            else:
                strategy = np.zeros(self.num_actions)
                for a in legal_action_types:
                    strategy[a] = 1.0 / len(legal_action_types)

            # 选择动作并遍历
            action_values = np.zeros(self.num_actions)
            for action_type in legal_action_types:
                try:
                    # 对加注动作使用预测的下注大小
                    if action_type == 2:  # 加注
                        pokers_action = self.action_type_to_pokers_action(
                            action_type, state, bet_size_multiplier
                        )
                    else:
                        pokers_action = self.action_type_to_pokers_action(
                            action_type, state
                        )

                    new_state = state.apply_action(pokers_action)

                    # 检查动作是否有效（只有 Invalid 才是错误）
                    if new_state.status == pokers.StateStatus.Invalid:
                        log_file = log_game_error(
                            state, pokers_action, f"状态错误 ({new_state.status})"
                        )
                        if STRICT_CHECKING:
                            raise ValueError(
                                f"CFR遍历期间状态状态不正常 ({new_state.status})。详情记录到 {log_file}"
                            )
                        elif VERBOSE:
                            print(
                                f"警告: 在深度 {depth} 处动作 {action_type} 无效。状态: {new_state.status}"
                            )
                            print(
                                f"玩家: {current_player}, 动作: {pokers_action.action}, 金额: {pokers_action.amount if pokers_action.action == pokers.ActionEnum.Raise else 'N/A'}"
                            )
                            print(
                                f"当前下注: {state.players_state[current_player].bet_chips}, 筹码: {state.players_state[current_player].stake}"
                            )
                            print(f"详情记录到 {log_file}")
                        continue  # 在非严格模式下跳过此动作并尝试其他动作

                    action_values[action_type] = self.cfr_traverse(
                        new_state, iteration, random_agents, depth + 1
                    )
                except Exception as e:
                    if VERBOSE:
                        print(f"动作 {action_type} 遍历中的错误: {e}")
                    action_values[action_type] = 0
                    if STRICT_CHECKING:
                        raise  # 在严格模式下重新抛出异常

            # 计算反事实遗憾并添加到记忆中
            ev = sum(strategy[a] * action_values[a] for a in legal_action_types)

            # 计算归一化因子
            max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)

            for action_type in legal_action_types:
                # 计算遗憾
                regret = action_values[action_type] - ev

                # 归一化并裁剪遗憾
                normalized_regret = regret / max_abs_val
                clipped_regret = np.clip(normalized_regret, -1.0, 1.0)  # 裁剪到[-1, 1]范围

                # Linear CFR: 使用迭代次数作为采样优先级权重，而不是放大遗憾值
                # 这样可以让后期样本更重要，但不会导致数值爆炸
                iteration_weight = np.sqrt(iteration) if iteration > 1 else 1.0

                # 以遗憾大小和迭代权重作为优先级存储在优先记忆中
                priority = (abs(clipped_regret) + 0.01) * iteration_weight

                # 对于加注动作，存储下注大小乘数
                if action_type == 2:
                    self.advantage_memory.add(
                        (
                            encode_state(state, self.player_id),
                            np.zeros(20),  # 对手特征的占位符
                            action_type,
                            bet_size_multiplier,
                            clipped_regret,
                        ),
                        priority,
                    )
                else:
                    self.advantage_memory.add(
                        (
                            encode_state(state, self.player_id),
                            np.zeros(20),  # 对手特征的占位符
                            action_type,
                            0.0,  # 非加注动作的默认下注大小
                            clipped_regret,
                        ),
                        priority,
                    )

            # 添加到策略记忆
            strategy_full = np.zeros(self.num_actions)
            for a in legal_action_types:
                strategy_full[a] = strategy[a]

            self.strategy_memory.append(
                (
                    encode_state(state, self.player_id),
                    np.zeros(20),  # 对手特征的占位符
                    strategy_full,
                    bet_size_multiplier if 2 in legal_action_types else 0.0,
                    iteration,
                )
            )

            return ev

        # 如果是其他玩家的回合（随机智能体）
        else:
            try:
                # 让随机智能体选择动作
                action = random_agents[current_player].choose_action(state)

                # 如果没有合法动作（自动化阶段或游戏结束），返回0
                if action is None:
                    if VERBOSE:
                        print(
                            f"随机智能体在深度 {depth} 处没有可用动作（可能是自动化阶段）"
                        )
                    return 0

                new_state = state.apply_action(action)

                # 检查动作是否有效（只有 Invalid 才是错误）
                if new_state.status == pokers.StateStatus.Invalid:
                    log_file = log_game_error(
                        state, action, f"状态错误 ({new_state.status})"
                    )
                    if STRICT_CHECKING:
                        raise ValueError(
                            f"随机智能体导致状态状态不正常 ({new_state.status})。详情记录到 {log_file}"
                        )
                    if VERBOSE:
                        print(
                            f"警告: 随机智能体在深度 {depth} 处做出无效动作。状态: {new_state.status}"
                        )
                        print(f"详情记录到 {log_file}")
                    return 0

                return self.cfr_traverse(new_state, iteration, random_agents, depth + 1)
            except Exception as e:
                if VERBOSE:
                    print(f"随机智能体遍历中的错误: {e}")
                if STRICT_CHECKING:
                    raise  # 在严格模式下重新抛出异常
                return 0

    def train_advantage_network(
        self, batch_size=512, epochs=5, beta_start=0.4, beta_end=1.0
    ):
        """
        使用优先经验回放训练优势网络。
        """
        if len(self.advantage_memory) < batch_size:
            return 0

        self.advantage_net.train()
        total_loss = 0

        # 计算当前重要性采样的beta值
        progress = min(1.0, self.iteration_count / 10000)
        beta = beta_start + progress * (beta_end - beta_start)

        for epoch in range(epochs):
            # 使用当前beta从优先记忆中采样批次
            batch, indices, weights = self.advantage_memory.sample(
                batch_size, beta=beta
            )
            states, opponent_features, action_types, bet_sizes, regrets = zip(*batch)

            # [DEBUG 1] 记录记忆中的遗憾值
            if self.iteration_count % 10 == 0 and epoch == 0:
                regret_array = np.array(regrets)
                print(
                    f"[DEBUG-MEMORY] 遗憾统计: min={np.min(regret_array):.2f}, max={np.max(regret_array):.2f}, mean={np.mean(regret_array):.2f}"
                )

            # 转换为张量
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            opponent_feature_tensors = torch.FloatTensor(
                np.array(opponent_features)
            ).to(self.device)
            action_type_tensors = torch.LongTensor(np.array(action_types)).to(
                self.device
            )
            bet_size_tensors = (
                torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            )
            regret_tensors = torch.FloatTensor(np.array(regrets)).to(self.device)
            weight_tensors = torch.FloatTensor(weights).to(self.device)

            # 前向传播
            action_advantages, bet_size_preds = self.advantage_net(
                state_tensors, opponent_feature_tensors
            )

            # [DEBUG 2] Log network raw outputs to identify explosion
            if self.iteration_count % 10 == 0 and epoch == 0:
                with torch.no_grad():
                    max_adv = torch.max(action_advantages).item()
                    min_adv = torch.min(action_advantages).item()
                    print(
                        f"[DEBUG-NETWORK] 网络输出: min={min_adv:.2f}, max={max_adv:.2f}"
                    )

            # 计算动作类型损失（适用于所有动作）
            predicted_regrets = action_advantages.gather(
                1, action_type_tensors.unsqueeze(1)
            ).squeeze(1)

            # [DEBUG 3] 记录收集的预测值
            if self.iteration_count % 10 == 0 and epoch == 0:
                with torch.no_grad():
                    max_pred = torch.max(predicted_regrets).item()
                    min_pred = torch.min(predicted_regrets).item()
                    max_target = torch.max(regret_tensors).item()
                    min_target = torch.min(regret_tensors).item()
                    print(
                        f"[DEBUG-PRED] 预测值: min={min_pred:.2f}, max={max_pred:.2f}"
                    )
                    print(
                        f"[DEBUG-TARGET] 目标值: min={min_target:.2f}, max={max_target:.2f}"
                    )

            action_loss = F.smooth_l1_loss(
                predicted_regrets, regret_tensors, reduction="none"
            )

            # [DEBUG 4] 记录加权前的原始损失值
            if self.iteration_count % 10 == 0 and epoch == 0:
                with torch.no_grad():
                    max_loss = torch.max(action_loss).item()
                    mean_loss = torch.mean(action_loss).item()
                    print(
                        f"[DEBUG-LOSS] 原始损失值: max={max_loss:.2f}, mean={mean_loss:.2f}"
                    )

            weighted_action_loss = (action_loss * weight_tensors).mean()

            # [DEBUG 5] 记录加权损失
            if self.iteration_count % 10 == 0 and epoch == 0:
                print(
                    f"[DEBUG-WEIGHTED] 加权动作损失: {weighted_action_loss.item():.2f}"
                )

                # 检查权重异常值
                max_weight = torch.max(weight_tensors).item()
                min_weight = torch.min(weight_tensors).item()
                print(
                    f"[DEBUG-WEIGHTS] 权重范围: min={min_weight:.4f}, max={max_weight:.4f}"
                )

            # 计算下注大小损失（仅适用于加注动作）
            raise_mask = action_type_tensors == 2
            if torch.any(raise_mask):
                # 计算所有下注大小的损失
                all_bet_losses = F.smooth_l1_loss(
                    bet_size_preds, bet_size_tensors, reduction="none"
                )

                # 仅计算加注动作的损失，其他动作设为零
                masked_bet_losses = all_bet_losses * raise_mask.float().unsqueeze(1)

                # 计算加权平均损失
                raise_count = raise_mask.sum().item()
                if raise_count > 0:
                    weighted_bet_size_loss = (
                        masked_bet_losses.squeeze() * weight_tensors
                    ).sum() / raise_count
                    combined_loss = weighted_action_loss + 0.5 * weighted_bet_size_loss

                    # [DEBUG 6] 记录下注大小损失
                    if self.iteration_count % 10 == 0 and epoch == 0:
                        print(
                            f"[DEBUG-BET] 加权下注大小损失: {weighted_bet_size_loss.item():.2f}"
                        )
                else:
                    combined_loss = weighted_action_loss
            else:
                combined_loss = weighted_action_loss

            # [DEBUG 7] 记录最终组合损失
            if self.iteration_count % 10 == 0 and epoch == 0:
                print(f"[DEBUG-COMBINED] 裁剪前的组合损失: {combined_loss.item():.2f}")

            # 反向传播并优化
            self.optimizer.zero_grad()
            combined_loss.backward()

            # [DEBUG 8] 检查裁剪前的梯度爆炸
            if self.iteration_count % 10 == 0 and epoch == 0:
                total_grad_norm = 0
                max_layer_norm = 0
                max_layer_name = ""
                for name, param in self.advantage_net.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm * grad_norm
                        if grad_norm > max_layer_norm:
                            max_layer_norm = grad_norm
                            max_layer_name = name

                total_grad_norm = np.sqrt(total_grad_norm)
                print(f"[DEBUG-GRAD] 裁剪前 - 总梯度范数: {total_grad_norm:.2f}")
                print(
                    f"[DEBUG-GRAD] 最大层梯度: {max_layer_name} = {max_layer_norm:.2f}"
                )

            # 应用梯度裁剪（现有代码）
            torch.nn.utils.clip_grad_norm_(
                self.advantage_net.parameters(), max_norm=0.5
            )

            # [DEBUG 9] 检查梯度裁剪效果
            if self.iteration_count % 10 == 0 and epoch == 0:
                total_grad_norm = 0
                for name, param in self.advantage_net.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm * grad_norm

                total_grad_norm = np.sqrt(total_grad_norm)
                print(f"[DEBUG-GRAD] 裁剪后 - 总梯度范数: {total_grad_norm:.2f}")

            self.optimizer.step()

            # [DEBUG 10] 检查更新后的极端参数值
            if self.iteration_count % 50 == 0 and epoch == 0:
                with torch.no_grad():
                    max_param_val = -float("inf")
                    max_param_name = ""
                    for name, param in self.advantage_net.named_parameters():
                        param_max = torch.max(torch.abs(param)).item()
                        if param_max > max_param_val:
                            max_param_val = param_max
                            max_param_name = name

                    print(
                        f"[DEBUG-PARAMS] 最大参数值: {max_param_name} = {max_param_val:.2f}"
                    )

            # 更新优先级
            with torch.no_grad():
                # 计算优先级更新的新误差
                new_action_errors = F.smooth_l1_loss(
                    predicted_regrets, regret_tensors, reduction="none"
                )

                # 对于加注动作，包括下注大小误差
                if torch.any(raise_mask):
                    # 计算每个样本的归一化下注大小误差
                    new_bet_errors = torch.zeros_like(new_action_errors)

                    # 仅为加注动作添加下注大小误差
                    raise_indices = torch.where(raise_mask)[0]
                    for i in raise_indices:
                        new_bet_errors[i] = F.smooth_l1_loss(
                            bet_size_preds[i], bet_size_tensors[i], reduction="mean"
                        )

                    # 组合误差，下注大小使用较小权重
                    combined_errors = new_action_errors + 0.5 * new_bet_errors
                else:
                    combined_errors = new_action_errors

                # [DEBUG 11] 检查优先级值
                if self.iteration_count % 10 == 0 and epoch == 0:
                    combined_errors_np = combined_errors.cpu().numpy()
                    max_priority = np.max(combined_errors_np) + 0.01
                    min_priority = np.min(combined_errors_np) + 0.01
                    mean_priority = np.mean(combined_errors_np) + 0.01
                    print(
                        f"[DEBUG-PRIORITY] 优先级: min={min_priority:.2f}, max={max_priority:.2f}, mean={mean_priority:.2f}"
                    )

                # 更新优先级（现有代码）
                combined_errors_np = combined_errors.cpu().numpy()
                for i, idx in enumerate(indices):
                    self.advantage_memory.update_priority(
                        idx, combined_errors_np[i] + 0.01
                    )

            total_loss += combined_loss.item()

        # 返回平均损失
        return total_loss / epochs

    def train_strategy_network(self, batch_size=512, epochs=5):
        """
        使用收集的样本训练策略网络。

        根据 Deep CFR 论文使用 MSE 损失和 Linear CFR 权重。

        参数:
            batch_size: 训练批次大小
            epochs: 每次调用的训练轮数

        返回:
            平均训练损失
        """
        if len(self.strategy_memory) < batch_size:
            return 0

        self.strategy_net.train()
        total_loss = 0

        for _ in range(epochs):
            # 从记忆中采样批次
            batch = random.sample(self.strategy_memory, batch_size)
            states, _, strategies, bet_sizes, iterations = zip(*batch)

            # 转换为张量
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            strategy_tensors = torch.FloatTensor(np.array(strategies)).to(self.device)
            bet_size_tensors = (
                torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            )
            iteration_tensors = (
                torch.FloatTensor(iterations).to(self.device).unsqueeze(1)
            )

            # Linear CFR 权重: 按迭代次数加权，归一化
            weights = iteration_tensors / torch.sum(iteration_tensors)

            # 前向传播
            action_logits, bet_size_preds = self.strategy_net(state_tensors)
            predicted_strategies = F.softmax(action_logits, dim=1)

            # 使用 MSE 损失（Deep CFR 论文推荐）
            # 计算每个样本的 MSE
            per_sample_mse = torch.sum((strategy_tensors - predicted_strategies) ** 2, dim=1)
            # 加权平均
            action_loss = torch.sum(weights.squeeze() * per_sample_mse)

            # 下注大小损失（仅适用于有加注动作的状态）
            raise_mask = strategy_tensors[:, 2] > 0
            if raise_mask.sum() > 0:
                raise_indices = torch.nonzero(raise_mask).squeeze(1)
                raise_bet_preds = bet_size_preds[raise_indices]
                raise_bet_targets = bet_size_tensors[raise_indices]
                raise_weights = weights[raise_indices]

                # 对下注大小使用MSE损失
                bet_size_loss = (raise_bet_preds - raise_bet_targets) ** 2
                weighted_bet_size_loss = torch.sum(
                    raise_weights * bet_size_loss.squeeze()
                )

                # 组合损失
                combined_loss = action_loss + 0.1 * weighted_bet_size_loss
            else:
                combined_loss = action_loss

            # 反向传播并优化
            self.strategy_optimizer.zero_grad()
            combined_loss.backward()

            # 应用梯度裁剪（论文推荐 max_norm=1.0）
            torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), max_norm=1.0)

            self.strategy_optimizer.step()

            total_loss += combined_loss.item()

        # 返回平均损失
        return total_loss / epochs

    def choose_action(self, state):
        """在实际游戏中为给定状态选择动作。"""
        legal_action_types = self.get_legal_action_types(state)

        # 如果没有合法动作（自动化阶段或游戏结束），返回 None
        if not legal_action_types:
            return None

        state_tensor = (
            torch.FloatTensor(encode_state(state, self.player_id))
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            logits, bet_size_pred = self.strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            bet_size_multiplier = bet_size_pred[0][0].item()

        # 过滤仅合法动作
        legal_probs = np.array([probs[a] for a in legal_action_types])
        if np.sum(legal_probs) > 0:
            legal_probs = legal_probs / np.sum(legal_probs)
        else:
            legal_probs = np.ones(len(legal_action_types)) / len(legal_action_types)

        # 根据概率选择动作
        action_idx = np.random.choice(len(legal_action_types), p=legal_probs)
        action_type = legal_action_types[action_idx]

        # 对加注动作使用预测的下注大小
        if action_type == 2:  # 加注
            return self.action_type_to_pokers_action(
                action_type, state, bet_size_multiplier
            )
        else:
            return self.action_type_to_pokers_action(action_type, state)

    def save_model(self, path_prefix):
        """将模型保存到磁盘。"""
        torch.save(
            {
                "iteration": self.iteration_count,
                "advantage_net": self.advantage_net.state_dict(),
                "strategy_net": self.strategy_net.state_dict(),
                "min_bet_size": self.min_bet_size,
                "max_bet_size": self.max_bet_size,
            },
            f"{path_prefix}_iteration_{self.iteration_count}.pt",
        )

    def load_model(self, path):
        """从磁盘加载模型。"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.iteration_count = checkpoint["iteration"]
        self.advantage_net.load_state_dict(checkpoint["advantage_net"])
        self.strategy_net.load_state_dict(checkpoint["strategy_net"])

        # 如果检查点中可用，则加载下注大小边界
        if "min_bet_size" in checkpoint:
            self.min_bet_size = checkpoint["min_bet_size"]
        if "max_bet_size" in checkpoint:
            self.max_bet_size = checkpoint["max_bet_size"]
