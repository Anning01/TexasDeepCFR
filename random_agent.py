import random
import pokerkit_adapter as pokers

from game_logger import log_game_error
from settings import STRICT_CHECKING


class RandomAgent:
    """
    简单的扑克随机智能体，选择随机的合法操作
    并确保有效的下注大小，尤其是在加注与跟注之间。
    """

    def __init__(self, player_id):
        self.player_id = player_id
        self.name = f"RandomAgent_{player_id}"  # 为了清晰起见添加名称

    def choose_action(self, state):
        """选择具有正确计算的下注大小的随机合法操作。"""
        # 如果没有合法动作，返回 None 或跳过
        # 这种情况发生在自动化阶段（发牌等）或游戏结束时
        if not state.legal_actions:
            # 不再尝试弃牌，直接返回 None
            # 调用方应该检查并跳过这个玩家的动作
            return None

        # 如果可以 Check（免费看牌），移除 Fold 选项
        # 避免在可以免费看牌时弃牌（PokerKit 会警告并可能导致状态错误）
        available_actions = list(state.legal_actions)
        if pokers.ActionEnum.Check in available_actions and pokers.ActionEnum.Fold in available_actions:
            available_actions.remove(pokers.ActionEnum.Fold)

        # 从可用操作中选择一个随机的合法操作类型
        action_enum = random.choice(available_actions)

        # 首先处理非加注操作
        if action_enum == pokers.ActionEnum.Fold:
            return pokers.Action(action_enum)
        elif action_enum == pokers.ActionEnum.Check:
            return pokers.Action(action_enum)
        elif action_enum == pokers.ActionEnum.Call:
            return pokers.Action(action_enum)

        # 处理加注操作
        elif action_enum == pokers.ActionEnum.Raise:
            player_state = state.players_state[state.current_player]
            current_bet = player_state.bet_chips
            available_stake = player_state.stake

            # 计算跟注金额（需要匹配当前最小下注）
            call_amount = max(0, state.min_bet - current_bet)

            # 检查玩家是否真的可以进行超出跟注金额的有效加注。
            # 加注需要投入*超过*跟注金额的筹码。
            # 加注的最小*额外*金额通常是大盲注或1个筹码。
            min_raise_increment = 1.0  # 一个小的默认最小增量
            # 尝试获取大盲注以获得更标准的最小加注大小
            if (
                state.min_bet > 0 and state.pot > 0
            ):  # 启发式：大盲注可能与初始下注/底池有关
                # 找到可能的大盲注大小（通常在翻牌前，当底池小于等于3倍最小下注时，state.min_bet就是大盲注）
                # 这并不完美，如果可能的话，可能需要将直接的大盲注值传递给state
                likely_bb = (
                    state.min_bet
                    if state.stage == pokers.Stage.Preflop
                    and state.pot <= 3 * state.min_bet
                    else state.min_bet / 2
                )
                min_raise_increment = max(1.0, likely_bb)
            elif hasattr(state, "bb"):  # 如果bb明确可用
                min_raise_increment = max(1.0, state.bb)

            if available_stake <= call_amount + min_raise_increment:
                # 玩家无法进行有效的加注（除了跟注金额外没有足够的筹码）
                # 或者玩家只是为了满足跟注金额而全下。
                # 这个操作应该被视为跟注，而不是加注。
                # print(f"RandomAgent: 选择了加注，但无法进行有效加注。筹码={available_stake}, 跟注金额={call_amount}, 最小增量={min_raise_increment}。切换到跟注。")  # 可选调试
                # 在返回跟注之前确保跟注是合法的
                if pokers.ActionEnum.Call in state.legal_actions:
                    return pokers.Action(pokers.ActionEnum.Call)
                else:
                    # 后备方案：如果跟注不合法（例如，已经全下匹配下注），则弃牌。
                    # print(f"RandomAgent 警告：无法跟注（不合法），回退到弃牌。")
                    # 如果可能的话，确保弃牌是合法的
                    if pokers.ActionEnum.Fold in state.legal_actions:
                        return pokers.Action(pokers.ActionEnum.Fold)
                    else:
                        # 最后手段，如果弃牌也不合法（极不可能）
                        # print(f"RandomAgent 严重警告：无法跟注或弃牌！")
                        # 无论如何都返回跟注，让Rust处理错误状态
                        return pokers.Action(pokers.ActionEnum.Call)

            # 如果我们到达这里，说明可以进行有效的加注。
            remaining_stake_after_call = available_stake - call_amount

            # 定义潜在的加注金额（作为跟注后的*额外*筹码）
            # 确保金额至少为最小增量，不超过剩余筹码
            valid_additional_amounts = []

            # 最小可能的加注增量
            valid_additional_amounts.append(min_raise_increment)

            # 半池加注（额外金额）
            half_pot_additional = max(state.pot * 0.5, min_raise_increment)
            if half_pot_additional <= remaining_stake_after_call:
                valid_additional_amounts.append(half_pot_additional)

            # 全池加注（额外金额）
            full_pot_additional = max(state.pot, min_raise_increment)
            if full_pot_additional <= remaining_stake_after_call:
                valid_additional_amounts.append(full_pot_additional)

            # 全下加注（额外金额是跟注后的剩余筹码）
            # 只有当它严格大于已存在的其他选项并且满足最小增量要求时才添加。
            if (
                remaining_stake_after_call >= min_raise_increment
                and remaining_stake_after_call not in valid_additional_amounts
            ):
                valid_additional_amounts.append(remaining_stake_after_call)

            # 再次过滤金额以确保安全（应该是多余的）
            possible_additional_amounts = [
                amount
                for amount in valid_additional_amounts
                if min_raise_increment <= amount <= remaining_stake_after_call
            ]

            # 如果不知何故没有有效的加注金额（初始检查应该已经覆盖了这种情况），则跟注。
            if not possible_additional_amounts:
                print(
                    f"RandomAgent 警告：过滤后没有找到有效的额外加注金额。回退到跟注。"
                )
                if pokers.ActionEnum.Call in state.legal_actions:
                    return pokers.Action(pokers.ActionEnum.Call)
                else:  # 如果跟注不合法，回退到弃牌
                    if pokers.ActionEnum.Fold in state.legal_actions:
                        return pokers.Action(pokers.ActionEnum.Fold)
                    else:  # 最后手段
                        return pokers.Action(pokers.ActionEnum.Call)

            # 从有效选项中选择一个随机的*额外*加注金额
            additional_raise = random.choice(possible_additional_amounts)

            # 创建最终的加注操作
            action = pokers.Action(action_enum, additional_raise)

            # 可选：严格检查（如果启用）
            if STRICT_CHECKING:
                # 临时应用操作以检查Rust状态
                test_state = state.apply_action(action)
                if test_state.status == pokers.StateStatus.Invalid:
                    log_file = log_game_error(
                        state,
                        action,
                        f"Random agent created invalid action: {test_state.status}",
                    )
                    # 如果生成的加注无效，则回退到跟注
                    print(
                        f"RandomAgent 严格检查失败：无效的加注({additional_raise})。状态：{test_state.status}。回退到跟注。日志：{log_file}"
                    )
                    if pokers.ActionEnum.Call in state.legal_actions:
                        return pokers.Action(pokers.ActionEnum.Call)
                    else:  # 如果跟注不合法，回退到弃牌
                        if pokers.ActionEnum.Fold in state.legal_actions:
                            return pokers.Action(pokers.ActionEnum.Fold)
                        else:  # 最后手段
                            return pokers.Action(pokers.ActionEnum.Call)

            return action
        else:
            # 如果action_enum来自legal_actions，这不应该发生
            print(f"警告：RandomAgent遇到意外的操作枚举：{action_enum}。回退到弃牌。")
            if pokers.ActionEnum.Fold in state.legal_actions:
                return pokers.Action(pokers.ActionEnum.Fold)
            else:  # 最后手段
                return pokers.Action(pokers.ActionEnum.Check)  # 如果弃牌不合法，则选择过牌  # 如果弃牌不合法，则选择过牌
