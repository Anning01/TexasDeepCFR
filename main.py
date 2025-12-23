import argparse
import glob
import os
import random

import pokerkit_adapter as pokers
import torch

from core.deepcfr import DeepCFRAgent
from core.model import set_verbose
from game_logger import log_game_error
from settings import STRICT_CHECKING, set_strict_checking


def get_action_description(action):
    """将扑克操作转换为人类可读的字符串。"""
    if action.action == pokers.ActionEnum.Fold:
        return "弃牌"
    elif action.action == pokers.ActionEnum.Check:
        return "过牌"
    elif action.action == pokers.ActionEnum.Call:
        return "跟注"
    elif action.action == pokers.ActionEnum.Raise:
        return f"加注到 {action.amount:.2f}"
    else:
        return f"未知操作: {action.action}"


def card_to_string(card):
    """将扑克牌转换为可读字符串。"""
    suits = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
    ranks = {
        0: "2",
        1: "3",
        2: "4",
        3: "5",
        4: "6",
        5: "7",
        6: "8",
        7: "9",
        8: "10",
        9: "J",
        10: "Q",
        11: "K",
        12: "A",
    }

    return f"{ranks[int(card.rank)]}{suits[int(card.suit)]}"


def display_game_state(state, player_id=0):
    """以人类可读的格式显示当前游戏状态。"""
    print("\n" + "=" * 70)

    # 修复Stage枚举 - 正确转换为字符串
    stage_names = {0: "翻牌前", 1: "翻牌", 2: "转牌", 3: "河牌", 4: "摊牌"}
    stage_name = stage_names.get(int(state.stage), str(state.stage))
    print(f"阶段: {stage_name}")

    print(f"底池: ${state.pot:.2f}")
    print(f"按钮位置: 玩家 {state.button}")

    # 显示公共牌
    community_cards = " ".join([card_to_string(card) for card in state.public_cards])
    print(f"公共牌: {community_cards if community_cards else '无'}")

    # 显示玩家的手牌
    hand = " ".join(
        [card_to_string(card) for card in state.players_state[player_id].hand]
    )
    print(f"你的手牌: {hand}")

    # 显示所有玩家的状态
    print("\n玩家列表:")
    for i, p in enumerate(state.players_state):
        status = "你" if i == player_id else "AI"
        active = "活跃" if p.active else "已弃牌"
        print(
            f"玩家 {i} ({status}): ${p.stake:.2f} - 下注: ${p.bet_chips:.2f} - {active}"
        )

    # 如果是人类玩家的回合，显示合法操作
    if state.current_player == player_id:
        print("\n合法操作:")
        for action_enum in state.legal_actions:
            if action_enum == pokers.ActionEnum.Fold:
                print("  f: 弃牌")
            elif action_enum == pokers.ActionEnum.Check:
                print("  c: 过牌")
            elif action_enum == pokers.ActionEnum.Call:
                # 计算跟注金额
                call_amount = max(
                    0, state.min_bet - state.players_state[player_id].bet_chips
                )
                print(f"  c: 跟注 ${call_amount:.2f}")
            elif action_enum == pokers.ActionEnum.Raise:
                min_raise = state.min_bet
                max_raise = state.players_state[player_id].stake
                print(f"  r: 加注 (最小: ${min_raise:.2f}, 最大: ${max_raise:.2f})")
                print("    h: 半池加注")
                print("    p: 全池加注")
                print("    m: 自定义加注金额")

    print("=" * 70)


def get_human_action(state, player_id=0):
    """通过控制台输入获取人类玩家的操作。"""
    while True:
        action_input = (
            input("你的操作 (f=弃牌, c=过牌/跟注, r=加注, h=半池, p=全池, m=自定义): ")
            .strip()
            .lower()
        )

        # 处理弃牌
        if action_input == "f" and pokers.ActionEnum.Fold in state.legal_actions:
            return pokers.Action(pokers.ActionEnum.Fold)

        # 处理过牌/跟注
        elif action_input == "c":
            if pokers.ActionEnum.Check in state.legal_actions:
                return pokers.Action(pokers.ActionEnum.Check)
            elif pokers.ActionEnum.Call in state.legal_actions:
                return pokers.Action(pokers.ActionEnum.Call)

        # 处理加注快捷方式
        elif (
            action_input in ["r", "h", "p", "m"]
            and pokers.ActionEnum.Raise in state.legal_actions
        ):
            player_state = state.players_state[player_id]
            current_bet = player_state.bet_chips
            available_stake = player_state.stake

            # 计算跟注金额
            call_amount = state.min_bet - current_bet

            # 如果玩家甚至无法跟注，则全下
            if available_stake <= call_amount:
                return pokers.Action(pokers.ActionEnum.Raise, available_stake)

            # 计算跟注后的剩余筹码
            remaining_stake = available_stake - call_amount

            # 如果玩家跟注后无法加注，就只跟注
            if remaining_stake <= 0:
                print("你没有足够的筹码进行加注。改为跟注。")
                return pokers.Action(pokers.ActionEnum.Call)

            # 定义最小加注（通常是1个筹码或大盲注）
            min_raise = 1.0
            if hasattr(state, "bb"):
                min_raise = state.bb

            if action_input == "h":  # 半池
                bet_amount = max(state.pot * 0.5, min_raise)  # 确保最小加注
                bet_amount = min(bet_amount, remaining_stake)  # 确保不超过筹码

                # 如果无法满足最小加注，回退到跟注
                if bet_amount < min_raise:
                    print(f"无法达到最小加注要求。改为跟注。")
                    return pokers.Action(pokers.ActionEnum.Call)

                return pokers.Action(pokers.ActionEnum.Raise, bet_amount)

            elif action_input == "p":  # 全池
                bet_amount = max(state.pot, min_raise)  # 确保最小加注
                bet_amount = min(bet_amount, remaining_stake)  # 确保不超过筹码

                # 如果无法满足最小加注，回退到跟注
                if bet_amount < min_raise:
                    print(f"无法达到最小加注要求。改为跟注。")
                    return pokers.Action(pokers.ActionEnum.Call)

                return pokers.Action(pokers.ActionEnum.Raise, bet_amount)

            elif action_input == "m" or action_input == "r":  # 自定义金额
                while True:
                    try:
                        amount_str = input(
                            f"输入加注金额 (最小: {min_raise:.2f}, 最大: {remaining_stake:.2f}): "
                        )
                        amount = float(amount_str)

                        # 检查金额是否满足最小加注
                        if amount >= min_raise and amount <= remaining_stake:
                            return pokers.Action(pokers.ActionEnum.Raise, amount)
                        else:
                            print(
                                f"金额必须在 {min_raise:.2f} 到 {remaining_stake:.2f} 之间"
                            )
                    except ValueError:
                        print("请输入有效的数字")

        print("无效操作。请重试。")


def select_random_models(models_dir, num_models=5, model_pattern="*.pt"):
    """
    从目录中选择随机的模型检查点文件。

    参数:
        models_dir: 包含模型检查点文件的目录
        num_models: 要选择的模型数量
        model_pattern: 匹配模型文件的文件模式

    返回:
        选中的模型文件路径列表
    """
    # 获取目录中所有模型检查点文件
    model_files = glob.glob(os.path.join(models_dir, model_pattern))

    if not model_files:
        print(f"没有找到匹配模式 '{model_pattern}' 的模型文件")
        return []

    # 选择随机模型
    selected_models = random.sample(model_files, min(num_models, len(model_files)))
    return selected_models


def play_against_models(
    models_dir=None,
    model_pattern="*.pt",
    num_models=5,
    player_position=0,
    initial_stake=200.0,
    small_blind=1.0,
    big_blind=2.0,
    verbose=False,
    shuffle_models=True,
):
    """
    与从目录中随机选择的AI模型对战。

    参数:
        models_dir: 包含模型检查点文件的目录
        model_pattern: 匹配模型文件的文件模式
        num_models: 要选择的模型数量
        player_position: 人类玩家的位置 (0-5)
        initial_stake: 所有玩家的初始筹码数
        small_blind: 小盲注金额
        big_blind: 大盲注金额
        verbose: 是否显示详细输出
        shuffle_models: 是否为每场游戏选择新的随机模型
    """
    set_verbose(verbose)

    # 检查模型目录是否存在
    if models_dir and not os.path.isdir(models_dir):
        print(f"警告: 模型目录 {models_dir} 未找到。")
        models_dir = None

    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 跟踪游戏统计数据
    num_games = 0
    total_profit = 0
    player_stake = initial_stake

    # 主游戏循环
    while True:
        if player_stake <= 0:
            print("\n你没有筹码了！游戏结束。")
            break

        # 在第一场游戏后询问玩家是否继续
        if num_games > 0:
            choice = input("\n继续游戏？(y/n): ").strip().lower()
            if choice != "y":
                print("感谢游玩！")
                break

        # 如果启用了洗牌或第一场游戏，为此游戏选择新的随机模型
        if (shuffle_models or num_games == 0) and models_dir:
            model_paths = select_random_models(models_dir, num_models, model_pattern)
            print(f"为本场游戏选择了 {len(model_paths)} 个随机模型:")
            for i, path in enumerate(model_paths):
                print(f"  模型 {i + 1}: {os.path.basename(path)}")
        elif not models_dir:
            model_paths = []
            print("未指定模型目录，使用随机智能体。")

        # 为每个位置创建智能体
        agents = []
        for i in range(6):
            if i == player_position:
                # 人类玩家
                agents.append(None)
            else:
                # 确定使用哪个模型
                model_idx = (i - 1) if i > player_position else i
                if models_dir and model_idx < len(model_paths):
                    # 加载模型
                    try:
                        agent = DeepCFRAgent(player_id=i, num_players=6, device=device)
                        agent.load_model(model_paths[model_idx])
                        agents.append(agent)
                        print(
                            f"为玩家 {i} 加载了模型: {os.path.basename(model_paths[model_idx])}"
                        )
                    except Exception as e:
                        print(f"为玩家 {i} 加载模型时出错: {e}")
                        print("改为使用随机智能体")
                        agents.append(RandomAgent(i))
                else:
                    # 使用随机智能体
                    agents.append(RandomAgent(i))
                    print(f"为玩家 {i} 使用随机智能体")

        num_games += 1
        print(f"\n--- 游戏 {num_games} ---")
        print(f"你当前的余额: ${player_stake:.2f}")

        # 为了公平起见，旋转按钮位置
        button_pos = (num_games - 1) % 6

        # 创建一个新的扑克游戏
        state = pokers.State.from_seed(
            n_players=6,
            button=button_pos,
            sb=small_blind,
            bb=big_blind,
            stake=initial_stake,
            seed=random.randint(0, 10000),
        )

        # 游戏直到结束
        while not state.final_state:
            current_player = state.current_player

            # 在人类行动前显示游戏状态
            if current_player == player_position:
                display_game_state(state, player_position)
                action = get_human_action(state, player_position)
                print(f"你选择了: {get_action_description(action)}")
            else:
                # AI回合的简化状态显示
                print(f"\n玩家 {current_player} 的回合")
                action = agents[current_player].choose_action(state)
                print(f"玩家 {current_player} 选择了: {get_action_description(action)}")

            # 应用操作
            new_state = state.apply_action(action)
            if new_state.status != pokers.StateStatus.Ok:
                log_file = log_game_error(
                    state, action, f"状态状态不是OK ({new_state.status})"
                )
                if STRICT_CHECKING:
                    raise ValueError(
                        f"状态状态不是OK ({new_state.status})。详细信息记录到 {log_file}"
                    )
                else:
                    print(
                        f"警告: 状态状态不是OK ({new_state.status})。详细信息记录到 {log_file}"
                    )
                    break  # 在非严格模式下跳过此游戏

            state = new_state

        # 游戏结束，显示结果
        print("\n--- 游戏结束 ---")

        # 显示所有玩家的手牌
        print("最终手牌:")
        for i, p in enumerate(state.players_state):
            if p.active:
                # 检查hand属性是否存在且有牌
                if hasattr(p, "hand") and p.hand:
                    hand = " ".join([card_to_string(card) for card in p.hand])
                    print(f"玩家 {i}: {hand}")
                else:
                    print(f"玩家 {i}: 手牌数据不可用")

        # 显示公共牌
        community_cards = " ".join(
            [card_to_string(card) for card in state.public_cards]
        )
        print(f"公共牌: {community_cards}")

        # 显示结果
        print("\n结果:")
        for i, p in enumerate(state.players_state):
            player_type = "你" if i == player_position else "AI"
            print(f"玩家 {i} ({player_type}): ${p.reward:.2f}")

        # 更新玩家的筹码
        game_profit = state.players_state[player_position].reward
        total_profit += game_profit
        player_stake += game_profit

        print(
            f"\n本场游戏: {'赢' if game_profit > 0 else '输'} ${abs(game_profit):.2f}"
        )
        print(f"累计总盈亏: ${total_profit:.2f}")
        print(f"当前余额: ${player_stake:.2f}")

    # 显示总体统计数据
    print("\n--- 总体统计数据 ---")
    print(f"游戏场次: {num_games}")
    print(f"累计总盈亏: ${total_profit:.2f}")
    print(f"每场游戏平均盈亏: ${total_profit / num_games if num_games > 0 else 0:.2f}")
    print(f"最终余额: ${player_stake:.2f}")


class RandomAgent:
    """简单的扑克随机智能体，确保有效的下注大小。"""

    def __init__(self, player_id):
        self.player_id = player_id

    def choose_action(self, state):
        """选择具有正确计算的下注大小的随机合法操作。"""
        if not state.legal_actions:
            raise ValueError(f"玩家 {self.player_id} 没有可用的合法操作")

        # 选择一个随机的合法操作
        action_enum = random.choice(state.legal_actions)

        # 对于弃牌、过牌和跟注，不需要金额
        if action_enum == pokers.ActionEnum.Fold:
            return pokers.Action(action_enum)
        elif action_enum == pokers.ActionEnum.Check:
            return pokers.Action(action_enum)
        elif action_enum == pokers.ActionEnum.Call:
            return pokers.Action(action_enum)
        # 对于加注，仔细计算有效金额
        elif action_enum == pokers.ActionEnum.Raise:
            player_state = state.players_state[state.current_player]
            current_bet = player_state.bet_chips
            available_stake = player_state.stake

            # 计算跟注金额（需要匹配当前最小下注）
            call_amount = max(0, state.min_bet - current_bet)

            # 如果玩家甚至无法跟注，则全下
            if available_stake <= call_amount:
                return pokers.Action(action_enum, available_stake)

            # 计算跟注后的剩余筹码
            remaining_stake = available_stake - call_amount

            # 如果玩家根本无法加注，就只跟注
            if remaining_stake <= 0:
                return pokers.Action(pokers.ActionEnum.Call)

            # 定义最小加注（通常是1个筹码或大盲注）
            min_raise = 1.0
            if hasattr(state, "bb"):
                min_raise = state.bb

            # 计算潜在的额外加注金额
            half_pot_raise = max(state.pot * 0.5, min_raise)
            full_pot_raise = max(state.pot, min_raise)

            # 创建有效的额外加注金额列表
            valid_amounts = []

            # 如果负担得起，添加半池
            if half_pot_raise <= remaining_stake:
                valid_amounts.append(half_pot_raise)

            # 如果负担得起，添加全池
            if full_pot_raise <= remaining_stake:
                valid_amounts.append(full_pot_raise)

            # 如果以上都负担不起，添加最小加注
            if not valid_amounts and min_raise <= remaining_stake:
                valid_amounts.append(min_raise)

            # 小概率全下
            if random.random() < 0.05 and remaining_stake > 0:  # 5%的概率
                valid_amounts.append(remaining_stake)

            # 如果无法负担任何有效的加注，回退到跟注
            if not valid_amounts:
                return pokers.Action(pokers.ActionEnum.Call)

            # 选择一个随机的额外加注金额
            additional_raise = random.choice(valid_amounts)

            # 确保不超过可用筹码
            additional_raise = min(additional_raise, remaining_stake)

            return pokers.Action(action_enum, additional_raise)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="与随机AI模型玩扑克")
    parser.add_argument(
        "--models-dir", type=str, default="../models", help="包含模型检查点文件的目录"
    )
    parser.add_argument(
        "--model-pattern", type=str, default="*.pt", help="匹配模型文件的文件模式"
    )
    parser.add_argument("--num-models", type=int, default=5, help="要选择的模型数量")
    parser.add_argument("--position", type=int, default=0, help="你在桌上的位置 (0-5)")
    parser.add_argument("--stake", type=float, default=200.0, help="初始筹码")
    parser.add_argument("--sb", type=float, default=1.0, help="小盲注金额")
    parser.add_argument("--bb", type=float, default=2.0, help="大盲注金额")
    parser.add_argument("--verbose", action="store_true", help="显示详细输出")
    parser.add_argument(
        "--no-shuffle", action="store_true", help="不为每场游戏选择新的随机模型"
    )
    parser.add_argument(
        "--strict", action="store_true", help="启用严格错误检查，对无效游戏状态引发异常"
    )
    args = parser.parse_args()

    set_strict_checking(args.strict)

    # 开始游戏
    play_against_models(
        models_dir=args.models_dir,
        model_pattern=args.model_pattern,
        num_models=args.num_models,
        player_position=args.position,
        initial_stake=args.stake,
        small_blind=args.sb,
        big_blind=args.bb,
        verbose=args.verbose,
        shuffle_models=not args.no_shuffle,
    )
