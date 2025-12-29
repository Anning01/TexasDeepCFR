import torch.nn as nn
import numpy as np

VERBOSE = False


def set_verbose(verbose_mode):
    """设置全局详细程度级别"""
    global VERBOSE
    VERBOSE = verbose_mode


class PokerNetwork(nn.Module):
    """具有连续下注大小功能的扑克网络。"""

    def __init__(self, input_size=500, hidden_size=256, num_actions=3):
        super().__init__()
        # 共享特征提取层
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # 动作类型预测（弃牌、过牌/跟注、加注）
        self.action_head = nn.Linear(hidden_size, num_actions)
        # 连续下注大小预测
        self.sizing_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),  # Output between 0-1
        )

    # 注意：我们接受并忽略opponent_features，因为在train.py中没有使用，但保持调用一致将有助于未来可能的交叉对战。
    def forward(self, x, opponent_features=None):
        """
        网络的前向传播。

        参数:
            x: 状态表示张量
            opponent_features: 可选的对手建模特征（在基类中忽略）

        返回:
            (动作logits, 下注大小预测)的元组
        """
        # 处理基础特征
        features = self.base(x)

        # 输出动作logits和下注大小
        action_logits = self.action_head(features)
        bet_size = 0.1 + 2.9 * self.sizing_head(features)

        return action_logits, bet_size


def encode_state(state, player_id=0):
    """
    将Poker状态转换为神经网络输入张量。

    参数:
        state: Poker状态
        player_id: 我们正在为其编码的玩家ID
    """
    encoded = []
    num_players = len(state.players_state)

    # 仅在详细模式下打印调试信息
    if VERBOSE:
        print(f"编码状态: current_player={state.current_player}, stage={state.stage}")
        print(
            f"玩家状态: {[(p.player, p.stake, p.bet_chips) for p in state.players_state]}"
        )
        print(f"底池: {state.pot}")

    # 编码玩家的手牌
    hand_cards = state.players_state[player_id].hand
    hand_enc = np.zeros(52)
    for card in hand_cards:
        card_idx = int(card.suit) * 13 + int(card.rank)
        hand_enc[card_idx] = 1
    encoded.append(hand_enc)

    # 编码公共牌
    community_enc = np.zeros(52)
    for card in state.public_cards:
        card_idx = int(card.suit) * 13 + int(card.rank)
        community_enc[card_idx] = 1
    encoded.append(community_enc)

    # 编码游戏阶段
    stage_enc = np.zeros(5)  # 翻牌前、翻牌、转牌、河牌、摊牌
    stage_enc[int(state.stage)] = 1
    encoded.append(stage_enc)

    # 获取初始筹码 - 防止除以零
    initial_stake = state.players_state[0].stake
    if initial_stake <= 0:
        if VERBOSE:
            print(f"警告: 初始筹码为零或负数: {initial_stake}")
        initial_stake = 1.0  # 使用1.0作为回退值以防止除以零

    # 编码底池大小（通过初始筹码归一化）
    pot_enc = [state.pot / initial_stake]
    encoded.append(pot_enc)

    # 编码庄家位置
    button_enc = np.zeros(num_players)
    button_enc[state.button] = 1
    encoded.append(button_enc)

    # 编码当前玩家
    current_player_enc = np.zeros(num_players)
    current_player_enc[state.current_player] = 1
    encoded.append(current_player_enc)

    # 编码玩家状态
    for p in range(num_players):
        player_state = state.players_state[p]

        # 活跃状态
        active_enc = [1.0 if player_state.active else 0.0]

        # 当前下注
        bet_enc = [player_state.bet_chips / initial_stake]

        # 底池筹码（已赢取）
        pot_chips_enc = [player_state.pot_chips / initial_stake]

        # 剩余筹码
        stake_enc = [player_state.stake / initial_stake]

        encoded.append(np.concatenate([active_enc, bet_enc, pot_chips_enc, stake_enc]))

    # 编码最小下注
    min_bet_enc = [state.min_bet / initial_stake]
    encoded.append(min_bet_enc)

    # 编码合法动作
    legal_actions_enc = np.zeros(4)  # 弃牌、过牌、跟注、加注
    for action_enum in state.legal_actions:
        legal_actions_enc[int(action_enum)] = 1
    encoded.append(legal_actions_enc)

    # 编码前一个动作（如果可用）
    prev_action_enc = np.zeros(4 + 1)  # 动作类型 + 归一化金额
    if state.from_action is not None:
        prev_action_enc[int(state.from_action.action.action)] = 1
        prev_action_enc[4] = state.from_action.action.amount / initial_stake
    encoded.append(prev_action_enc)

    # 连接所有特征
    return np.concatenate(encoded)
