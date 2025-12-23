"""
PokerKit适配层 - 将PokerKit的API适配为类似pokers-db的API
这样可以最小化对现有DeepCFR代码的修改，并支持完整的6人无限注德州扑克
"""

import numpy as np
from enum import IntEnum
from typing import List, Optional
from decimal import Decimal
from pokerkit import NoLimitTexasHoldem, Automation, Mode
from pokerkit.state import State as PokerKitState
from pokerkit.notation import HandHistory


# ===== 枚举定义 =====


class ActionEnum(IntEnum):
    """动作类型枚举（与pokers-db兼容）"""

    Fold = 0
    Check = 1
    Call = 2
    Raise = 3


class Stage(IntEnum):
    """游戏阶段枚举"""

    Preflop = 0
    Flop = 1
    Turn = 2
    River = 3
    Showdown = 4


class StateStatus(IntEnum):
    """状态状态枚举"""

    Ok = 0
    Invalid = 1
    GameOver = 2


# ===== 数据类 =====


class Card:
    """扑克牌类"""

    SUIT_MAP = {"s": 3, "h": 2, "d": 1, "c": 0}  # PokerKit使用小写
    RANK_MAP = {
        "2": 0,
        "3": 1,
        "4": 2,
        "5": 3,
        "6": 4,
        "7": 5,
        "8": 6,
        "9": 7,
        "T": 8,
        "J": 9,
        "Q": 10,
        "K": 11,
        "A": 12,
    }

    def __init__(self, pokerkit_card):
        """
        从PokerKit的卡牌对象或字符串创建Card对象

        参数:
            pokerkit_card: PokerKit的Card对象，或简短字符串形式如 'As', '2h'
        """
        # 如果是PokerKit Card对象，使用repr()获取简短形式
        if hasattr(pokerkit_card, "rank") and hasattr(pokerkit_card, "suit"):
            # 这是一个PokerKit Card对象，使用repr()
            card_str = repr(pokerkit_card)  # 返回如 'As', '2h' 的形式
        else:
            # 假设是字符串
            card_str = str(pokerkit_card)

        # 提取卡牌简写（处理括号形式）
        # 例如："ACE OF SPADES (As)" -> "As"
        if "(" in card_str and ")" in card_str:
            card_str = card_str.split("(")[1].split(")")[0]

        if not card_str or len(card_str) < 2:
            raise ValueError(f"Invalid card string: {card_str}")

        rank_char = card_str[0]
        suit_char = card_str[1].lower()

        self.rank = self.RANK_MAP.get(rank_char, 0)
        self.suit = self.SUIT_MAP.get(suit_char, 0)
        self._str = card_str

    def __repr__(self):
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
        return f"{ranks[self.rank]}{suits[self.suit]}"


class PlayerState:
    """玩家状态类"""

    def __init__(
        self,
        player_id: int,
        chips: float,
        bet: float,
        active: bool,
        hand: List[Card] = None,
    ):
        self.player = player_id
        self.stake = chips  # 剩余筹码
        self.bet_chips = bet  # 当前轮下注
        self.pot_chips = 0.0  # 已进入底池的筹码
        self.active = active  # 是否还在游戏中
        self.hand = hand or []  # 手牌
        self.reward = 0.0  # 最终收益


class Action:
    """动作类"""

    def __init__(self, action: ActionEnum, amount: float = 0.0):
        self.action = action
        self.amount = amount  # 对于Raise，这是额外加注的金额（相对于跟注之后）

    def __repr__(self):
        if self.action == ActionEnum.Raise:
            return f"Action({ActionEnum(self.action).name}, {self.amount:.2f})"
        return f"Action({ActionEnum(self.action).name})"


class ActionRecord:
    """动作记录"""

    def __init__(self, player_id: int, action: Action):
        self.player_id = player_id
        self.action = action


# ===== 主状态类 =====


class State:
    """
    游戏状态类 - 包装PokerKit环境
    提供与pokers-db兼容的API，支持6人无限注德州扑克

    注意：所有状态属性都是动态的，每次访问时从底层PokerKit状态读取
    """

    def __init__(
        self,
        pokerkit_state: PokerKitState,
        initial_stacks: List[float],
        button: int,
        sb: float,
        bb: float,
        seed: int,
    ):
        """不要直接调用，使用State.from_seed()"""
        self._pk_state = pokerkit_state
        self._initial_stacks = initial_stacks
        self.button = button
        self.sb = sb
        self.bb = bb
        self._seed = seed
        self.from_action = None  # 上一个动作记录

    @classmethod
    def from_seed(
        cls, n_players: int, button: int, sb: float, bb: float, stake: float, seed: int
    ) -> "State":
        """
        创建一个新的游戏状态（与pokers-db API兼容）

        参数:
            n_players: 玩家数量（2-10人，推荐6人）
            button: 庄家位置
            sb: 小盲注
            bb: 大盲注
            stake: 每个玩家的初始筹码
            seed: 随机种子
        """
        # 设置随机种子
        np.random.seed(seed)

        # 创建初始筹码列表
        starting_stacks = [stake] * n_players

        # 创建PokerKit状态
        pk_state = NoLimitTexasHoldem.create_state(
            # 自动化配置
            automations=(
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.CARD_BURNING,
                Automation.HOLE_DEALING,
                Automation.BOARD_DEALING,
                Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING,
            ),
            # 前注修剪状态
            ante_trimming_status=True,
            # 前注（不使用）
            raw_antes=0,
            # 盲注设置 (small_blind, big_blind)
            raw_blinds_or_straddles=(int(sb), int(bb)),
            # 最小下注
            min_bet=int(bb),
            # 初始筹码
            raw_starting_stacks=[int(s) for s in starting_stacks],
            # 玩家数量
            player_count=n_players,
            # 模式：现金局
            mode=Mode.CASH_GAME,
        )

        return cls(pk_state, starting_stacks, button, sb, bb, seed)

    # ===== 动态属性（properties） =====

    @property
    def current_player(self) -> int:
        """当前需要行动的玩家索引"""
        return (
            self._pk_state.actor_index if self._pk_state.actor_index is not None else 0
        )

    @property
    def pot(self) -> float:
        """当前底池大小"""
        return float(self._pk_state.total_pot_amount)

    @property
    def min_bet(self) -> float:
        """最小下注金额（用于跟注）"""
        amount = self._pk_state.checking_or_calling_amount
        return float(amount) if amount is not None else self.bb

    @property
    def stage(self) -> Stage:
        """当前游戏阶段"""
        return self._detect_stage()

    @property
    def final_state(self) -> bool:
        """游戏是否已经结束"""
        # PokerKit: status=False表示游戏结束
        # 但当 actor_index=None 且没有合法动作时，也应视为游戏结束
        # （这发生在 all-in showdown 等情况，没有更多决策需要做）
        if self._pk_state.status is False:
            return True

        # 如果没有玩家需要行动且没有合法动作，游戏实际上已经结束
        if self._pk_state.actor_index is None and not self._get_legal_actions():
            return True

        return False

    @property
    def status(self) -> StateStatus:
        """游戏状态"""
        if self.final_state:
            return StateStatus.GameOver
        else:
            return StateStatus.Ok

    @property
    def public_cards(self) -> List[Card]:
        """公共牌"""
        return self._parse_board_cards()

    @property
    def players_state(self) -> List[PlayerState]:
        """所有玩家的状态"""
        return self._build_player_states()

    @property
    def legal_actions(self) -> List[ActionEnum]:
        """当前可用的合法动作"""
        return self._get_legal_actions()

    # ===== 辅助方法 =====

    def _detect_stage(self) -> Stage:
        """检测当前游戏阶段"""
        pk = self._pk_state

        # PokerKit使用street_index来表示阶段
        # 0: preflop, 1: flop, 2: turn, 3: river
        if not hasattr(pk, "street_index") or pk.street_index is None:
            return Stage.Preflop

        street = pk.street_index
        if street == 0:
            return Stage.Preflop
        elif street == 1:
            return Stage.Flop
        elif street == 2:
            return Stage.Turn
        elif street == 3:
            return Stage.River
        else:
            return Stage.Showdown

    def _parse_board_cards(self) -> List[Card]:
        """解析公共牌"""
        pk = self._pk_state
        cards = []

        # PokerKit的board_cards是按street组织的列表的列表
        if hasattr(pk, "board_cards") and pk.board_cards:
            for street_cards in pk.board_cards:
                if street_cards:
                    for card in street_cards:
                        if card:
                            try:
                                cards.append(Card(card))
                            except Exception as e:
                                print(f"解析公共牌出错: {e}, card={card}")

        return cards

    def _build_player_states(self) -> List[PlayerState]:
        """构建玩家状态列表"""
        pk = self._pk_state
        player_states = []

        n_players = len(self._initial_stacks)

        # 检查游戏是否真正结束（使用与 final_state property 相同的逻辑）
        is_final = (pk.status is False) or (
            pk.actor_index is None and not self._get_legal_actions()
        )

        for i in range(n_players):
            # 获取筹码
            chips = float(pk.stacks[i]) if i < len(pk.stacks) else 0.0

            # 获取当前轮下注
            bet = float(pk.bets[i]) if i < len(pk.bets) else 0.0

            # 判断是否还在游戏中
            # PokerKit: statuses[i]=True表示玩家活跃，False表示已弃牌/出局
            active = bool(pk.statuses[i]) if i < len(pk.statuses) else False

            # 获取手牌（仅当我们能看到时）
            hand_cards = []
            if hasattr(pk, "hole_cards") and i < len(pk.hole_cards):
                hole = pk.hole_cards[i]
                if hole:
                    for card in hole:
                        if card:
                            try:
                                hand_cards.append(Card(card))
                            except Exception as e:
                                print(f"解析手牌出错: {e}, card={card}")

            ps = PlayerState(i, chips, bet, active, hand_cards)

            # 如果游戏结束，计算收益
            if is_final:
                # 收益 = 最终筹码 - 初始筹码
                final_stack = float(pk.stacks[i]) if i < len(pk.stacks) else 0.0
                initial_stack = self._initial_stacks[i]
                ps.reward = final_stack - initial_stack

            player_states.append(ps)

        return player_states

    def _get_legal_actions(self) -> List[ActionEnum]:
        """获取合法动作列表"""
        pk = self._pk_state

        # 直接检查 PokerKit 状态，避免使用 property 导致循环
        is_final = pk.status is False

        if is_final:
            return []

        # 如果没有玩家需要行动（自动化阶段），返回空列表
        if pk.actor_index is None:
            return []

        legal = []

        # Fold - 使用 can_fold() 方法检查
        if hasattr(pk, "can_fold") and pk.can_fold():
            legal.append(ActionEnum.Fold)

        # Check/Call - 使用 can_check_or_call() 方法检查
        if pk.can_check_or_call():
            if pk.checking_or_calling_amount == 0:
                legal.append(ActionEnum.Check)
            else:
                legal.append(ActionEnum.Call)

        # Raise - 使用 can_complete_bet_or_raise_to() 方法检查
        if pk.can_complete_bet_or_raise_to():
            legal.append(ActionEnum.Raise)

        return legal

    def apply_action(self, action: Action) -> "State":
        """
        应用动作并返回新状态

        参数:
            action: 要应用的动作

        返回:
            新的游戏状态
        """
        pk = self._pk_state

        # 获取当前玩家（避免多次访问 property）
        current_player_id = pk.actor_index if pk.actor_index is not None else 0

        # 创建PokerKit状态的副本（始终创建副本以避免状态共享）
        import copy

        new_pk_state = copy.deepcopy(pk)

        # 首先检查游戏是否已经结束
        if new_pk_state.status is False:
            print(f"警告: 游戏已经结束，无法应用动作 {action}")
            new_state = State(
                new_pk_state,
                self._initial_stacks,
                self.button,
                self.sb,
                self.bb,
                self._seed,
            )
            new_state.from_action = ActionRecord(current_player_id, action)
            return new_state

        # 检查是否有玩家需要行动
        if new_pk_state.actor_index is None:
            print(f"警告: 没有玩家需要行动（自动化阶段），无法应用动作 {action}")
            new_state = State(
                new_pk_state,
                self._initial_stacks,
                self.button,
                self.sb,
                self.bb,
                self._seed,
            )
            new_state.from_action = ActionRecord(current_player_id, action)
            return new_state

        try:
            if action.action == ActionEnum.Fold:
                new_pk_state.fold()

            elif action.action in (ActionEnum.Check, ActionEnum.Call):
                new_pk_state.check_or_call()

            elif action.action == ActionEnum.Raise:
                # action.amount 是额外加注的金额（在跟注之后）
                # 例如：如果需要跟注 $10，玩家加注 $20，则最终下注目标是 $30

                # 获取当前玩家状态
                actor_idx = new_pk_state.actor_index
                current_bet = (
                    float(new_pk_state.bets[actor_idx])
                    if actor_idx is not None
                    else 0.0
                )
                current_stack = (
                    float(new_pk_state.stacks[actor_idx])
                    if actor_idx is not None
                    else 0.0
                )

                # 玩家总共可以下注的最大金额（剩余筹码 + 已下注）
                max_possible_bet = current_stack + current_bet

                # 计算需要跟注的金额（额外投入）
                call_amount = (
                    float(new_pk_state.checking_or_calling_amount)
                    if new_pk_state.checking_or_calling_amount
                    else 0.0
                )

                # 计算总下注目标 = 当前已下注 + 需要跟注 + 额外加注
                total_target = current_bet + call_amount + action.amount

                # 限制不能超过玩家的总筹码
                total_target = min(total_target, max_possible_bet)

                # 获取最小加注要求（如果有）
                min_raise = (
                    float(new_pk_state.min_completion_betting_or_raising_to_amount)
                    if new_pk_state.min_completion_betting_or_raising_to_amount
                    else None
                )

                if min_raise is not None:
                    # 如果玩家筹码不够达到最小加注，只能全下
                    if max_possible_bet < min_raise:
                        total_target = max_possible_bet
                    else:
                        # 确保满足最小加注要求
                        total_target = max(total_target, min_raise)
                        # 再次确认不超过最大可能
                        total_target = min(total_target, max_possible_bet)

                new_pk_state.complete_bet_or_raise_to(int(total_target))

            # 创建新的State对象
            new_state = State(
                new_pk_state,
                self._initial_stacks,
                self.button,
                self.sb,
                self.bb,
                self._seed,
            )
            new_state.from_action = ActionRecord(current_player_id, action)

            return new_state

        except Exception as e:
            print(f"应用动作时出错: {e}, action={action}")
            import traceback

            traceback.print_exc()
            # 返回一个标记为无效的状态
            new_state = State(
                new_pk_state,
                self._initial_stacks,
                self.button,
                self.sb,
                self.bb,
                self._seed,
            )
            new_state.from_action = ActionRecord(current_player_id, action)
            return new_state


# 可视化函数（用于调试）
def visualize_state(state: State):
    """可视化游戏状态（调试用）"""
    print(f"\n=== 游戏状态 ===")
    print(f"阶段: {Stage(state.stage).name}")
    print(f"底池: ${state.pot:.2f}")
    print(f"当前玩家: {state.current_player}")
    print(f"公共牌: {[str(c) for c in state.public_cards]}")
    print(f"合法动作: {[ActionEnum(a).name for a in state.legal_actions]}")
    for ps in state.players_state:
        print(
            f"玩家 {ps.player}: 筹码=${ps.stake:.2f}, 下注=${ps.bet_chips:.2f}, 活跃={ps.active}, 手牌={[str(c) for c in ps.hand]}"
        )


def visualize_trace(states: List[State]):
    """可视化游戏轨迹（调试用）"""
    for i, state in enumerate(states):
        print(f"\n--- 步骤 {i} ---")
        visualize_state(state)
