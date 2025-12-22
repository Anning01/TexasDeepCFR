"""
DeepCFR扑克AI项目的日志工具。
提供游戏和训练过程中的详细错误日志记录功能。
"""

import os
import time
import traceback
import pokers
from typing import Callable, Optional


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


def log_game_error(
    state: pokers.State,
    action: pokers.Action,
    error_msg: str,
    card_converter: Callable = None,
) -> Optional[str]:
    """
    当游戏状态错误发生时，将详细错误信息记录到文件中。

    参数:
        state: 应用操作前的游戏状态
        action: 导致错误的操作
        error_msg: 错误消息或状态
        card_converter: 可选的函数，用于将卡牌转换为字符串
                      (默认为内部的card_to_string函数)

    返回:
        创建的日志文件路径，如果日志记录失败则返回None
    """
    # 使用提供的卡牌转换器或默认转换器
    if card_converter is None:
        card_converter = card_to_string

    # 如果logs目录不存在则创建
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # 为文件名创建时间戳
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(logs_dir, f"poker_error_{timestamp}.txt")

    try:
        with open(log_filename, "w") as f:
            # 写入错误摘要
            f.write(f"=== 扑克游戏错误日志 ===\n")
            f.write(f"时间戳: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"错误: {error_msg}\n\n")
            f.write(f"堆栈跟踪:\n{traceback.format_exc()}\n\n")

            # 写入游戏状态信息
            f.write(f"=== 游戏状态 ===\n")
            f.write(f"阶段: {state.stage}\n")
            f.write(f"底池: ${state.pot:.2f}\n")
            f.write(f"按钮位置: 玩家 {state.button}\n")
            f.write(f"当前玩家: 玩家 {state.current_player}\n")
            f.write(f"最小下注: ${state.min_bet:.2f}\n\n")

            # 写入公共牌
            community_cards = " ".join(
                [card_converter(card) for card in state.public_cards]
            )
            f.write(f"公共牌: {community_cards if community_cards else '无'}\n\n")

            # 写入所有玩家的状态
            f.write(f"=== 玩家状态 ===\n")
            for i, p in enumerate(state.players_state):
                hand = "未知"
                if hasattr(p, "hand") and p.hand:
                    hand = " ".join([card_converter(card) for card in p.hand])

                f.write(
                    f"玩家 {i}: ${p.stake:.2f} - 下注: ${p.bet_chips:.2f} - {'活跃' if p.active else '已弃牌'}\n"
                )
                f.write(f"  手牌: {hand}\n")
            f.write("\n")

            # 写入导致错误的操作
            f.write(f"=== 导致错误的操作 ===\n")
            if action.action == pokers.ActionEnum.Raise:
                f.write(f"操作: {action.action} ${action.amount:.2f}\n")
            else:
                f.write(f"操作: {action.action}\n")
            f.write("\n")

            # 写入合法操作
            f.write(f"=== 合法操作 ===\n")
            f.write(f"{state.legal_actions}\n\n")

            # 如果有前一个操作则写入
            if hasattr(state, "from_action") and state.from_action:
                f.write(f"=== 前一个操作 ===\n")
                prev_action = state.from_action.action
                if prev_action.action == pokers.ActionEnum.Raise:
                    f.write(f"操作: {prev_action.action} ${prev_action.amount:.2f}\n")
                else:
                    f.write(f"操作: {prev_action.action}\n")

        print(f"错误详情已记录到 {log_filename}")
        return log_filename
    except Exception as log_error:
        print(f"写入错误日志失败: {log_error}")
        return None
