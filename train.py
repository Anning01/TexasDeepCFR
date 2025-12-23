import os
import random
import time

import numpy as np
import pokerkit_adapter as pokers
import torch

from core.deepcfr import DeepCFRAgent
from core.model import set_verbose, encode_state
from game_logger import log_game_error
from random_agent import RandomAgent
from settings import STRICT_CHECKING, set_strict_checking


def evaluate_against_random(agent, num_games=500, num_players=6):
    """评估训练好的智能体与随机策略对手的表现。"""
    # 创建随机策略对手列表
    random_agents = [RandomAgent(i) for i in range(num_players)]
    total_profit = 0  # 总利润
    completed_games = 0  # 完成的游戏数量

    for game in range(num_games):
        try:
            # 创建一个新的扑克游戏状态
            state = pokers.State.from_seed(
                n_players=num_players,  # 玩家数量
                button=game % num_players,  # 按钮位置（为了公平性，每局轮换）
                sb=1,  # 小盲注
                bb=2,  # 大盲注
                stake=200.0,  # 初始筹码
                seed=game,  # 随机种子
            )
            # 游戏循环，直到游戏结束
            while not state.final_state:
                current_player = state.current_player  # 当前行动玩家
                # 根据当前玩家选择不同的代理
                if current_player == agent.player_id:
                    action = agent.choose_action(state)  # 使用训练好的智能体选择动作
                else:
                    action = random_agents[current_player].choose_action(
                        state
                    )  # 使用随机策略对手选择动作
                # 如果没有合法动作，说明游戏状态有问题，跳出循环
                if action is None:
                    print(f"警告: 游戏 {game} 中出现 action=None，跳出游戏循环")
                    break

                # 应用动作并检查状态是否有效
                new_state = state.apply_action(action)
                # 只有 Invalid 状态才是错误，Ok 和 GameOver 都是正常的
                if new_state.status == pokers.StateStatus.Invalid:
                    # 记录游戏错误
                    log_file = log_game_error(
                        state, action, f"状态错误 ({new_state.status})"
                    )
                    if STRICT_CHECKING:
                        raise ValueError(
                            f"状态错误 ({new_state.status}). 详细信息记录在 {log_file}"
                        )
                    else:
                        print(
                            f"警告: 游戏 {game} 中状态错误 ({new_state.status}). 详细信息记录在 {log_file}"
                        )
                        break  # 非严格模式下跳过本局游戏
                state = new_state  # 更新游戏状态

            # 只统计已完成的游戏
            if state.final_state:
                # 计算本局游戏的利润
                profit = state.players_state[agent.player_id].reward
                total_profit += profit
                completed_games += 1

        except Exception as e:
            if STRICT_CHECKING:
                raise  # 严格模式下重新抛出异常
            else:
                print(f"游戏 {game} 中出错: {e}")
                # 非严格模式下继续下一局游戏

    # 仅对已完成的游戏计算平均利润
    if completed_games == 0:
        print("警告: 评估过程中没有完成任何游戏!")
        return 0

    return total_profit / completed_games


def evaluate_against_checkpoint_agents(agent, opponent_agents, num_games=100):
    """
    评估训练后的智能体对抗对手智能体的表现。
    每个智能体将从自己的视角接收和处理观察。
    """
    total_profit = 0  # 总收益
    completed_games = 0  # 完成的游戏数量

    class AgentWrapper:
        """包装器，确保智能体从自己的视角接收观察"""

        def __init__(self, agent):
            self.agent = agent
            self.player_id = agent.player_id

        def choose_action(self, state):
            # 每个智能体从自己的视角处理状态
            return self.agent.choose_action(state)

    # 包装模型对手智能体
    opponent_wrappers = [None] * 6
    for pos in range(6):
        if pos != agent.player_id:
            opponent_wrappers[pos] = AgentWrapper(opponent_agents[pos])

    for game in range(num_games):
        try:
            # 创建一个带有轮换按钮的新扑克游戏
            state = pokers.State.from_seed(
                n_players=6,  # 玩家数量
                button=game % 6,  # 轮换按钮位置以保证公平性
                sb=1,  # 小盲注
                bb=2,  # 大盲注
                stake=200.0,  # 初始筹码
                seed=game + 10000,  # 使用与训练不同的种子
            )

            # 游戏进行直到结束
            while not state.final_state:
                current_player = state.current_player

                if current_player == agent.player_id:
                    action = agent.choose_action(state)  # 训练智能体选择动作
                else:
                    action = opponent_wrappers[current_player].choose_action(
                        state
                    )  # 对手智能体选择动作

                # 如果没有合法动作，说明游戏状态有问题，跳出循环
                if action is None:
                    print(f"警告: 游戏 {game} 中出现 action=None，跳出游戏循环")
                    break

                # 应用动作并检查状态
                new_state = state.apply_action(action)
                # 只有 Invalid 状态才是错误，Ok 和 GameOver 都是正常的
                if new_state.status == pokers.StateStatus.Invalid:
                    log_file = log_game_error(
                        state, action, f"状态错误 ({new_state.status})"
                    )
                    if STRICT_CHECKING:
                        raise ValueError(
                            f"状态错误 ({new_state.status}). 详情已记录到 {log_file}"
                        )
                    else:
                        print(
                            f"警告: 游戏 {game} 中状态错误 ({new_state.status})。详情已记录到 {log_file}"
                        )
                        break  # 在非严格模式下跳过此游戏

                state = new_state  # 更新游戏状态

            # 只计算已完成的游戏
            if state.final_state:
                # 添加此游戏的收益
                profit = state.players_state[agent.player_id].reward
                total_profit += profit
                completed_games += 1

        except Exception as e:
            if STRICT_CHECKING:
                raise  # 在严格模式下重新抛出异常
            else:
                print(f"游戏 {game} 出错: {e}")
                # 在非严格模式下继续下一场游戏

    # 仅返回已完成游戏的平均收益
    if completed_games == 0:
        print("警告: 评估期间没有完成任何游戏!")
        return 0

    return total_profit / completed_games


def train_deep_cfr(
    num_iterations=1000,
    traversals_per_iteration=200,
    num_players=6,
    player_id=0,
    save_dir="models",
    log_dir="logs/deepcfr",
    verbose=False,
):
    """
    训练Deep CFR智能体进行6人无限制德州扑克游戏，对抗5个随机对手。

    参数:
        num_iterations: CFR迭代次数
        traversals_per_iteration: 每次迭代的遍历次数
        num_players: 玩家数量
        player_id: 训练智能体的玩家ID
        save_dir: 模型保存目录
        log_dir: TensorBoard日志目录
        verbose: 是否启用详细输出
    """
    # 导入TensorBoard
    from torch.utils.tensorboard import SummaryWriter

    # 设置详细输出模式
    set_verbose(verbose)

    # 创建必要的目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 初始化TensorBoard写入器
    writer = SummaryWriter(log_dir)

    # 初始化Deep CFR智能体
    agent = DeepCFRAgent(
        player_id=player_id,
        num_players=num_players,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 创建随机策略对手
    random_agents = [RandomAgent(i) for i in range(num_players)]

    # 用于跟踪学习进度
    losses = []  # 损失记录
    profits = []  # 利润记录
    print(f"玩家: {num_players}")
    # 在训练开始前进行初始评估
    print("初始评估中...")
    initial_profit = evaluate_against_random(
        agent, num_games=500, num_players=num_players
    )
    profits.append(initial_profit)
    print(f"初始每局平均利润: {initial_profit:.2f}")
    writer.add_scalar("Performance/Profit", initial_profit, 0)

    # 模型保存频率
    checkpoint_frequency = 100  # 每100次迭代保存一次模型

    # 训练循环
    for iteration in range(1, num_iterations + 1):
        agent.iteration_count = iteration  # 设置当前迭代次数
        start_time = time.time()  # 记录迭代开始时间

        print(f"迭代 {iteration}/{num_iterations}")

        # 进行多次遍历收集数据
        print("  收集数据中...")
        for _ in range(traversals_per_iteration):
            # 创建一个新的扑克游戏
            state = pokers.State.from_seed(
                n_players=num_players,
                button=random.randint(0, num_players - 1),  # 随机选择按钮位置
                sb=1,  # 小盲注
                bb=2,  # 大盲注
                stake=200.0,  # 初始筹码
                seed=random.randint(0, 10000),  # 随机种子
            )

            # 执行CFR遍历以收集数据
            agent.cfr_traverse(state, iteration, random_agents)

        # 记录遍历时间
        traversal_time = time.time() - start_time
        writer.add_scalar("Time/Traversal", traversal_time, iteration)

        # 训练优势网络
        print("  训练优势网络中...")
        adv_loss = agent.train_advantage_network()  # 计算优势网络损失
        losses.append(adv_loss)
        print(f"  优势网络损失: {adv_loss:.6f}")

        # 将损失记录到TensorBoard
        writer.add_scalar("Loss/Advantage", adv_loss, iteration)
        writer.add_scalar("Memory/Advantage", len(agent.advantage_memory), iteration)

        # 每10次迭代或最后一次迭代，训练策略网络并评估
        if iteration % 10 == 0 or iteration == num_iterations:
            print("  训练策略网络中...")
            strat_loss = agent.train_strategy_network()  # 计算策略网络损失
            print(f"  策略网络损失: {strat_loss:.6f}")
            writer.add_scalar("Loss/Strategy", strat_loss, iteration)

            # 评估智能体性能
            print("  评估智能体中...")
            avg_profit = evaluate_against_random(
                agent, num_games=500, num_players=num_players
            )
            profits.append(avg_profit)
            print(f"  每局平均利润: {avg_profit:.2f}")
            writer.add_scalar("Performance/Profit", avg_profit, iteration)

            # 保存模型（不用每次都保存，使用定期保存）
            # model_path = f"{save_dir}/deep_cfr_iter_{iteration}.pt"
            # agent.save_model(model_path)
            # print(f"  模型保存到 {model_path}")

        # 定期保存模型
        if iteration % checkpoint_frequency == 0:
            checkpoint_path = f"{save_dir}/checkpoint_iter_{iteration}.pt"
            torch.save(
                {
                    "iteration": iteration,  # 当前迭代次数
                    "advantage_net": agent.advantage_net.state_dict(),  # 优势网络权重
                    "strategy_net": agent.strategy_net.state_dict(),  # 策略网络权重
                    "losses": losses,  # 损失历史
                    "profits": profits,  # 利润历史
                },
                checkpoint_path,
            )
            print(f"  模型保存到 {checkpoint_path}")

        # 计算迭代总时间
        elapsed = time.time() - start_time
        writer.add_scalar("Time/Iteration", elapsed, iteration)
        print(f"  迭代完成，耗时 {elapsed:.2f} 秒")
        print(f"  优势网络内存大小: {len(agent.advantage_memory)}")
        print(f"  策略网络内存大小: {len(agent.strategy_memory)}")
        writer.add_scalar("Memory/Strategy", len(agent.strategy_memory), iteration)

        # 提交TensorBoard日志
        writer.flush()
        print()

    # 最终评估
    print("最终评估中...")
    avg_profit = evaluate_against_random(agent, num_games=1000)
    print(f"最终表现: 每局平均利润: {avg_profit:.2f}")
    writer.add_scalar("Performance/FinalProfit", avg_profit, 0)

    # 关闭TensorBoard写入器
    writer.close()

    return agent, losses, profits


def continue_training(
    checkpoint_path,
    additional_iterations=1000,
    traversals_per_iteration=200,
    save_dir="models",
    log_dir="logs/deepcfr_continued",
    verbose=False,
):
    """
    从保存的模型继续训练Deep CFR智能体。

    参数:
        checkpoint_path: 保存的模型路径
        additional_iterations: 要继续训练的迭代次数
        traversals_per_iteration: 每次迭代的遍历次数
        save_dir: 新模型保存目录
        log_dir: TensorBoard日志目录
        verbose: 是否启用详细输出
    """
    # 导入TensorBoard
    from torch.utils.tensorboard import SummaryWriter

    # 设置详细输出模式
    set_verbose(verbose)

    # 创建必要的目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 初始化TensorBoard写入器
    writer = SummaryWriter(log_dir)

    # 加载模型
    print(f"从 {checkpoint_path} 加载模型")
    checkpoint = torch.load(checkpoint_path)

    # 初始化智能体
    num_players = 6  # 与原始训练保持一致，假设6名玩家
    player_id = 0  # 与原始训练保持一致，训练ID为0的玩家
    agent = DeepCFRAgent(
        player_id=player_id,
        num_players=num_players,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 加载模型权重
    agent.advantage_net.load_state_dict(checkpoint["advantage_net"])
    agent.strategy_net.load_state_dict(checkpoint["strategy_net"])

    # 从模型设置迭代计数
    start_iteration = checkpoint["iteration"] + 1
    agent.iteration_count = start_iteration - 1

    # 加载训练历史（如果可用）
    losses = checkpoint.get("losses", [])
    profits = checkpoint.get("profits", [])

    print(f"从迭代 {start_iteration - 1} 加载模型")
    print(f"继续训练 {additional_iterations} 次迭代")

    # 创建随机策略对手
    random_agents = [RandomAgent(i) for i in range(num_players)]

    # 添加：继续训练前进行初始评估
    print("已加载模型的初始评估...")
    initial_profit = evaluate_against_random(
        agent, num_games=500, num_players=num_players
    )
    if not profits:  # 仅当利润列表为空时添加
        profits.append(initial_profit)
    print(f"初始每局平均利润: {initial_profit:.2f}")
    writer.add_scalar("Performance/Profit", initial_profit, start_iteration - 1)

    # 模型保存频率
    checkpoint_frequency = 100  # 每100次迭代保存一次模型

    # 训练循环
    for iteration in range(start_iteration, start_iteration + additional_iterations):
        agent.iteration_count = iteration  # 设置当前迭代次数
        start_time = time.time()  # 记录迭代开始时间

        print(f"迭代 {iteration}/{start_iteration + additional_iterations - 1}")

        # 进行多次遍历收集数据
        print("  收集数据中...")
        for _ in range(traversals_per_iteration):
            # 创建一个新的扑克游戏
            state = pokers.State.from_seed(
                n_players=num_players,
                button=random.randint(0, num_players - 1),  # 随机选择按钮位置
                sb=1,  # 小盲注
                bb=2,  # 大盲注
                stake=200.0,  # 初始筹码
                seed=random.randint(0, 10000),  # 随机种子
            )

            # 执行CFR遍历以收集数据
            agent.cfr_traverse(state, iteration, random_agents)

        # 记录遍历时间
        traversal_time = time.time() - start_time
        writer.add_scalar("Time/Traversal", traversal_time, iteration)

        # 训练优势网络
        print("  训练优势网络中...")
        adv_loss = agent.train_advantage_network()  # 计算优势网络损失
        losses.append(adv_loss)
        print(f"  优势网络损失: {adv_loss:.6f}")

        # 将损失记录到TensorBoard
        writer.add_scalar("Loss/Advantage", adv_loss, iteration)
        writer.add_scalar("Memory/Advantage", len(agent.advantage_memory), iteration)

        # 每10次迭代或最后一次迭代，训练策略网络并评估
        if (
            iteration % 10 == 0
            or iteration == start_iteration + additional_iterations - 1
        ):
            print("  训练策略网络中...")
            strat_loss = agent.train_strategy_network()  # 计算策略网络损失
            print(f"  策略网络损失: {strat_loss:.6f}")
            writer.add_scalar("Loss/Strategy", strat_loss, iteration)

            # 评估智能体性能
            print("  评估智能体中...")
            avg_profit = evaluate_against_random(
                agent, num_games=500, num_players=num_players
            )
            profits.append(avg_profit)
            print(f"  每局平均利润: {avg_profit:.2f}")
            writer.add_scalar("Performance/Profit", avg_profit, iteration)

            # 保存模型
            model_path = f"{save_dir}/deep_cfr_iter_{iteration}.pt"
            agent.save_model(model_path)
            print(f"  模型保存到 {model_path}")

        # 定期保存模型
        if iteration % checkpoint_frequency == 0:
            checkpoint_path = f"{save_dir}/checkpoint_iter_{iteration}.pt"
            torch.save(
                {
                    "iteration": iteration,  # 当前迭代次数
                    "advantage_net": agent.advantage_net.state_dict(),  # 优势网络权重
                    "strategy_net": agent.strategy_net.state_dict(),  # 策略网络权重
                    "losses": losses,  # 损失历史
                    "profits": profits,  # 利润历史
                },
                checkpoint_path,
            )
            print(f"  模型保存到 {checkpoint_path}")

        # 计算迭代总时间
        elapsed = time.time() - start_time
        writer.add_scalar("Time/Iteration", elapsed, iteration)
        print(f"  迭代完成，耗时 {elapsed:.2f} 秒")
        print(f"  优势网络内存大小: {len(agent.advantage_memory)}")
        print(f"  策略网络内存大小: {len(agent.strategy_memory)}")
        writer.add_scalar("Memory/Strategy", len(agent.strategy_memory), iteration)

        # 提交TensorBoard日志
        writer.flush()
        print()

    # 最终评估
    print("最终评估中...")
    avg_profit = evaluate_against_random(agent, num_games=1000)
    print(f"最终表现: 每局平均利润: {avg_profit:.2f}")
    writer.add_scalar("Performance/FinalProfit", avg_profit, 0)

    # 关闭TensorBoard写入器
    writer.close()

    return agent, losses, profits


def train_against_checkpoint(
    checkpoint_path,
    additional_iterations=1000,
    traversals_per_iteration=200,
    save_dir="models",
    log_dir="logs/deepcfr_selfplay",
    verbose=False,
):
    """
    训练一个新的Deep CFR智能体，使其对抗从模型加载的固定智能体。

    参数:
        checkpoint_path: 用作对手的已保存模型路径
        additional_iterations: 训练迭代次数
        traversals_per_iteration: 每次迭代的遍历次数
        save_dir: 新模型保存目录
        log_dir: TensorBoard日志目录
        verbose: 是否启用详细输出
    """
    from torch.utils.tensorboard import SummaryWriter

    # 设置详细输出模式
    set_verbose(verbose)

    # 创建必要的目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 初始化TensorBoard写入器
    writer = SummaryWriter(log_dir)

    print(f"从模型加载对手: {checkpoint_path}")

    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 为所有位置创建对手智能体
    opponent_agents = []
    for pos in range(6):
        # 为每个位置创建一个新智能体
        pos_agent = DeepCFRAgent(player_id=pos, num_players=6, device=device)
        pos_agent.load_model(checkpoint_path)  # 加载模型模型
        opponent_agents.append(pos_agent)

    # 初始化位置0的新学习智能体
    learning_agent = DeepCFRAgent(player_id=0, num_players=6, device=device)

    # 可选：从模型权重初始化学习智能体
    # 这将为它提供更好的起点
    # learning_agent.advantage_net.load_state_dict(opponent_agents[0].advantage_net.state_dict())
    # learning_agent.strategy_net.load_state_dict(opponent_agents[0].strategy_net.state_dict())

    # 用于跟踪学习进度
    losses = []  # 损失记录
    profits = []  # 利润记录

    # 添加：训练开始前进行初始评估
    print("初始评估中...")
    initial_profit_vs_checkpoint = evaluate_against_checkpoint_agents(
        learning_agent, opponent_agents, num_games=100
    )
    print(f"对抗模型对手的初始平均利润: {initial_profit_vs_checkpoint:.2f}")
    writer.add_scalar("Performance/ProfitVsCheckpoint", initial_profit_vs_checkpoint, 0)

    initial_profit_random = evaluate_against_random(
        learning_agent, num_games=500, num_players=6
    )
    profits.append(initial_profit_random)
    print(f"对抗随机对手的初始平均利润: {initial_profit_random:.2f}")
    writer.add_scalar("Performance/ProfitVsRandom", initial_profit_random, 0)

    # 模型保存频率
    checkpoint_frequency = 100  # 自博弈训练时保存频率更高

    class AgentWrapper:
        """包装器，确保智能体从自己的视角接收观察"""

        def __init__(self, agent):
            self.agent = agent
            self.player_id = agent.player_id

        def choose_action(self, state):
            # 这一点很关键 - 智能体从自己的视角查看状态
            return self.agent.choose_action(state)

    # 保存原始的cfr_traverse方法，以便稍后恢复
    original_cfr_traverse = DeepCFRAgent.cfr_traverse

    # 定义一个用于自博弈的修改版cfr_traverse方法
    def self_play_cfr_traverse(self, state, iteration, opponent_agents, depth=0):
        """
        修改后的CFR遍历方法，确保每个智能体都有正确的状态视角。
        """
        # 添加递归深度保护
        max_depth = 1000
        if depth > max_depth:
            if verbose:
                print(f"警告: 达到最大递归深度 ({max_depth})。返回零值。")
            return 0

        if state.final_state:
            # 返回训练智能体的收益
            return state.players_state[self.player_id].reward

        current_player = state.current_player

        # 当前状态的调试信息
        if verbose and depth % 100 == 0:
            print(f"深度: {depth}, 玩家: {current_player}, 阶段: {state.stage}")

        # 如果轮到训练智能体行动
        if current_player == self.player_id:
            legal_action_ids = self.get_legal_action_ids(state)

            if not legal_action_ids:
                if verbose:
                    print(
                        f"警告: 在深度 {depth} 为玩家 {current_player} 找不到合法行动"
                    )
                return 0

            # 从该智能体的视角编码状态
            state_tensor = torch.FloatTensor(encode_state(state, self.player_id)).to(
                self.device
            )

            # 从网络获取优势值
            with torch.no_grad():
                advantages = self.advantage_net(state_tensor.unsqueeze(0))[0]

            # 使用遗憾匹配计算策略
            advantages_np = advantages.cpu().numpy()
            advantages_masked = np.zeros(self.num_actions)
            for a in legal_action_ids:
                advantages_masked[a] = max(advantages_np[a], 0)  # 仅保留正的优势值

            # 根据策略选择行动
            if sum(advantages_masked) > 0:
                strategy = advantages_masked / sum(advantages_masked)  # 归一化策略
            else:
                # 如果所有优势值都非正，则使用均匀分布
                strategy = np.zeros(self.num_actions)
                for a in legal_action_ids:
                    strategy[a] = 1.0 / len(legal_action_ids)

            # 选择行动并遍历
            action_values = np.zeros(self.num_actions)
            for action_id in legal_action_ids:
                try:
                    pokers_action = self.action_id_to_pokers_action(action_id, state)
                    new_state = state.apply_action(pokers_action)

                    # 检查行动是否有效（只有 Invalid 才是错误）
                    if new_state.status == pokers.StateStatus.Invalid:
                        if verbose:
                            print(
                                f"警告: 在深度 {depth} 处行动 {action_id} 无效。状态: {new_state.status}"
                            )
                        continue

                    # 递归调用cfr_traverse进行下一个状态的遍历
                    action_values[action_id] = self.cfr_traverse(
                        new_state, iteration, opponent_agents, depth + 1
                    )
                except Exception as e:
                    if verbose:
                        print(f"行动 {action_id} 的遍历出错: {e}")
                    action_values[action_id] = 0

            # 计算反事实遗憾并添加到内存
            ev = sum(
                strategy[a] * action_values[a] for a in legal_action_ids
            )  # 期望收益

            # 计算归一化因子
            max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)

            for action_id in legal_action_ids:
                # 计算遗憾值
                regret = action_values[action_id] - ev

                # 归一化并裁剪遗憾值
                normalized_regret = regret / max_abs_val
                clipped_regret = np.clip(normalized_regret, -10.0, 10.0)

                # 应用缩放因子
                scale_factor = np.sqrt(iteration) if iteration > 1 else 1.0

                # 将数据添加到优势网络内存
                self.advantage_memory.append(
                    (
                        encode_state(state, self.player_id),  # 从该智能体的视角编码
                        action_id,
                        clipped_regret * scale_factor,
                    )
                )

            # 将策略添加到策略网络内存
            strategy_full = np.zeros(self.num_actions)
            for a in legal_action_ids:
                strategy_full[a] = strategy[a]

            self.strategy_memory.append(
                (
                    encode_state(state, self.player_id),  # 从该智能体的视角编码
                    strategy_full,
                    iteration,
                )
            )

            return ev

        # 如果轮到其他玩家行动（对手智能体）
        else:
            try:
                # 让相应的对手智能体选择行动
                if opponent_agents[current_player] is not None:
                    action = opponent_agents[current_player].choose_action(state)

                    # 如果没有合法动作（自动化阶段），返回0
                    if action is None:
                        return 0

                    new_state = state.apply_action(action)

                    # 检查行动是否有效（只有 Invalid 才是错误）
                    if new_state.status == pokers.StateStatus.Invalid:
                        if verbose:
                            print(
                                f"警告: 在深度 {depth} 处对手智能体行动无效。状态: {new_state.status}"
                            )
                        return 0

                    # 递归调用cfr_traverse进行下一个状态的遍历
                    return self.cfr_traverse(
                        new_state, iteration, opponent_agents, depth + 1
                    )
                else:
                    # 这不应发生 - 所有对手位置都应该有智能体
                    if verbose:
                        print(f"警告: 位置 {current_player} 没有对手智能体")
                    return 0
            except Exception as e:
                if verbose:
                    print(f"对手智能体遍历出错: {e}")
                return 0

    # Replace the cfr_traverse method with our self-play version
    DeepCFRAgent.cfr_traverse = self_play_cfr_traverse

    try:
        # Training loop
        for iteration in range(1, additional_iterations + 1):
            learning_agent.iteration_count = iteration
            start_time = time.time()

            print(f"Self-play Iteration {iteration}/{additional_iterations}")

            # Run traversals to collect data
            print("  Collecting data...")
            for t in range(traversals_per_iteration):
                # Rotate the button position for fairness
                button_pos = t % 6

                # Create a new poker game
                state = pokers.State.from_seed(
                    n_players=6,
                    button=button_pos,
                    sb=1,
                    bb=2,
                    stake=200.0,
                    seed=random.randint(0, 10000),
                )

                # Set up the opponent agents for this traversal
                # The learning agent always plays as player 0
                opponent_wrappers = [None] * 6
                for pos in range(6):
                    if pos != 0:  # Not the learning agent's position
                        # Use the opponent agent for this position
                        opponent_wrappers[pos] = AgentWrapper(opponent_agents[pos])

                # Perform CFR traversal
                learning_agent.cfr_traverse(state, iteration, opponent_wrappers)

            # 跟踪遍历时间
            traversal_time = time.time() - start_time
            writer.add_scalar("Time/Traversal", traversal_time, iteration)

            # 训练优势网络
            print("  训练优势网络中...")
            adv_loss = learning_agent.train_advantage_network()
            losses.append(adv_loss)
            print(f"  优势网络损失: {adv_loss:.6f}")

            # 将损失记录到tensorboard
            writer.add_scalar("Loss/Advantage", adv_loss, iteration)
            writer.add_scalar(
                "Memory/Advantage", len(learning_agent.advantage_memory), iteration
            )

            # 每隔几次迭代，训练策略网络并评估
            if iteration % 10 == 0 or iteration == additional_iterations:
                print("  Training strategy network...")
                strat_loss = learning_agent.train_strategy_network()
                print(f"  Strategy network loss: {strat_loss:.6f}")
                writer.add_scalar("Loss/Strategy", strat_loss, iteration)

                # Evaluate against checkpoint agents
                print("  Evaluating against checkpoint agent...")
                avg_profit_vs_checkpoint = evaluate_against_checkpoint_agents(
                    learning_agent, opponent_agents, num_games=100
                )
                print(f"  Average profit vs checkpoint: {avg_profit_vs_checkpoint:.2f}")
                writer.add_scalar(
                    "Performance/ProfitVsCheckpoint",
                    avg_profit_vs_checkpoint,
                    iteration,
                )

                # Also evaluate against random for comparison
                print("  Evaluating against random agents...")
                avg_profit_random = evaluate_against_random(
                    learning_agent, num_games=500, num_players=6
                )
                profits.append(avg_profit_random)
                print(f"  Average profit vs random: {avg_profit_random:.2f}")
                writer.add_scalar(
                    "Performance/ProfitVsRandom", avg_profit_random, iteration
                )

                # 保存模型
                # model_path = f"{save_dir}/deep_cfr_selfplay_iter_{iteration}"
                # learning_agent.save_model(model_path)
                # print(f"  Model saved to {model_path}")

            # 定期保存模型
            if iteration % checkpoint_frequency == 0:
                checkpoint_path = f"{save_dir}/selfplay_checkpoint_iter_{iteration}.pt"
                torch.save(
                    {
                        "iteration": iteration,
                        "advantage_net": learning_agent.advantage_net.state_dict(),
                        "strategy_net": learning_agent.strategy_net.state_dict(),
                        "losses": losses,
                        "profits": profits,
                    },
                    checkpoint_path,
                )
                print(f"  Checkpoint saved to {checkpoint_path}")

            elapsed = time.time() - start_time
            writer.add_scalar("Time/Iteration", elapsed, iteration)
            print(f"  迭代在 {elapsed:.2f} 秒内完成")
            print(f"  优势内存大小: {len(learning_agent.advantage_memory)}")
            print(f"  Strategy memory size: {len(learning_agent.strategy_memory)}")
            writer.add_scalar(
                "Memory/Strategy", len(learning_agent.strategy_memory), iteration
            )

            # Commit the tensorboard logs
            writer.flush()
            print()

        # Final evaluation with more games
        print("Final evaluation...")
        avg_profit_vs_checkpoint = evaluate_against_checkpoint_agents(
            learning_agent, opponent_agents, num_games=500
        )
        print(
            f"Final performance vs checkpoint: Average profit per game: {avg_profit_vs_checkpoint:.2f}"
        )
        writer.add_scalar(
            "Performance/FinalProfitVsCheckpoint", avg_profit_vs_checkpoint, 0
        )

        avg_profit_random = evaluate_against_random(learning_agent, num_games=500)
        print(
            f"Final performance vs random: Average profit per game: {avg_profit_random:.2f}"
        )
        writer.add_scalar("Performance/FinalProfitVsRandom", avg_profit_random, 0)

        writer.close()

        return learning_agent, losses, profits

    finally:
        # 恢复原始的cfr_traverse方法以避免副作用
        DeepCFRAgent.cfr_traverse = original_cfr_traverse


def train_with_mixed_checkpoints(
    checkpoint_dir,
    training_model_prefix="t_",
    additional_iterations=1000,
    traversals_per_iteration=200,
    save_dir="models",
    log_dir="logs/deepcfr_mixed",
    refresh_interval=1000,
    num_opponents=5,
    verbose=False,
):
    """
    训练Deep CFR智能体对抗从模型池中随机选择的对手。

    参数:
        checkpoint_dir: 包含模型模型的目录
        training_model_prefix: 应该包含在选择池中的模型前缀
        additional_iterations: 训练迭代次数
        traversals_per_iteration: 每次迭代的遍历次数
        save_dir: 保存新模型的目录
        log_dir: TensorBoard日志目录
        refresh_interval: 刷新对手池的频率（以迭代次数为单位）
        num_opponents: 从池中选择的对手数量
        verbose: 是否打印详细输出
    """
    from torch.utils.tensorboard import SummaryWriter
    import glob
    import os
    import random

    # 设置详细输出级别
    set_verbose(verbose)

    # 创建必要的目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 初始化TensorBoard日志写入器
    writer = SummaryWriter(log_dir)

    # 设备配置（优先使用GPU，如果可用）
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化学习智能体
    learning_agent = DeepCFRAgent(player_id=0, num_players=6, device=device)

    # 跟踪学习进度的变量
    losses = []  # 损失历史
    profits = []  # 收益历史
    profits_vs_checkpoints = []  # 对抗模型对手的收益历史

    # 保存学习智能体的模型频率
    checkpoint_frequency = 100

    # 智能体包装器辅助类
    class AgentWrapper:
        """包装器，确保智能体从自己的视角接收观察"""

        def __init__(self, agent):
            self.agent = agent
            self.player_id = agent.player_id

        def choose_action(self, state):
            # 这一点很关键 - 智能体从自己的视角查看状态
            return self.agent.choose_action(state)

    # 保存原始的cfr_traverse方法，以便稍后恢复
    original_cfr_traverse = DeepCFRAgent.cfr_traverse

    # 定义用于混合模型训练的修改版cfr_traverse方法
    def mixed_checkpoints_cfr_traverse(
        self, state, iteration, opponent_agents, depth=0
    ):
        """
        修改后的CFR遍历方法，确保每个智能体都有正确的状态视角。
        """
        # 添加递归深度保护
        max_depth = 1000
        if depth > max_depth:
            if verbose:
                print(f"警告: 达到最大递归深度 ({max_depth})。返回零值。")
            return 0

        if state.final_state:
            # 返回训练智能体的收益
            return state.players_state[self.player_id].reward

        current_player = state.current_player

        # 当前状态的调试信息
        if verbose and depth % 100 == 0:
            print(f"深度: {depth}, 玩家: {current_player}, 阶段: {state.stage}")

        # 如果轮到训练智能体行动
        if current_player == self.player_id:
            legal_action_ids = self.get_legal_action_ids(state)

            if not legal_action_ids:
                if verbose:
                    print(
                        f"警告: 在深度 {depth} 为玩家 {current_player} 找不到合法行动"
                    )
                return 0

            # 从当前智能体的角度编码状态
            state_tensor = torch.FloatTensor(encode_state(state, self.player_id)).to(
                self.device
            )

            # 从网络获取优势值
            with torch.no_grad():
                advantages = self.advantage_net(state_tensor.unsqueeze(0))[0]

            # 使用遗憾匹配来计算策略
            advantages_np = advantages.cpu().numpy()
            advantages_masked = np.zeros(self.num_actions)
            for a in legal_action_ids:
                advantages_masked[a] = max(advantages_np[a], 0)

            # 根据策略选择动作
            if sum(advantages_masked) > 0:
                strategy = advantages_masked / sum(advantages_masked)
            else:
                strategy = np.zeros(self.num_actions)
                for a in legal_action_ids:
                    strategy[a] = 1.0 / len(legal_action_ids)

            # Choose actions and traverse
            action_values = np.zeros(self.num_actions)
            for action_id in legal_action_ids:
                try:
                    pokers_action = self.action_id_to_pokers_action(action_id, state)
                    new_state = state.apply_action(pokers_action)

                    # Check if the action was valid (only Invalid is an error)
                    if new_state.status == pokers.StateStatus.Invalid:
                        if verbose:
                            print(
                                f"WARNING: Invalid action {action_id} at depth {depth}. Status: {new_state.status}"
                            )
                        continue

                    action_values[action_id] = self.cfr_traverse(
                        new_state, iteration, opponent_agents, depth + 1
                    )
                except Exception as e:
                    if verbose:
                        print(f"ERROR in traversal for action {action_id}: {e}")
                    action_values[action_id] = 0

            # 计算反事实遗憾并添加到内存
            ev = sum(strategy[a] * action_values[a] for a in legal_action_ids)

            # 计算归一化因子
            max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)

            for action_id in legal_action_ids:
                # 计算遗憾
                regret = action_values[action_id] - ev

                # 归一化和裁剪遗憾
                normalized_regret = regret / max_abs_val
                clipped_regret = np.clip(normalized_regret, -10.0, 10.0)

                # 应用缩放因子
                scale_factor = np.sqrt(iteration) if iteration > 1 else 1.0

                self.advantage_memory.append(
                    (
                        encode_state(
                            state, self.player_id
                        ),  # 从当前智能体的角度编码状态
                        action_id,
                        clipped_regret * scale_factor,
                    )
                )

            # 添加到策略内存
            strategy_full = np.zeros(self.num_actions)
            for a in legal_action_ids:
                strategy_full[a] = strategy[a]

            self.strategy_memory.append(
                (
                    encode_state(state, self.player_id),  # 从当前智能体的角度编码状态
                    strategy_full,
                    iteration,
                )
            )

            return ev

        # 如果轮到其他玩家行动（模型智能体或随机智能体）
        else:
            try:
                # Let the appropriate opponent agent choose an action
                if opponent_agents[current_player] is not None:
                    action = opponent_agents[current_player].choose_action(state)

                    # 如果没有合法动作（自动化阶段），返回0
                    if action is None:
                        return 0

                    new_state = state.apply_action(action)

                    # Check if the action was valid (only Invalid is an error)
                    if new_state.status == pokers.StateStatus.Invalid:
                        if verbose:
                            print(
                                f"WARNING: Opponent agent made invalid action at depth {depth}. Status: {new_state.status}"
                            )
                        return 0

                    return self.cfr_traverse(
                        new_state, iteration, opponent_agents, depth + 1
                    )
                else:
                    # 这不应该发生 - 所有对手位置都应该有智能体
                    if verbose:
                        print(
                            f"WARNING: No opponent agent for position {current_player}"
                        )
                    return 0
            except Exception as e:
                if verbose:
                    print(f"ERROR in opponent agent traversal: {e}")
                return 0

    # 替换cfr_traverse方法为我们的混合模型版本
    DeepCFRAgent.cfr_traverse = mixed_checkpoints_cfr_traverse

    try:
        # 从模型池中选择随机模型的函数
        def select_random_checkpoints():
            # 获取所有具有指定前缀的模型文件
            checkpoint_files = glob.glob(
                os.path.join(checkpoint_dir, f"{training_model_prefix}*.pt")
            )

            if not checkpoint_files:
                print(
                    f"警告: 在 {checkpoint_dir} 中未找到前缀为 '{training_model_prefix}' 的模型文件"
                )
                # 创建随机智能体作为备用
                return [RandomAgent(i) for i in range(6)]

            # 选择随机模型
            selected_files = random.sample(
                checkpoint_files, min(num_opponents, len(checkpoint_files))
            )
            print(f"已选择模型: {[os.path.basename(f) for f in selected_files]}")

            # 从所选模型创建智能体
            selected_agents = []

            # 始终在位置1保留一个随机智能体（可选，可修改）
            random_agent = RandomAgent(1)

            # 为其他位置加载模型智能体
            current_pos = 1
            for checkpoint_file in selected_files:
                # 跳过位置0，因为它是为我们的学习智能体保留的
                if current_pos == 0:
                    current_pos += 1

                # 创建并加载智能体
                checkpoint_agent = DeepCFRAgent(
                    player_id=current_pos, num_players=6, device=device
                )
                checkpoint_agent.load_model(checkpoint_file)

                # 添加到列表
                selected_agents.append((current_pos, checkpoint_agent))
                current_pos += 1
                if current_pos >= 6:
                    current_pos = 1  # 环绕，跳过位置0

            # 创建完整的对手列表
            opponent_agents = [None] * 6  # 初始化为None

            # 将位置0设置为None（将是我们的学习智能体）
            opponent_agents[0] = None

            # 在位置1设置随机智能体
            opponent_agents[1] = random_agent

            # 填充模型智能体
            for pos, agent in selected_agents:
                if pos != 1:  # 跳过位置1，因为它已经是随机智能体
                    opponent_agents[pos] = agent

            # 用随机智能体填充任何剩余的位置
            for i in range(6):
                if opponent_agents[i] is None and i != 0:
                    opponent_agents[i] = RandomAgent(i)

            return opponent_agents

        # 选择初始对手智能体
        opponent_agents = select_random_checkpoints()

        # 添加：在训练开始前进行初始评估
        print("初始评估...")
        initial_profit_vs_mixed = evaluate_against_checkpoint_agents(
            learning_agent, opponent_agents, num_games=100
        )
        profits_vs_checkpoints.append(initial_profit_vs_mixed)
        print(f"初始平均利润 vs 混合对手: {initial_profit_vs_mixed:.2f}")
        writer.add_scalar("Performance/ProfitVsMixed", initial_profit_vs_mixed, 0)

        # 还对随机智能体进行评估，作为基线比较
        initial_profit_random = evaluate_against_random(
            learning_agent, num_games=500, num_players=6
        )
        profits.append(initial_profit_random)
        print(f"初始平均利润 vs 随机: {initial_profit_random:.2f}")
        writer.add_scalar("Performance/ProfitVsRandom", initial_profit_random, 0)

        # 包装智能体以确保正确的视角
        opponent_wrappers = [None] * 6
        for pos in range(6):
            if pos != 0 and opponent_agents[pos] is not None:
                opponent_wrappers[pos] = AgentWrapper(opponent_agents[pos])

        # 训练循环
        for iteration in range(1, additional_iterations + 1):
            learning_agent.iteration_count = iteration
            start_time = time.time()

            # 按指定间隔刷新对手
            if iteration % refresh_interval == 1:
                print(f"在迭代 {iteration} 刷新对手池")
                opponent_agents = select_random_checkpoints()

                # 重新包装智能体
                opponent_wrappers = [None] * 6
                for pos in range(6):
                    if pos != 0 and opponent_agents[pos] is not None:
                        opponent_wrappers[pos] = AgentWrapper(opponent_agents[pos])

            print(f"混合模型训练迭代 {iteration}/{additional_iterations}")

            # 运行遍历以收集数据
            print("  收集数据中...")
            for t in range(traversals_per_iteration):
                # 为了公平起见，旋转按钮位置
                button_pos = t % 6

                # 创建一个新的扑克游戏
                state = pokers.State.from_seed(
                    n_players=6,
                    button=button_pos,
                    sb=1,
                    bb=2,
                    stake=200.0,
                    seed=random.randint(0, 10000),
                )

                # 执行CFR遍历
                learning_agent.cfr_traverse(state, iteration, opponent_wrappers)

            # 跟踪遍历时间
            traversal_time = time.time() - start_time
            writer.add_scalar("Time/Traversal", traversal_time, iteration)

            # 训练优势网络
            print("  Training advantage network...")
            adv_loss = learning_agent.train_advantage_network()
            losses.append(adv_loss)
            print(f"  Advantage network loss: {adv_loss:.6f}")

            # 将损失记录到tensorboard
            writer.add_scalar("Loss/Advantage", adv_loss, iteration)
            writer.add_scalar(
                "Memory/Advantage", len(learning_agent.advantage_memory), iteration
            )

            # 每隔几次迭代，训练策略网络并评估
            if iteration % 10 == 0 or iteration == additional_iterations:
                print("  训练策略网络中...")
                strat_loss = learning_agent.train_strategy_network()
                print(f"  策略网络损失: {strat_loss:.6f}")
                writer.add_scalar("Loss/Strategy", strat_loss, iteration)

                # 对当前对手进行评估
                print("  对当前混合对手进行评估中...")
                avg_profit_vs_mixed = evaluate_against_checkpoint_agents(
                    learning_agent, opponent_agents, num_games=100
                )
                profits_vs_checkpoints.append(avg_profit_vs_mixed)
                print(f"  平均利润 vs 混合对手: {avg_profit_vs_mixed:.2f}")
                writer.add_scalar(
                    "Performance/ProfitVsMixed", avg_profit_vs_mixed, iteration
                )

                # 还对随机智能体进行评估，作为基线比较
                print("  对随机智能体进行评估中...")
                avg_profit_random = evaluate_against_random(
                    learning_agent, num_games=500, num_players=6
                )
                profits.append(avg_profit_random)
                print(f"  平均利润 vs 随机: {avg_profit_random:.2f}")
                writer.add_scalar(
                    "Performance/ProfitVsRandom", avg_profit_random, iteration
                )

                # Save the model
                # model_path = f"{save_dir}/deep_cfr_mixed_iter_{iteration}"
                # learning_agent.save_model(model_path)
                # print(f"  Model saved to {model_path}")

                # 还保存为可选择的训练模型
                if iteration % 100 == 0:
                    t_model_path = f"{checkpoint_dir}/{training_model_prefix}mixed_iter_{iteration}.pt"
                    torch.save(
                        {
                            "iteration": iteration,
                            "advantage_net": learning_agent.advantage_net.state_dict(),
                            "strategy_net": learning_agent.strategy_net.state_dict(),
                        },
                        t_model_path,
                    )
                    print(f"  训练模型已保存到 {t_model_path}")

            # 定期保存模型
            if iteration % checkpoint_frequency == 0:
                checkpoint_path = f"{save_dir}/mixed_checkpoint_iter_{iteration}.pt"
                torch.save(
                    {
                        "iteration": iteration,
                        "advantage_net": learning_agent.advantage_net.state_dict(),
                        "strategy_net": learning_agent.strategy_net.state_dict(),
                        "losses": losses,
                        "profits": profits,
                        "profits_vs_checkpoints": profits_vs_checkpoints,
                    },
                    checkpoint_path,
                )
                print(f"  模型已保存到 {checkpoint_path}")

            elapsed = time.time() - start_time
            writer.add_scalar("Time/Iteration", elapsed, iteration)
            print(f"  Iteration completed in {elapsed:.2f} seconds")
            print(f"  Advantage memory size: {len(learning_agent.advantage_memory)}")
            print(f"  Strategy memory size: {len(learning_agent.strategy_memory)}")
            writer.add_scalar(
                "Memory/Strategy", len(learning_agent.strategy_memory), iteration
            )

            # 提交tensorboard日志
            writer.flush()
            print()

        # 使用更多游戏进行最终评估
        print("最终评估...")

        # 对随机对手进行评估
        avg_profit_random = evaluate_against_random(learning_agent, num_games=500)
        print(f"最终表现 vs 随机: 每场游戏平均利润: {avg_profit_random:.2f}")
        writer.add_scalar("Performance/FinalProfitVsRandom", avg_profit_random, 0)

        # 对混合对手进行评估
        avg_profit_vs_mixed = evaluate_against_checkpoint_agents(
            learning_agent, opponent_agents, num_games=500
        )
        print(f"最终表现 vs 混合对手: 每场游戏平均利润: {avg_profit_vs_mixed:.2f}")
        writer.add_scalar("Performance/FinalProfitVsMixed", avg_profit_vs_mixed, 0)

        writer.close()

        return learning_agent, losses, profits, profits_vs_checkpoints

    finally:
        # Restore the original cfr_traverse method to avoid side effects
        DeepCFRAgent.cfr_traverse = original_cfr_traverse


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练用于扑克的Deep CFR智能体")
    parser.add_argument("--verbose", action="store_true", help="启用详细输出")
    parser.add_argument("--iterations", type=int, default=1000, help="CFR迭代次数")
    parser.add_argument(
        "--traversals", type=int, default=200, help="每次迭代的遍历次数"
    )
    parser.add_argument("--save-dir", type=str, default="models", help="保存模型的目录")
    parser.add_argument(
        "--log-dir", type=str, default="logs/deepcfr", help="tensorboard日志目录"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="继续训练的模型路径"
    )
    parser.add_argument(
        "--self-play", action="store_true", help="针对模型训练而不是随机智能体"
    )
    parser.add_argument("--mixed", action="store_true", help="针对混合模型训练")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="models", help="包含模型模型的目录"
    )
    parser.add_argument(
        "--model-prefix", type=str, default="t_", help="选择池中包含的模型前缀"
    )
    parser.add_argument(
        "--refresh-interval", type=int, default=1000, help="刷新对手池的间隔"
    )
    parser.add_argument(
        "--num-opponents", type=int, default=5, help="要选择的模型对手数量"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="启用严格的错误检查，对无效游戏状态抛出异常",
    )
    args = parser.parse_args()

    # 用于调试的严格训练
    set_strict_checking(args.strict)

    # # 使用subprocess运行tensorboard
    # import subprocess
    # tensorboard_process = subprocess.Popen(
    #     f"tensorboard --logdir={args.log_dir}", shell=True
    # )
    print("\n查看训练进度:")
    print("然后在浏览器中打开 http://localhost:6006")

    if args.mixed:
        print(f"使用来自 {args.checkpoint_dir} 的模型开始混合模型训练")
        agent, losses, profits, profits_vs_checkpoints = train_with_mixed_checkpoints(
            checkpoint_dir=args.checkpoint_dir,
            training_model_prefix=args.model_prefix,
            additional_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            save_dir=args.save_dir,
            log_dir=args.log_dir + "_mixed",
            refresh_interval=args.refresh_interval,
            num_opponents=args.num_opponents,
            verbose=args.verbose,
        )
    elif args.checkpoint and args.self_play:
        print(f"开始针对模型 {args.checkpoint} 的自玩训练")
        agent, losses, profits = train_against_checkpoint(
            checkpoint_path=args.checkpoint,
            additional_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            save_dir=args.save_dir,
            log_dir=args.log_dir + "_selfplay",
            verbose=args.verbose,
        )
    elif args.checkpoint:
        print(f"从模型 {args.checkpoint} 继续训练")
        agent, losses, profits = continue_training(
            checkpoint_path=args.checkpoint,
            additional_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            save_dir=args.save_dir,
            log_dir=args.log_dir + "_continued",
            verbose=args.verbose,
        )
    else:
        print(f"开始Deep CFR训练，共 {args.iterations} 次迭代")
        print(f"每次迭代使用 {args.traversals} 次遍历")
        print(f"日志将保存到: {args.log_dir}")
        print(f"模型将保存到: {args.save_dir}")

        # 训练Deep CFR智能体
        agent, losses, profits = train_deep_cfr(
            num_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            num_players=6,
            player_id=0,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            verbose=args.verbose,
        )

    print("\n训练总结:")
    print(f"最终损失: {losses[-1]:.6f}")
    if profits:
        print(f"最终平均利润 vs 随机: {profits[-1]:.2f}")
    if "profits_vs_checkpoints" in locals() and profits_vs_checkpoints:
        print(f"最终平均利润 vs 混合模型: {profits_vs_checkpoints[-1]:.2f}")


