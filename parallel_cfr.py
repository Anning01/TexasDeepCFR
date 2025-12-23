"""
并行化 CFR 遍历 - 利用多核CPU
"""
import multiprocessing as mp
from functools import partial
import random
import pokerkit_adapter as pokers
from copy import deepcopy


def run_single_traversal(args):
    """
    单个CFR遍历的工作函数（在子进程中运行）

    参数:
        args: (agent_state_dict, iteration, seed, num_players)

    返回:
        收集到的经验数据
    """
    agent_state_dict, iteration, seed, num_players = args

    # 在子进程中重新创建agent（避免序列化整个agent）
    from core.deepcfr import DeepCFRAgent
    agent = DeepCFRAgent(player_id=0, num_players=num_players, device='cpu')

    # 加载网络权重（用于推理，不需要训练）
    agent.advantage_net.load_state_dict(agent_state_dict['advantage_net'])
    agent.strategy_net.load_state_dict(agent_state_dict['strategy_net'])

    # 创建随机对手
    from random_agent import RandomAgent
    random_agents = [RandomAgent(i) for i in range(num_players)]

    # 创建游戏
    state = pokers.State.from_seed(
        n_players=num_players,
        button=seed % num_players,
        sb=1,
        bb=2,
        stake=200.0,
        seed=seed
    )

    # 执行遍历（收集经验）
    agent.cfr_traverse(state, iteration, random_agents)

    # 返回收集到的经验
    return {
        'advantage_memory': list(agent.advantage_memory.buffer),
        'strategy_memory': list(agent.strategy_memory)
    }


def parallel_cfr_traversals(agent, iteration, num_traversals, num_players=6, num_workers=None):
    """
    并行执行多个CFR遍历

    参数:
        agent: DeepCFRAgent实例
        iteration: 当前迭代次数
        num_traversals: 遍历次数
        num_players: 玩家数量
        num_workers: 工作进程数（None=CPU核心数）
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    # 准备agent状态（只序列化网络权重）
    agent_state_dict = {
        'advantage_net': agent.advantage_net.state_dict(),
        'strategy_net': agent.strategy_net.state_dict()
    }

    # 准备任务参数
    tasks = [
        (agent_state_dict, iteration, random.randint(0, 100000), num_players)
        for _ in range(num_traversals)
    ]

    # 并行执行
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(run_single_traversal, tasks)

    # 合并结果到主agent的内存中
    for result in results:
        for exp in result['advantage_memory']:
            agent.advantage_memory.add(exp)
        for exp in result['strategy_memory']:
            agent.strategy_memory.append(exp)

    return len(results)


if __name__ == '__main__':
    # 测试代码
    print("测试并行CFR遍历...")
    from core.deepcfr import DeepCFRAgent
    import time

    agent = DeepCFRAgent(player_id=0, num_players=6, device='cpu')

    # 串行版本
    print("\n串行执行 10 次遍历:")
    start = time.time()
    from random_agent import RandomAgent
    random_agents = [RandomAgent(i) for i in range(6)]
    for i in range(10):
        state = pokers.State.from_seed(
            n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=i
        )
        agent.cfr_traverse(state, 1, random_agents)
    serial_time = time.time() - start
    print(f"耗时: {serial_time:.2f}秒")

    # 并行版本
    print(f"\n并行执行 10 次遍历 (使用 {mp.cpu_count()} 核):")
    agent2 = DeepCFRAgent(player_id=0, num_players=6, device='cpu')
    start = time.time()
    parallel_cfr_traversals(agent2, 1, 10, num_workers=mp.cpu_count())
    parallel_time = time.time() - start
    print(f"耗时: {parallel_time:.2f}秒")

    print(f"\n加速比: {serial_time/parallel_time:.2f}x")
