# DeepCFR Poker AI

基于深度反事实后悔最小化（Deep Counterfactual Regret Minimization）的6人无限注德州扑克AI训练系统。

> **版本 0.3.0 重要更新**：项目已从`pokers-db`迁移至`pokerkit`库，实现完整的6人游戏支持和跨平台兼容。详见[迁移说明](#pokerkit迁移说明)。

## 项目简介

本项目实现了一个可扩展的扑克AI训练框架，使用Deep CFR算法训练智能体在6人无限注德州扑克中进行博弈。该系统支持多种训练模式，包括对抗随机智能体、自博弈训练以及混合对手训练。

### 核心特性

- **Deep CFR算法实现**：结合神经网络的CFR算法，用优势网络和策略网络分别学习遗憾值和策略
- **多训练模式**：
  - 从零开始训练（对抗随机智能体）
  - 断点续训
  - 自博弈训练（对抗历史检查点）
  - 混合对手训练（对抗随机选择的多个检查点）
- **优先经验回放**：使用优先级经验回放缓冲区，提高学习效率
- **连续下注大小**：支持连续的下注大小空间，而非离散的下注选项
- **完善的日志系统**：详细的游戏错误日志，方便调试和分析
- **TensorBoard集成**：实时监控训练进度和性能指标
- **完整6人游戏支持**：基于PokerKit，支持2-10人游戏（推荐6人）
- **跨平台支持**：支持Linux、macOS和Windows

## 技术栈

- **深度学习框架**：PyTorch
- **扑克引擎**：PokerKit (University of Toronto Computer Poker Research Group)
- **可视化**：TensorBoard
- **Python版本**：3.11+（推荐3.12）

## 安装

```bash
# 克隆项目
git clone https://github.com/Anning01/TexasDeepCFR.git
cd deepCFR

# 创建虚拟环境（推荐使用 uv）
uv venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
uv sync # 或者 pip install -e .
```

## 快速开始

### 基础训练

从零开始训练一个AI智能体：

```bash
python -m train --iterations 1000 --traversals 200
```

### 继续训练

从检查点继续训练：

```bash
python -m train --checkpoint models/checkpoint_iter_1000.pt --iterations 1000
```

### 自博弈训练

对抗已训练的检查点进行自博弈：

```bash
python -m train --checkpoint models/checkpoint_iter_1000.pt --self-play --iterations 1000
```

### 混合对手训练

对抗随机选择的多个检查点：

```bash
python -m train --mixed --checkpoint-dir models --model-prefix t_ --iterations 1000 --num-opponents 5 --refresh-interval 100
```

### 与AI对战

```bash
python -m main --models-dir models --position 0
```

## 训练参数说明

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--iterations` | CFR迭代次数 | 1000 |
| `--traversals` | 每次迭代的遍历次数 | 200 |
| `--save-dir` | 模型保存目录 | models |
| `--log-dir` | TensorBoard日志目录 | logs/deepcfr |
| `--checkpoint` | 检查点路径（用于继续训练） | None |
| `--self-play` | 启用自博弈模式 | False |
| `--mixed` | 启用混合对手模式 | False |
| `--verbose` | 显示详细输出 | False |
| `--strict` | 启用严格错误检查 | False |

## 训练成本估算

根据实际测试数据：

- **单次迭代耗时**：~0.55秒
- **每次迭代包含**：200次遍历
- **内存占用**：优势网络和策略网络各300,000样本

### 规模化训练成本

**研究级训练（1000次迭代）**
- 训练时间：~9分钟
- 适合：概念验证、算法调试

**进阶训练（10万次迭代）**
- 训练时间：~15小时
- 适合：初步性能评估

**商业化训练（10亿轮自博弈）**
- 估算训练时间：
  - 按当前配置：约 17,400年（单机单GPU）
  - 需要：大规模分布式训练集群
  - 建议：至少100-1000个GPU并行训练
  - 实际时间：数月到一年（取决于集群规模）

**成本建议**：
- 对于商业化应用，建议使用云计算平台（如AWS、GCP、Azure）的GPU集群
- 预估成本：数十万到数百万美元（取决于训练规模和云服务商）
- 可考虑混合方案：关键阶段使用高性能GPU，其他阶段使用CPU集群

## 项目结构

```
deepCFR/
├── core/
│   ├── deepcfr.py      # Deep CFR智能体实现
│   └── model.py        # 神经网络模型和状态编码
├── train.py            # 训练脚本（多种训练模式）
├── main.py             # 人机对战脚本
├── random_agent.py     # 随机智能体基准
├── game_logger.py      # 游戏日志工具
├── settings.py         # 全局设置
└── models/             # 保存的模型检查点
```

## 监控训练进度

启动TensorBoard：

```bash
tensorboard --logdir=logs/deepcfr
```

然后在浏览器中打开 http://localhost:6006

### 关键指标

- **Performance/Profit**：对抗随机智能体的平均每局收益
- **Loss/Advantage**：优势网络训练损失
- **Loss/Strategy**：策略网络训练损失
- **Memory/Advantage**：优势网络记忆缓冲区大小
- **Memory/Strategy**：策略网络记忆缓冲区大小
- **Time/Iteration**：每次迭代耗时

## 算法原理

Deep CFR通过以下方式工作：

1. **CFR遍历**：模拟游戏树，计算每个决策点的反事实遗憾值
2. **优势网络**：学习预测每个行动的累积遗憾值
3. **策略网络**：基于遗憾匹配学习近似纳什均衡策略
4. **下注大小预测**：网络同时学习最优的下注大小
5. **优先经验回放**：优先采样重要的训练样本，提高学习效率

## 性能优化

当前实现包含以下优化：

- ✅ 优先经验回放（Prioritized Experience Replay）
- ✅ 梯度裁剪防止梯度爆炸
- ✅ 动态下注大小调整
- ✅ 批量训练和经验重放
- ✅ GPU加速（如果可用）

## 已知限制与未来改进

### 当前限制

1. **训练规模**：需要数十亿次迭代才能达到商业化水平
2. **单机训练**：当前为单机实现，缺乏分布式训练支持
3. **对手多样性**：主要对抗随机智能体，可能导致策略过拟合

### 计划改进

- [ ] 分布式训练支持（多GPU/多机）
- [ ] 更高效的状态表示学习
- [ ] 对手建模和适应性策略
- [ ] 支持更多扑克变体（短牌等）
- [ ] 实时推理优化和部署工具
- [ ] 集成专业扑克数据库进行监督学习预训练

## PokerKit迁移说明

### 为什么选择PokerKit？

原项目使用的`pokers-db`库只提供macOS ARM64的wheel包，无法在Linux服务器上运行。我们评估了多个方案后选择PokerKit：

**PokerKit的优势：**
- ✅ **完整的6人游戏支持** - 支持2-10人德州扑克
- ✅ **完全支持Linux平台** - 纯Python实现，无需编译
- ✅ **学术级质量** - University of Toronto计算机扑克研究组开发
- ✅ **发表在IEEE Transactions on Games** - 经过同行评审
- ✅ **完整的文档和示例** - 易于集成和扩展
- ✅ **活跃维护** - 持续更新和社区支持

**为什么不选RLCard？**
- ❌ RLCard的no-limit-holdem只支持2人游戏
- ❌ 原代码设计为6人游戏，改为2人需要重新调整策略
- ❌ 缺乏灵活性，难以扩展到多人场景

### 迁移后的变化

**API变化：**
- 代码已通过适配层(`pokerkit_adapter.py`)最小化修改
- 外部API保持兼容，原有代码基本无需改动
- 完整保留连续下注大小功能

**技术实现：**

适配层(`pokerkit_adapter.py`)提供了与`pokers-db`完全兼容的API：

- `State` - 游戏状态类（支持6人游戏）
- `Action` - 动作类（连续下注金额）
- `ActionEnum` - 动作枚举（Fold=0, Check=1, Call=2, Raise=3）
- `Card` - 扑克牌类
- `PlayerState` - 玩家状态类
- `Stage` - 游戏阶段枚举

所有现有代码通过`import pokerkit_adapter as pokers`使用适配层，无需修改核心逻辑。

**兼容性测试：**
```bash
# 运行适配层测试（验证6人游戏）
python test_pokerkit_adapter.py

# 预期输出：所有测试通过，显示6人游戏完整支持
```

### Python版本要求

PokerKit需要Python 3.11+（推荐3.12）。如果你的Python版本较低，请升级：

```bash
# 使用uv升级Python
uv python install 3.12

# 重新创建虚拟环境
uv venv --python 3.12
source .venv/bin/activate  # Linux/Mac
```

### 迁移优势总结

从`pokers-db` → `rlcard` → `pokerkit`的迁移路径实现了：

1. **功能完整性** - 保留原始6人游戏设计 ✅
2. **跨平台支持** - 在Linux服务器上顺利运行 ✅
3. **连续动作空间** - 保留任意金额加注功能 ✅
4. **学术可信度** - 使用经过同行评审的库 ✅
5. **长期维护** - 活跃的社区和持续更新 ✅

## 贡献

欢迎提交Issue和Pull Request！


## 致谢

本项目基于以下研究和开源项目：

- Deep Counterfactual Regret Minimization (Brown et al., 2019)
- Pluribus: Superhuman AI for multiplayer poker (Brown & Sandholm, 2019)
- [PokerKit](https://github.com/uoftcprg/pokerkit): A Comprehensive Python Library for Fine-Grained Multi-Variant Poker Game Simulations (University of Toronto)
- PokerKit论文发表于IEEE Transactions on Games, 2025

---

**⚠️ 免责声明**：本项目仅供学术研究和教育目的。请遵守当地关于扑克和博弈的法律法规。
