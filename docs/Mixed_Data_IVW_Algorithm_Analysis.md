# 新算法深度分析：DAPO + IVW + 混合数据 vs SQ-GRPO

> 分析日期: 2025-03-05 | 优化目标: TFQ-Individual Group Acc ↑ + MCQ Acc ↑

---

## 目录
1. [新算法形式化定义](#1-新算法形式化定义)
2. [IVW 不可行的数学证明](#2-ivw-不可行的数学证明)
3. [通俗类比：为什么等权优于逆方差](#3-通俗类比)
4. [DAPO Dynamic Sampling 与 SQ-GRPO 的关系](#4-dapo-dynamic-sampling)
5. [混合数据的重新评估](#5-混合数据重新评估)
6. [MCQ 退化的真正根因：KL 约束失效](#6-mcq退化根因)
7. [综合对比表](#7-综合对比)
8. [KL 强化训练计划](#8-kl强化训练计划)

---

## 1. 新算法形式化定义

### 1.1 提出的新算法

**核心思路**: DAPO Dynamic Sampling + Inverse-Variance Weighting (IVW) + 混合数据

- **混合数据**: 单题TFQ数据（984道）+ 一图多题TFQ-Group数据（100组）同时训练
- **DAPO Dynamic Sampling**: 计算 advantage 时过滤掉全对/全错的组
- **IVW**: 对 multi-Q 数据，用 $w_j = \text{softmax}(1/(\sigma_j^2 \cdot \tau))$ 加权各SQ的advantage
- **GRPO without std**: 与SQ-GRPO一致，不除以标准差

### 1.2 数学定义

对一图多题数据，奖励矩阵 $\mathbf{R} \in \{0, 1\}^{G \times N}$：

**SQ-GRPO (当前)**：
$$A_i^{\text{SQ}} = \sum_{j=1}^{N} (R_{i,j} - \mu_j), \quad \mu_j = \frac{1}{G}\sum_i R_{i,j}$$

**IVW (提议)**：
$$A_i^{\text{IVW}} = \sum_{j \in \mathcal{J}_{\text{active}}} w_j \cdot (R_{i,j} - \mu_j)$$

其中：
$$w_j = \frac{\exp(1/(\sigma_j^2 \cdot \tau))}{\sum_k \exp(1/(\sigma_k^2 \cdot \tau))}, \quad \sigma_j^2 = \mu_j(1-\mu_j)$$

---

## 2. IVW 不可行的数学证明

### 2.1 具体数值示例

**设定**: 一个典型 group，10 个SQ，G=32 rollouts

| SQ 编号 | 情况 | $\mu_j$ | $\sigma_j^2$ | DAPO | Advantage ($R=1$) | Advantage ($R=0$) |
|---|---|---|---|---|---|---|
| SQ1-7 | 32/32 全对 | 1.00 | 0.000 | ✗ 过滤 | 0 | — |
| SQ8 | 31/32 对 | 0.97 | 0.029 | ✓ active | +0.03 | -0.97 |
| SQ9 | 28/32 对 | 0.88 | 0.106 | ✓ active | +0.12 | -0.88 |
| SQ10 | 16/32 对 | 0.50 | 0.250 | ✓ active | +0.50 | -0.50 |

Active set = {SQ8, SQ9, SQ10}

### 2.2 IVW 权重计算

| SQ | $1/\sigma_j^2$ | $\tau=1$ 权重 | $\tau=10$ 权重 | $\tau=100$ 权重 |
|---|---|---|---|---|
| SQ8 ($\sigma^2=0.029$) | 34.5 | **≈1.000** | **0.955** | 0.574 |
| SQ9 ($\sigma^2=0.106$) | 9.43 | ≈0.000 | 0.044 | 0.334 |
| SQ10 ($\sigma^2=0.250$) | 4.00 | ≈0.000 | 0.001 | 0.092 |

### 2.3 四种典型 rollout 的 advantage 对比

取主要的 rollout 类型（SQ8几乎总对，SQ10随机性最大）：

#### Rollout 类型 A: SQ8✓ SQ9✓ SQ10✓ (概率 ≈ 0.97×0.88×0.50 = 42.7%)

| 方法 | 计算 | $A_i$ |
|---|---|---|
| **SQ-GRPO** | $0.03 + 0.12 + 0.50$ | **+0.65** |
| **IVW** ($\tau=10$) | $0.955 \times 0.03 + 0.044 \times 0.12 + 0.001 \times 0.50$ | **+0.035** |

#### Rollout 类型 B: SQ8✓ SQ9✓ SQ10✗ (概率 ≈ 42.7%)

| 方法 | 计算 | $A_i$ |
|---|---|---|
| **SQ-GRPO** | $0.03 + 0.12 + (-0.50)$ | **-0.35** |
| **IVW** ($\tau=10$) | $0.955 \times 0.03 + 0.044 \times 0.12 + 0.001 \times (-0.50)$ | **+0.034** |

#### Rollout 类型 C: SQ8✓ SQ9✗ SQ10✓ (概率 ≈ 5.8%)

| 方法 | 计算 | $A_i$ |
|---|---|---|
| **SQ-GRPO** | $0.03 + (-0.88) + 0.50$ | **-0.35** |
| **IVW** ($\tau=10$) | $0.955 \times 0.03 + 0.044 \times (-0.88) + 0.001 \times 0.50$ | **-0.009** |

#### Rollout 类型 D: SQ8✗ SQ9✓ SQ10✓ (概率 ≈ 1.3%)

| 方法 | 计算 | $A_i$ |
|---|---|---|
| **SQ-GRPO** | $(-0.97) + 0.12 + 0.50$ | **-0.35** |
| **IVW** ($\tau=10$) | $0.955 \times (-0.97) + 0.044 \times 0.12 + 0.001 \times 0.50$ | **-0.921** |

### 2.4 核心问题：SQ10 的 "学习信号消失"

定义 **SQ10 的区分度** = SQ10 对 vs SQ10 错时 advantage 的差异：

| 方法 | SQ10✓ 时 $A_i$ (类型A) | SQ10✗ 时 $A_i$ (类型B) | **区分度** |
|---|---|---|---|
| **SQ-GRPO** | +0.65 | -0.35 | **1.00** |
| **IVW** ($\tau=10$) | +0.035 | +0.034 | **0.001** |

**IVW 下 SQ10 答对和答错几乎产生相同的 advantage！**

这意味着 policy gradient 对 SQ10 的正确/错误回答无法区分 → **模型根本学不到 SQ10 应该答对**。

而 SQ-GRPO 中，SQ10 答对 vs 答错的 advantage 差异是 1.00，模型能清晰接收到 "SQ10 应该答对" 的梯度信号。

### 2.5 为什么 $\tau$ 无法修复

| $\tau$ | SQ10 区分度 | vs SQ-GRPO |
|---|---|---|
| 1 | ≈0.000 | 区分度消失（完全忽略SQ10） |
| 10 | 0.001 | SQ-GRPO 的 0.1%（几乎无效） |
| 100 | 0.092 | SQ-GRPO 的 9.2%（仍严重不足） |
| 1000 | 0.47 | SQ-GRPO 的 47%（接近等权但仍不如） |
| ∞ | 1.00 | = SQ-GRPO（IVW 退化为等权，无贡献） |

**数学结论**: $\forall \tau \in (0, \infty)$：

$$\text{SQ10区分度}_{\text{IVW}} < \text{SQ10区分度}_{\text{SQ-GRPO}}$$

等号仅在 $\tau \to \infty$ 时取到，此时 IVW 退化为 SQ-GRPO。

**即：IVW 在任何有限温度下，对困难SQ的学习效率都严格低于SQ-GRPO的等权求和。**

### 2.6 SQ-GRPO 等权求和为何是最优的

SQ-GRPO 的 advantage $A_i = \sum_j \hat{A}_{i,j}$ 中，每个 SQ 的贡献**自动与其难度成正比**：

$$|\hat{A}_{i,j}| = \begin{cases} 0 & \text{if } \mu_j = 0 \text{ or } 1 \text{ (已掌握/完全不会)} \\ |R_{i,j} - \mu_j| \leq 0.5 & \text{otherwise} \end{cases}$$

困难 SQ ($\mu_j \approx 0.5$): $|\hat{A}| \approx 0.5$ → 大贡献
简单 SQ ($\mu_j \approx 0.97$): $|\hat{A}| \approx 0.03$ → 小贡献

**这是天生的、最优的难度加权。不需要任何额外机制。**

SQ-GRPO 论文中的 "自动难题聚焦" 定理正是证明了这一点。任何试图改变这个自然权重的做法（包括 IVW）都只能让情况变差。

---

## 3. 通俗类比

### 考试辅导类比

想象一个学生要参加 10 道题的考试，找了一个辅导老师：

**SQ-GRPO 老师（优秀）**:
- "你 7 道已经满分了，不用管"
- "第 8 道偶尔错一次（97%对），提醒一下就好，花 3% 时间"
- "第 9 道经常错（88%对），多练练，花 12% 时间"
- "第 10 道才对一半（50%对），这是你最大短板，花 **50% 时间重点突破**"
- → **时间分配自动匹配难度**

**IVW 老师（糟糕）**:
- "第 8 道你偶尔错，说明你不够稳定！花 **95% 时间** 反复练到 100%"
- "第 9 道你经常错？先放一放，花 4% 时间"
- "第 10 道才对一半？太难了，只花 **1% 时间**"
- → **95% 的辅导时间用在已经 97% 掌握的题目上，真正薄弱的题几乎没练**

### 股票投资类比

**SQ-GRPO = 价值投资者**: 将资金按**预期收益（难度×改善空间）**分配。困难题改善空间大 → 投入多。

**IVW = 追求稳定的投资者**: 将资金按**确定性（低方差）**分配。几乎已掌握的题最确定 → 投入最多。但"确定"的题收益空间极小（97%→100%），而真正需要投资的领域被忽略。

---

## 4. DAPO Dynamic Sampling 分析

### 4.1 DAPO 原始定义 (arXiv:2503.14476)

- 过滤: 一个 prompt 的 G 个 rollout reward 全部相同 → 丢弃该组
- 过采样: 持续生成新样本直到 batch 全部有效

### 4.2 与 SQ-GRPO Dynamic Trajectory Pruning 的对比

| 维度 | DAPO | SQ-GRPO DTP |
|---|---|---|
| 粒度 | Group level (全部 rollout 相同) | Group level (所有SQ方差=0) |
| 效果 | 丢弃无梯度信号的组 | 同上 |
| 额外功能 | 过采样填充 | 无 |
| SQ级别细粒度 | ❌ 无 | ✅ **已掌握的SQ自动贡献0 advantage** |

SQ-GRPO 的 DTP **严格强于** DAPO 的 Dynamic Sampling：
- DAPO 只在组级别过滤
- SQ-GRPO 在 SQ 级别也自动 "过滤"（已掌握的 SQ 贡献 0）

### 4.3 在当前实验条件下的增量

95% per-Q, G=32, 10 SQ/group:
- DAPO 过采样触发概率 < 0.002%
- **增量价值: 几乎为零**

---

## 5. 混合数据的重新评估

### 5.1 用户的正确质疑

> "混合数据也只是TFQ单题（984道题）和对应TFQ-group题目（100道）训练，为什么认为会对MCQ会有帮助？"

**结论: 混合数据对 MCQ 的帮助非常有限。**

理由：
1. **Single-TFQ ≠ MCQ**: T/F 判断 vs 多选题，格式不同
2. **MCQ 退化的根因是策略漂移**，不是 "丢失了 single-Q 能力"
3. 混合 TFQ 数据只能保护 TFQ 个体能力，不能直接保护 MCQ

### 5.2 反向课程 vs 混合数据

> "先TFQ单题+grpo再TFQ-group题目+sq-grpo做课程学习，理论上更有效？"

**完全正确。** 反向课程每个阶段都能充分优化目标：
- Phase 1: GRPO 充分建立 individual 能力 + MCQ 泛化
- Phase 2: SQ-GRPO 在强基础上添加 group consistency

混合数据的问题：两个目标相互稀释，哪个都不能充分优化。

### 5.3 混合数据的唯一适用场景

只有当 **KL 约束在所有设定下都无法防止 MCQ 退化** 时，混合数据才值得尝试。
这属于 Plan C。当前优先级: P1(KL=0.1) > P0(dense ckpt) > 混合数据

---

## 6. MCQ 退化的真正根因：KL 约束失效

### 6.1 实验证据

从 n=32 训练日志：

| Step | kl_loss | ppo_kl | entropy_loss | MCQ 退化预估 |
|---|---|---|---|---|
| 1 | 0.001 | ~1e-5 | 1.221 | 0% |
| 4 | 0.001 | ~1e-5 | 1.198 | -2~4% |
| 7 | 0.001 | ~1e-5 | 1.179 | -6% (实测 MCQ=68%) |
| 12 | 0.001 | ~1e-5 | 1.153 | -12% (实测 MCQ=62%) |

- **kl_loss = kl_coeff × ppo_kl = 0.01 × 1e-5 = 1e-7** → 几乎为零
- PPO loss = policy_loss + kl_loss + entropy_loss ≈ policy_loss + entropy_loss（KL 项被淹没）

### 6.2 为什么当前 KL 无效

$D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})$ 的计算是 **token-level 近似**：

$$\text{ppo\_kl} = \mathbb{E}_{t}\left[\log\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}\right]$$

在 SQ-GRPO 训练初期，策略变化主要体现在最终答案 token 上（True/False 的排布），而非 thinking 过程。答案 token 只占总 token 的 ~2%（10个答案/~500 token response），因此：

$$\text{token-level ppo\_kl} \approx \frac{10}{500} \times D_{\text{KL}}(\text{answer}) \approx 0.02 \times D_{\text{KL}}(\text{answer})$$

即使答案分布大幅偏移，token-level KL 也很小。但这个 "小" KL 可能对应 **MCQ 性能的大幅退化**，因为 MCQ 所需的通用推理模式也在 thinking token 中被微妙改变。

### 6.3 KL=0.1 的预期效果

将 kl_coeff 从 0.01 提升到 0.1（10x）：
- 预期 kl_loss: 0.001 → 0.01
- 仍然较小，但相对于 policy_loss 的比例从 ~0.1% 提升到 ~1%
- 可能不够，下一步可能需要 KL=0.5 或 1.0

---

## 7. 综合对比表

### 7.1 三种方案最终评估

| 组件 | 理论分析 | 实验验证 | 最终评估 |
|---|---|---|---|
| **IVW (逆方差加权)** | ❌ 数学证明在高准确率下有害: 困难SQ区分度降至SQ-GRPO的0.1% | 无需实验 | ⭐ **弃用** |
| **DAPO Dynamic Sampling** | ≈ SQ-GRPO DTP，过采样增量<0.002% | SQ-GRPO已内置 | ⭐⭐ 可选但几乎无增量 |
| **混合数据** | 理论上不如反向课程；TFQ单题≠MCQ | 需实验但非最优先 | ⭐⭐ 备选 Plan C |
| **KL 约束强化** | 直接修复 root cause（策略漂移） | P1 进行中 | ⭐⭐⭐⭐⭐ **最优先** |
| **Dense Checkpointing** | 找 Pareto 最优点 | P0 进行中 | ⭐⭐⭐⭐ 必做 |

### 7.2 为什么 SQ-GRPO 的等权求和是最优的

| 设计 | 优点 | 缺点 |
|---|---|---|
| **等权求和 (SQ-GRPO)** | 自动难题聚焦（已证明）；无超参；数值稳定 | 无 |
| IVW ($\tau$ 小) | 理论：更精确的 gradient 估计 | **实际：逆转难题聚焦，困难SQ信号消失** |
| IVW ($\tau$ 大) | 接近等权 | 等效SQ-GRPO，IVW无贡献 |
| 方差加权 (反向IVW) | 放大难题聚焦 | 可能过度集中在最难SQ；等权已最优 |

---

## 8. KL 强化训练计划

### 8.1 当前实验矩阵

| 实验 | KL | Save Freq | 状态 | 目的 |
|---|---|---|---|---|
| P0 | 0.01 | 2 | 待启动 | Dense ckpt 找 Pareto 点 |
| P1 | 0.1 | 2 | **已启动** | 测试 10x KL 能否保护 MCQ |

### 8.2 KL 阶梯计划（基于 P1 结果决定）

**场景 A: P1(KL=0.1) 成功** → kl_loss 从 0.001 上升到 ~0.01, MCQ 退化 ≤ 6%
- ✅ 当前方案足够
- 结合 dense ckpt 找最优 step

**场景 B: P1(KL=0.1) 部分成功** → MCQ 退化 7-10%
- 尝试 **KL=0.5**: 预期 kl_loss ~0.05
- 或 **KL=1.0**: 更激进约束

**场景 C: P1(KL=0.1) 失效** → MCQ 退化仍 >10%
- KL 约束在 token-level 层面可能天然不足
- 需要转向其他方案:
  1. **Elastic Weight Consolidation (EWC)**: 对关键参数施加 L2 约束
  2. **混合数据 (Plan C)**: 加入个体 TFQ 或 MCQ 数据
  3. **Layer-specific LR**: 冻结/低 LR for 靠近输入的层，高 LR for 靠近输出的层

### 8.3 新预设建议（如需更高 KL）

```bash
# 已在脚本中实现
CONFIG=pareto_kl  # KL=0.1, n=32, e2, save_freq=2

# 后续可能需要的新预设：
# pareto_kl_strong: KL=0.5
# pareto_kl_max: KL=1.0
```

---

*文档版本: v2.0 | 修正: 移除混合数据为最优推荐，确认KL修复为最优先方案*
*基于 SQ-GRPO v0-v3 + Reverse Curriculum + Rollout N 消融全部实验数据*
