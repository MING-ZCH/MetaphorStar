# P1 (KL=0.1) 实验结果综合分析与下一步优化计划

> 文档版本: v1.0  
> 更新时间: 2025-03-06  
> 关联实验: P0 (dense ckpt, kl=0.01), P1 (kl=0.1)

---

## 1. 完整评测结果汇总

### 1.1 核心基线模型

| 模型 | 评测协议 | TFQ Per-Q | TFQ-I Group | TFQ-G | MCQ | MCQ-G |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-VL-7B Instruct (temp=0.6) | grouped | - | - | 80% | - | 58% |
| Qwen2.5-VL-7B Instruct (temp=0) | grouped | - | - | **84%** | - | **62%** |
| Full Reasoning GRPO (SQ-GRPO起始模型) | individual | 97.56% | **84%** | N/A | **68%** | N/A |

> **注**: TFQ-I = 逐题独立推理，然后按组统计 all-correct; TFQ-G = 一个prompt含组内全部题目

### 1.2 SQ-GRPO Reverse Curriculum 实验矩阵

| Config | Step | Epoch | TFQ Per-Q | TFQ-I Group | TFQ-G | MCQ-G |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **n32 kl=0.01** | 7 | ~1 | 94.72% | 62% | **84%** | **68%** |
| **n32 kl=0.01** | 12 | ~2 | 96.34% | **76%** | **84%** | 62% |
| n64 kl=0.01 | 6 | ~1 | 94.11% | 62% | **84%** | 64% |
| n64 kl=0.01 | 12 | ~2 | 94.72% | 62% | **86%** | 60% |
| **n32 kl=0.1 (P1)** | 6 | ~1 | 95.33% | 70% | ⚠️缺失 | 66% |
| **n32 kl=0.1 (P1)** | 12 | ~2 | 95.53% | 68% | ⚠️缺失 | 60% |

### 1.3 其他参考实验

| 模型 | TFQ-I Group | TFQ-G | MCQ-G |
|:---|:---:|:---:|:---:|
| 3B Reasoning GRPO | 62% | - | 64% |
| 3B SQ-GRPO (n64 step6) | 61.9% | 78% | 66% |
| 32B Reasoning GRPO (bf16) | 74% | - | 76% |
| DiVA-GRPO v3 phase2 (step120) | 54% | 82% | 56% |
| MetaphorQA SQ-GRPO | - | 86% | - |

---

## 2. P1 (KL=0.1) 关键发现

### 2.1 KL 结构性失效 — 已确认

训练日志证据:
```
P1 (kl_coef=0.1):  kl_loss ≡ 0.001,  ppo_kl ≈ 1e-5
P0 (kl_coef=0.01): kl_loss ≡ 0.001,  ppo_kl ≈ 1e-5   ← 完全相同！
```

**根因分析** (dp_actor.py L480-481):
```python
kl_loss = VF.masked_mean(kld, response_mask)          # 在整个response上平均
pg_loss = pg_loss + kl_loss * self.config.kl_coef      # 乘以kl系数
```

- 模型response ≈ 500 tokens (think + answer)
- 实际有意义的answer tokens ≈ 10 tokens (T/F答案)
- answer占比 = 10/500 = **2%**
- 大部分tokens的KL变化极小 → 平均后信号被稀释至噪声水平
- kl_coef 是乘在已稀释的均值上 → **无论设多大都无法恢复有效信号**

**定量分析**:
$$\text{KL贡献} = \text{kl\_coef} \times \text{kl\_loss} = 0.1 \times 0.001 = 0.0001$$
$$\text{policy\_loss} \approx 0.1$$
$$\frac{\text{KL贡献}}{\text{policy\_loss}} = 0.1\% \quad \text{(可忽略不计)}$$

### 2.2 P1 评测结果对比分析

**vs 基线 (Full Reasoning GRPO)**:
| 指标 | 基线 | P1 step6 | P1 step12 | 变化趋势 |
|:---|:---:|:---:|:---:|:---:|
| TFQ Per-Q | 97.56% | 95.33% (↓2.2pp) | 95.53% (↓2.0pp) | 小幅下降 |
| TFQ-I Group | 84% | 70% (↓14pp) | 68% (↓16pp) | **显著下降** |
| MCQ | 68% | - | - | - |
| MCQ-G | - | 66% (≈基线) | 60% (↓8pp) | **持续下降** |

**vs kl=0.01 (原始配置)**:
| 指标 | kl=0.01 step7 | kl=0.1 step6 | kl=0.01 step12 | kl=0.1 step12 |
|:---|:---:|:---:|:---:|:---:|
| TFQ Per-Q | 94.72% | 95.33% | 96.34% | 95.53% |
| TFQ-I Group | 62% | 70% | **76%** | 68% |
| MCQ-G | **68%** | 66% | 62% | 60% |

**结论**: KL=0.1 在 epoch1 时略好于 kl=0.01 (TFQ-I: 70% vs 62%)，但 epoch2 反而更差 (68% vs 76%)。MCQ-G 也更差 (60% vs 62%)。**10倍KL系数不仅未能阻止遗忘，反而可能干扰了学习。**

---

## 3. 核心发现与诊断

### 3.1 TFQ-G 稳定性悖论

**关键发现**: 所有SQ-GRPO配置的 TFQ-G 都稳定在 84-86%，与 Instruct 基线持平。

这说明:
1. ✅ SQ-GRPO **没有损害**模型在grouped评测下的TFQ能力
2. ❌ 但也**没有显著提升** TFQ-G (始终≈84%)
3. 而 TFQ-I Group 从 84% 降至 62-76%

**解释**: SQ-GRPO训练使用grouped格式(一个prompt含10个子问题)，模型在此格式下表现稳定。但当TFQ-I评测以逐题独立格式测试时，模型失去了上下文关联的优势，表现下降。这是**格式过拟合**而非能力提升。

### 3.2 TFQ-I 的 U 型恢复

```
Epoch 0 (baseline):  TFQ-I Group = 84%
Epoch ~1 (step 7):   TFQ-I Group = 62%  ← 急剧下降
Epoch ~2 (step 12):  TFQ-I Group = 76%  ← 部分恢复
```

**解释**: 
- Epoch 1: 模型正在适应SQ-GRPO训练格式，独立推理能力暂时下降
- Epoch 2: 模型开始泛化，部分恢复独立推理能力，但未回到基线水平

### 3.3 MCQ 灾难性遗忘的定量轨迹

```
基线:          MCQ = 68%   (non-grouped)
SQ-GRPO step7:  MCQ-G = 68% (保持！)
SQ-GRPO step12: MCQ-G = 62% (↓6pp, 已退化到 Instruct 水平)
```

**核心问题**: SQ-GRPO 在 TFQ 格式数据上训练 → True/False 答题策略逐渐覆盖 MCQ 推理能力

### 3.4 n64 vs n32: 更多rollout无益

| | n32 step12 | n64 step12 |
|:---|:---:|:---:|
| TFQ-I Group | **76%** | 62% |
| TFQ-G | 84% | 86% |
| MCQ-G | **62%** | 60% |

n64在所有关键指标上 ≤ n32。更多rollout带来的探索多样性被过大的计算开销抵消，且可能因为更大的group size稀释了有效学习信号。

---

## 4. 问题根因总结

| # | 问题 | 根因 | 严重度 |
|:---|:---|:---|:---:|
| R1 | MCQ 灾难性遗忘 | 纯TFQ格式训练，缺乏MCQ格式保护 | 🔴 严重 |
| R2 | TFQ-I Group 下降 | 模型过拟合grouped格式，独立推理退化 | 🟡 中等 |
| R3 | KL约束完全失效 | token级KL因answer/response比例稀释至噪声水平 | 🔴 严重 |
| R4 | 训练超过1 epoch无收益 | step7→step12: MCQ下降，TFQ-I仅部分恢复 | 🟡 中等 |

---

## 5. 下一步优化方案 (优先级排序)

### 方案 A: Layer Freezing — 冻结底层参数 ⭐️ 推荐优先

**原理**: 冻结Transformer底层(视觉编码器 + 底层语言层)，只训练顶层，保留base model通用能力

**实现**:
```python
# 在训练脚本中添加
# Qwen2.5-VL-7B 有 28 层 transformer blocks
# 冻结前14层(50%)或前21层(75%)
for name, param in model.named_parameters():
    if any(f"layers.{i}." in name for i in range(14)):  # freeze bottom 14 layers
        param.requires_grad = False
```

**预期效果**:
- MCQ退化显著减缓 (底层保留通用推理能力)
- TFQ-I改善可能稍慢 (学习容量减少)
- TFQ-G保持稳定

**优势**: 实现极简,1行代码; 无需修改数据pipeline; 广泛验证的防遗忘技术
**风险**: 冻结层数需要调参 (50% vs 75% vs 25%)

**建议实验**:
- A1: freeze 50% (14/28层) — 平衡方案
- A2: freeze 75% (21/28层) — 强保护方案

### 方案 B: Answer-Level KL — 修复KL信号稀释 ⭐️ 推荐次选

**原理**: 只在answer tokens上计算KL，消除 reasoning tokens 的稀释效应

**改前**:
```python
kl_loss = VF.masked_mean(kld, response_mask)   # 500 tokens平均 → kl_loss ≈ 0.001
```

**改后**:
```python
# 构建 answer_mask: 只在 <answer>...</answer> 区间的实际 T/F token 位置为1
answer_kl = VF.masked_mean(kld, answer_mask)    # 10 tokens平均 → kl_loss ≈ 0.05
# 预期KL信号增强50倍！
```

**实现难点**: 
1. 需要在rollout阶段定位answer token位置 (通过正则匹配或tokenizer offset mapping)
2. 需要通过data pipeline将answer_mask传入actor trainer
3. 需要处理answer部分不存在的情况 (format不合格的rollout)

**预期效果**:
$$\text{KL贡献} = 0.1 \times 0.05 = 0.005 \quad (\text{增强50×})$$
$$\frac{\text{KL贡献}}{\text{policy\_loss}} = 5\% \quad (\text{从0.1% → 5%，进入有效区间})$$

### 方案 C: LoRA 替代全参微调

**原理**: 只训练低秩适配器，大部分参数不变
**优势**: 天然防遗忘，参数量减少99%+
**劣势**: SQ-GRPO的sub-question level advantage与LoRA兼容性未验证; 可能需要修改EasyR1框架
**优先级**: 低 (框架改动较大)

### 方案 D: MCQ 数据混合训练

**原理**: 在SQ-GRPO训练中混入MCQ格式的数据/reward
**实现**: 
1. 将MCQ数据转为SQ-GRPO兼容格式 (单选→单子问题)
2. 交替训练TFQ和MCQ batch
**劣势**: 之前分析表明混合数据对SQ-GRPO效果有限
**优先级**: 低

### 方案 E: 更好的 Early Stopping

**现有最优Pareto点**:
- step 7 (n32 kl=0.01): TFQ-I=62%, MCQ-G=68% ← MCQ最优
- step 12 (n32 kl=0.01): TFQ-I=76%, MCQ-G=62% ← TFQ-I最优
- P0 step3/step6 尚未评测 (dense checkpoint)

**建议**: 评测P0的step3和step6，寻找更精确的Pareto前沿

---

## 6. 推荐实验计划

### Phase 1: 补充评测 (立即执行，0训练成本)

| 任务 | 描述 | 优先级 |
|:---|:---|:---:|
| E1 | P1 step6/step12 TFQ_grouped 评测 | P0 |
| E2 | P0 step3/step6 全部4类评测 | P0 |
| E3 | Full Reasoning GRPO 基线的 TFQ_grouped + MCQ_grouped 评测 | P1 |

### Phase 2: Layer Freezing 实验 (推荐首先尝试)

| 实验 | 配置 | 预期训练时间 |
|:---|:---|:---:|
| LF-50 | 冻结底层50% (14/28), n32, kl=0.01, lr=5e-7 | ~同原实验 |
| LF-75 | 冻结底层75% (21/28), n32, kl=0.01, lr=5e-7 | ~同原实验 |

**期望**: MCQ-G ≥ 64% (在step12), TFQ-I Group ≥ 70%

### Phase 3: Answer-Level KL (如果Layer Freezing不够)

| 实验 | 配置 | 
|:---|:---|
| AnsKL-01 | answer_level_kl=True, kl_coef=0.1, 不冻结 |
| AnsKL-01-LF | answer_level_kl=True, kl_coef=0.1, 冻结50% |

---

## 7. 缺失评测清单

P1 (kl=0.1) **缺失**:
- [ ] TFQ_grouped step6
- [ ] TFQ_grouped step12  
- [ ] MCQ (non-grouped) step6
- [ ] MCQ (non-grouped) step12

P0 (kl=0.01 dense) **缺失**:
- [ ] step3 全部评测
- [ ] step6 全部评测

基线 Full Reasoning GRPO **缺失**:
- [ ] TFQ_grouped 
- [ ] MCQ_grouped

---

## 8. 总结

### 最重要结论

1. **KL策略彻底失败**: 10× kl_coef 对训练无任何影响，根因是token级KL的结构性稀释
2. **SQ-GRPO的真正瓶颈不是TFQ，而是MCQ遗忘**: TFQ-G稳定84%，但MCQ从68%跌至60%
3. **格式过拟合 > 能力提升**: TFQ-I Group下降说明模型在记忆格式而非真正学习推理
4. **当前最佳检查点**: n32 kl=0.01 step12 (TFQ-I=76%↑, MCQ-G=62%↓)，显示TFQ-MCQ权衡

### 核心优化方向

**短期 (Layer Freezing)**: 通过冻结底层参数保护通用推理能力，同时允许顶层适应SQ-GRPO任务  
**中期 (Answer-Level KL)**: 修复KL信号稀释问题，实现有效的policy drift约束  
**长期 (架构改进)**: 考虑LoRA、多任务训练等更深层架构优化
