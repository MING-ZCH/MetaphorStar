# DiVA-GRPO v3: Two-Phase Curriculum Training

## 目录
1. [背景与问题诊断](#1-背景与问题诊断)
2. [v3 设计方案](#2-v3-设计方案)
3. [代码修改清单](#3-代码修改清单)
4. [使用指南](#4-使用指南)
5. [实验计划](#5-实验计划)
6. [测试验证](#6-测试验证)

---

## 1. 背景与问题诊断

### 1.1 现有评测结果对比

| Model | TFQ (per-group) | TFQ (per-Q) | MCQ | OSQ |
|-------|-----------------|-------------|-----|-----|
| **Target: Reasoning GRPO** | **70%** | **95.53%** | **74%** | **3.22** |
| Baseline (Qwen2.5-VL-7B) | 80% (grouped) | — | 58% | — |
| SQ-GRPO v0 (ga=0) | **86%** (grouped) | — | — | — |
| DiVA v2-M1 (ga=0.3+warmup) | 84% (grouped) | — | 60% | — |
| DiVA v2-M2 (ga=0.3) | 76% (grouped) | — | 64% | — |
| DiVA v2-M3 (ga=0.5) | 72% (grouped) | — | 66% | — |

### 1.2 关键发现：Group Advantage (GA) 信号是有害的

**证据链 (TFQ grouped)**: SQ-GRPO v0 (ga=0) = 86% > M1 (ga=0.3+wu) = 84% > M2 (ga=0.3) = 76% > M3 (ga=0.5) = 72%

**ga_coeff 越高，TFQ 性能越差。** 根本原因分析：

1. **Winner-take-all 动态**: 100组小数据中，soft power mode 放大了"幸运"rollout的信号
2. **Entropy 坍塌**: M2 entropy 从 1.087 降至 0.645 (-40.7%)，M3 response length 从 471 降至 388
3. **格式失配**: Grouped 训练 (numbered T/F list) ≠ Individual 评测 (single T/F)
4. **MCQ transfer**: 最好的 MCQ=66%，距离目标 74% 差距大

### 1.3 核心矛盾

- **Grouped 训练的优势**: 数据效率高 (100组×10SQ = 1000个学习信号), SQ-GRPO 的粒度优势
- **Individual 评测的要求**: 需要模型能回答单独的 T/F 问题
- **GA 信号有害**: 组间信号引入噪声，破坏了 SQ-GRPO 的粒度优势

---

## 2. v3 设计方案

### 2.1 核心策略：Two-Phase Curriculum + Entropy 保护

```
Phase 1: Clean SQ-GRPO (ga=0) on grouped data
    → 学习视觉推理能力 (per-SQ variance-aware)
    → 保持 entropy, 防止 mode collapse
    
Phase 2: Standard GRPO on individual data  
    → 适配 individual 评测格式
    → 低学习率微调，保留 Phase 1 能力
```

### 2.2 Phase 1: Clean SQ-GRPO

| 参数 | v0 (baseline) | v2 (M1) | v3 Phase 1 |
|------|--------------|---------|------------|
| `algorithm` | sq_grpo | sq_grpo | sq_grpo |
| `group_advantage_coeff` | 0.0 | 0.3 | **0.0** |
| `kl_coef` | 0.01 (default) | 0.01 | **0.02** |
| `entropy_bonus_coeff` | — | — | **0.01** |
| `total_epochs` | 3 | 3 | **4** |
| `rollout_n` | 8 | 8 | 8 |
| `lr` | 1e-6 | 1e-6 | 1e-6 |
| data | MetaphorQA-SQ | MetaphorQA-SQ | MetaphorQA-SQ |

**理论依据**:
- **ga=0.0**: 回归 v0 的强劲表现，移除有害的 GA 信号
- **kl_coef=0.02**: 2倍 KL penalty 防止过度偏离参考策略
- **entropy_bonus=0.01**: 显式鼓励 policy 保持探索，防止 entropy 坍塌
- **epochs=4**: 比 v0 多1轮，让 entropy bonus 和 KL 充分发挥保护作用

### 2.3 Phase 2: Standard GRPO

| 参数 | Phase 2 Setting | 说明 |
|------|----------------|------|
| `algorithm` | grpo | 标准 GRPO |
| `kl_coef` | 0.02 | 保持稳定 |
| `entropy_bonus_coeff` | 0.01 | 保持探索 |
| `total_epochs` | 2 | 轻量微调 |
| `rollout_n` | 16 | 标准 GRPO 需更多 rollout |
| `lr` | **5e-7** | 低学习率防灾难性遗忘 |
| data | MetaphorQA (individual) | 984 个独立 T/F 问题 |
| model_path | Phase 1 checkpoint | 自动检测 |

**理论依据**:
- **低学习率 5e-7**: 保留 Phase 1 学到的视觉推理能力
- **更多 rollout (n=16)**: Individual format 需更多样本估计 advantage
- **Only 2 epochs**: 格式适配不需要太多训练

### 2.4 Entropy Bonus 机制

代码实现位于 `dp_actor.py`:

```python
# entropy_loss = -masked_mean(log_probs, response_mask) > 0
# 当 entropy_loss 越大 (策略越随机), pg_loss 越小 → 鼓励探索
if self.config.entropy_bonus_coeff > 0:
    pg_loss = pg_loss - self.config.entropy_bonus_coeff * entropy_loss
```

- `entropy_loss` 已有计算 (原本只用于 metrics)
- coeff=0.01: 适度鼓励探索，不会主导优化方向
- 当 coeff=0 时完全不影响原始行为 (向后兼容)

---

## 3. 代码修改清单

### 3.1 修改文件 (3个)

| 文件 | 修改内容 | 行数变化 |
|------|---------|-----------|
| `verl/trainer/config.py` | 新增 `entropy_bonus_coeff` 字段 + post_init wiring | +2 行 |
| `verl/workers/actor/config.py` | 新增 `entropy_bonus_coeff` init=False 字段 | +1 行 |
| `verl/workers/actor/dp_actor.py` | 新增 entropy bonus 逻辑 | +4 行 |

### 3.2 新增文件 (3个)

| 文件 | 说明 |
|------|------|
| `examples/qwen2_5_vl_7b_MetaphorQA_diva_grpo_v3.sh` | 两阶段训练脚本 |
| `tests/test_diva_grpo_v3.py` | v3 CPU 测试套件 (27 tests) |
| `docs/diva_grpo_v3_design.md` | 本文档 |

### 3.3 备份文件

所有修改的原始文件已备份为 `*.v2backup`:
- `verl/trainer/config.py.v2backup`
- `verl/workers/actor/config.py.v2backup`
- `verl/workers/actor/dp_actor.py.v2backup`

### 3.4 详细代码变更

#### config.py (AlgorithmConfig)
```python
# Line 91 (新增):
entropy_bonus_coeff: float = 0.0  # Entropy bonus in policy loss (0=disabled)
```

#### config.py (PPOConfig.post_init)
```python
# Line 137 (新增):
self.worker.actor.entropy_bonus_coeff = self.algorithm.entropy_bonus_coeff
```

#### actor/config.py (ActorConfig)
```python
# Line 99 (新增):
entropy_bonus_coeff: float = field(default=0.0, init=False)
```

#### dp_actor.py (loss computation)
```python
# After pg_loss calculation (新增 4 行):
# DiVA-GRPO v3: Entropy preservation bonus
if hasattr(self.config, 'entropy_bonus_coeff') and self.config.entropy_bonus_coeff > 0:
    pg_loss = pg_loss - self.config.entropy_bonus_coeff * entropy_loss
    metrics["actor/entropy_bonus"] = (self.config.entropy_bonus_coeff * entropy_loss).detach().item()
```

---

## 4. 使用指南

### 4.1 完整两阶段训练

```bash
# 使用默认参数 (Phase 1: 4 epochs → Phase 2: 2 epochs)
PHASE=both bash examples/qwen2_5_vl_7b_MetaphorQA_diva_grpo_v3.sh

# 自定义参数
PHASE=both PHASE1_EPOCHS=3 PHASE2_EPOCHS=1 KL_COEF=0.03 ENTROPY_BONUS=0.02 \
    bash examples/qwen2_5_vl_7b_MetaphorQA_diva_grpo_v3.sh
```

### 4.2 分阶段训练

```bash
# 只运行 Phase 1
PHASE=1 bash examples/qwen2_5_vl_7b_MetaphorQA_diva_grpo_v3.sh

# 只运行 Phase 2 (指定 Phase 1 checkpoint)
PHASE=2 PHASE1_CKPT=/path/to/phase1/global_step_XX/actor/huggingface \
    bash examples/qwen2_5_vl_7b_MetaphorQA_diva_grpo_v3.sh

# 只运行 Phase 2 (自动检测最新 Phase 1 checkpoint)
PHASE=2 bash examples/qwen2_5_vl_7b_MetaphorQA_diva_grpo_v3.sh
```

### 4.3 评测

Phase 2 输出的 checkpoint 在:
```
/mnt/shared-storage-user/work/MetaphorStar/diva_grpo_v3_phase2/global_step_XX/actor/huggingface/
```

这是标准 HuggingFace model 格式，可直接用于评测:
- Individual TFQ (per-Q accuracy, per-group accuracy)
- Grouped TFQ
- MCQ
- OSQ

---

## 5. 实验计划

### 5.1 推荐实验矩阵

| 实验 | Phase 1 | Phase 2 | 目的 |
|------|---------|---------|------|
| **E1: v3-default** | 4ep, kl=0.02, ent=0.01 | 2ep, lr=5e-7 | 默认 baseline |
| **E2: v3-long-p1** | 6ep, kl=0.02, ent=0.01 | 2ep, lr=5e-7 | 更长 Phase 1 |
| **E3: v3-high-ent** | 4ep, kl=0.02, ent=0.03 | 2ep, ent=0.03 | 更强 entropy |
| **E4: v3-no-p2** | 4ep, kl=0.02, ent=0.01 | — (skip) | Phase 1 only ablation |
| **E5: v3-p2-only** | — (use base model) | 2ep, lr=5e-7 | Phase 2 from base |

### 5.2 关键监控指标

训练阶段:
- `actor/entropy_loss`: 应稳定在 1.0+ (Phase 1), 不低于 0.8 (Phase 2)
- `actor/entropy_bonus`: 新增 metric，应在 0.01-0.05 范围
- `actor/pg_loss`: 应稳定下降
- `critic/reward/mean`: 应稳步上升

评测阶段:
- TFQ per-group accuracy (target ≥70%)
- TFQ per-question accuracy (target ≥95%)
- MCQ accuracy (target ≥74%)
- OSQ score (target ≥3.22)

### 5.3 预期效果分析

**Phase 1 预期**:
- TFQ grouped: ≥86% (匹配或超过 v0)
- Entropy: 保持在 0.9+ (vs v2-M2 的 0.645 坍塌)
- Response length: 保持 ≥450 tokens

**Phase 2 预期**:
- Individual TFQ: 模型学会回答单独 T/F 问题
- MCQ 提升: Phase 2 的 individual training 应帮助 MCQ (also individual format)
- Grouped TFQ: 可能略微下降 (格式切换代价), 但 per-group accuracy 应 ≥70%

---

## 6. 测试验证

### 6.1 v3 新测试 (27/27 PASS)
```
Test Group 1: Config Wiring (5 tests) — 源码检查，无需 ray
Test Group 2: Clean SQ-GRPO (3 tests) — ga=0 数值验证
Test Group 3: Full Pipeline (3 tests) — 完整 advantage 链路
Test Group 4: Entropy Bonus (3 tests) — 公式和边界条件
Test Group 5: Two-Phase (2 tests) — 阶段兼容性
Test Group 6: Edge Cases (6 tests) — 数值稳定性
Test Group 7: v2 Backward (2 tests) — 向后兼容
Test Group 8: Reward Functions (3 tests) — 奖励函数正确性
```

运行: `python3 tests/test_diva_grpo_v3.py`

### 6.2 v2 回归测试 (37/37 PASS)
```
10 test groups covering reward functions, mean-centering,
pruning, normalization, N-scaling, gradient explosion,
integration, data preprocessing, training simulation, and
old vs new algorithm comparison.
```

运行: `python3 tests/test_sq_grpo.py`

---

## 附录: 设计决策记录

### 为什么不继续优化 GA 信号？
尽管理论上 Group Advantage 可以传递组间信息, 但实验明确表明:
1. 在小数据 (100组) 上, GA 更容易过拟合
2. GA 放大了训练噪声 (soft power mode)
3. v0 (ga=0) 已经是 TFQ 最优

### 为什么选择两阶段而不是多目标？
1. 避免 grouped 和 individual 格式的冲突
2. 可以精确控制每阶段的超参数
3. Phase 1 checkpoint 可复用于多个 Phase 2 实验

### 为什么 Phase 2 用低学习率？
1. 防止灾难性遗忘 Phase 1 学到的视觉推理
2. Phase 2 目标是格式适配, 不是重新学习内容
3. 经验法则: fine-tuning over RL checkpoint 用 0.5x lr

