# Reverse Curriculum SQ-GRPO 训练分析报告
## 实验配置: `e3_n32_lr1.0e-6_kl0.01_eb0.005` (20260306_214457)

---

## 一、实验配置概览

| 参数 | 值 |
|------|-----|
| 模型 | Qwen2.5-VL-7B (8.29B params) |
| GPU | 2× H200 140GB, FSDP |
| 算法 | SQ-GRPO (adv_estimator=grpo) |
| KL系数 | 0.01 (disable_kl=false) |
| Entropy Bonus | 0.005 |
| Clip Ratio | low=0.2, high=0.3, dual=3.0 |
| Group Advantage | 0.0 (禁用) |
| Rollout N | 32 |
| Batch Size | 64 |
| LR | 1.0e-6 |
| Epochs | 3 |
| Max Response Length | 5000 |
| Total Steps | 129 |

---

## 二、评测结果汇总

### 2.1 多Benchmark对比表

| 模型/Checkpoint | TFQ-G Per-Q | TFQ-G Group | TFQ-I Per-Q | TFQ-I Group | MCQ-G |
|----------------|:-----------:|:-----------:|:-----------:|:-----------:|:-----:|
| **Baseline (Qwen2.5-VL-7B)** | 96.54% (475/492) | 80.00% (40/50) | - | - | 58.00% (29/50) |
| **Prior Best (e2 n32 step12, lr=5e-7)** | 97.15% (478/492) | 84.00% (42/50) | - | - | 62.00% (31/50) |
| **KL=0.1 (e2 n32 step12, lr=5e-7)** | - | - | - | - | 60.00% (30/50) |
| **本次 e3 step88** | **99.59%** (490/492) | **96.00%** (48/50) | 92.28% (454/492) | 50.00% | 59.38%* (19/32) |
| **本次 e3 step129** | **99.59%** (490/492) | **96.00%** (48/50) | 92.48% (455/492) | 56.00% | **64.00%** (32/50) |

> *注: MCQ step88 仅加载32条数据（非完整50条），结果可能不完整，需排查推理是否中断。

### 2.2 与基线对比提升幅度

| Benchmark | 基线 → 本次best | 提升 |
|-----------|:--------------:|:----:|
| TFQ-G Per-Q | 96.54% → 99.59% | **+3.05pp** |
| TFQ-G Group | 80.00% → 96.00% | **+16.00pp** ⬆️⬆️⬆️ |
| MCQ-G | 58.00% → 64.00% | **+6.00pp** ⬆️ |

### 2.3 TFQ-G 错误分析

step88和step129 **完全相同**的2个错误:
- Group 24 Q5: model=F, gt=T  
- Group 37 Q10: model=T, gt=F

这意味着模型在这两个样本上存在稳定的认知偏差（非随机型错误），进一步训练很难修正。

### 2.4 MCQ-G 错误分析 (step129)

18/50 错误 (64%正确)，错误模式:
- 多个ID一致性错误（ID 1, 4, 6, 7, 8, 10, 11, 15等在step88也出错）→ **核心难题**
- step129比step88新增: ID 23, 31, 35, 38, 40, 41, 49 (7个新增错误)
- step129修复: step88中 ID 0, 3的错误在step129中被修复

MCQ任务的准确率从基线58%→64%有所提升，但improvement有限。

---

## 三、训练动态分析

### 3.1 训练收敛曲线

```
Train Accuracy (step → value):
  1 → 0.903   11 → 0.954   22 → 0.992   33 → 0.982   44 → 0.996
 55 → 0.997   66 → 0.997   77 → 0.997   88 → 0.994   99 → 1.000
110 → 0.999  121 → 1.000  129 → 0.988

Train Group Accuracy:
  1 → 0.564   11 → 0.670   22 → 0.936   33 → 0.845   44 → 0.963
 55 → 0.968   66 → 0.972   77 → 0.979   88 → 0.945   99 → 0.999
110 → 0.999  121 → 1.000  129 → 0.900
```

**关键观察:**
1. **快速收敛**: 前22步 (约35%第一遍epoch) Train Acc就达到0.992
2. **训练集几近完美**: step 64起多次达到 1.000
3. **epoch边界震荡**: step 33 (epoch=0.77) 处出现明显GA下降0.936→0.845, step 88-89 (epoch≈2.0)处也出现下降0.945→0.900, 与epoch重洗牌有关

### 3.2 验证集趋势

```
Val Accuracy:  0.943 → 0.951 → 0.978 → 0.988 → 0.992 → 0.992 → 0.992 → 0.996(best) → 0.996 → 0.992 → 0.994 → 0.994 → 0.994
Val Group Acc: 0.620 → 0.660 → 0.800 → 0.900 → 0.940 → 0.940 → 0.940 → 0.960(best) → 0.960 → 0.940 → 0.960 → 0.960 → 0.960
```

**关键观察:**
1. **最优点 step 77**: Val Acc=0.996, Val GA=0.960
2. **step 44-55即达plateau**: Val GA=0.940后再难突破，仅step 77偶发达到0.960
3. **step 99出现微弱退化**: Val GA从0.960降至0.940，之后恢复→训练后期开始震荡
4. **明显过拟合**: Train GA从step 64起频繁达1.000，而Val GA停留在0.940-0.960

### 3.3 KL散度分析

```
KL_loss: 0.001 → 0.021 (缓慢上升，非常小)
PPO_KL: ~3.8e-05 → ~4.0e-05 (几乎恒定，接近0)
```

**诊断: KL约束完全失效**
- PPO_KL在训练全程保持在 ~4e-05 量级，意味着策略几乎没有偏离参考策略
- kl_coef=0.01 × PPO_KL≈4e-05 = kl_loss≈4e-07，对总loss贡献可忽略
- 这与之前实验的发现一致: KL惩罚在当前配置下完全不起作用

### 3.4 Entropy下降与回弹

```
Entropy: 1.001 → 0.497(min,step44) → 0.550(step77) → 0.728(step88) → 0.864(step129)
```

**关键观察:**
1. **前44步大幅下降**: 1.001→0.497，模型快速变得确信
2. **Epoch 2开始回弹**: step 77起entropy从0.55恢复到0.72-0.86
3. **这种回弹可能与epoch边界的数据reshuffling和重复学习有关**
4. Entropy Bonus=0.005太小，未能有效维持探索性

### 3.5 梯度范数 (Gradient Norm)

```
GradNorm: 2.133 → 0.855 → 1.065 → 0.897 → 0.291 → 0.158 → 0.095 → 0.128 → 0.269 → 0.535 → 0.285 → 0.022 → 0.511
```

- 前22步: 快速下降 2.1→0.9 (主要学习阶段)
- 中期稳定: step 44-77 在 0.1-0.3 之间 (已基本收敛)
- 后期波动: step 99起回升到 0.3-0.5 (可能学到noise)

### 3.6 Response Length

```
RespLen: 483 → 445 → 356(min) → 360 → 382 → 394 → 389 → 391 → 389 → 418 → 414 → 404 → 401
```

回答长度从483逐步下降到356后又回升，最终稳定在400左右。这可能代表模型学会了更简洁地回答。

---

## 四、关键发现与诊断

### 4.1 ✅ 达到预期的方面

1. **TFQ-Grouped 大幅提升**: Group Acc从基线80% → 96%，提升16pp，是迄今为止最强结果
2. **训练稳定性**: 3个epoch训练平稳，无发散或崩溃现象
3. **TFQ-Grouped性能一致性**: step88和step129给出完全相同的结果(96%)和相同错误，表明模型在TFQ任务上收敛到了稳定解

### 4.2 ⚠️ 未达预期的方面

1. **MCQ提升有限**: 58% → 64% (仅+6pp)，表明TFQ训练对MCQ的泛化有限
2. **TFQ-Individual Group Acc偏低**: 仅56%（step129），远低于TFQ-Grouped的96%
   - 这说明**分组评估(grouped)和逐题评估(individual)之间存在巨大差异**
   - Grouped模式下模型得以利用上下文一致性，Individual模式下该优势消失
3. **KL约束仍然无效**: PPO_KL≈4e-05，说明kl_coef=0.01完全不够
4. **过拟合明显**: 训练集GAa=1.000 vs 验证集GA=0.960，gap≈4pp

### 4.3 TFQ-Grouped vs TFQ-Individual 差异深入分析

| Metric | TFQ-Grouped | TFQ-Individual | 差异 |
|--------|:-----------:|:--------------:|:----:|
| Per-Q Accuracy (step129) | 99.59% | 92.48% | -7.11pp |
| Group Accuracy (step129) | 96.00% | 56.00% | -40.00pp |

**巨大的Group Accuracy差异(40pp!)揭示了关键问题:**
- Grouped模式下，模型看到同一图片的所有问题，可以利用题目间的一致性约束
- Individual模式下每题独立推理，暴露了模型真实的单题判断能力
- Per-Q也有7pp差距，说明grouped context确实帮助了per-question accuracy
- **这意味着模型的实际隐喻理解能力提升不如TFQ-Grouped数字所示那么大，部分accuracy来自题目间的"互相校准"**

---

## 五、训练epoch数分析

### 最优停止点建议

| 训练阶段 | Steps | Val GA | 状态 |
|---------|:-----:|:------:|:----:|
| Epoch 1 (~step 43) | 44 | 0.940 | 快速提升结束 |
| Epoch 1.5 (~step 65) | 66 | 0.940 | 无进一步提升 |
| **Epoch 1.8 (~step 77)** | **77** | **0.960** | **验证集最优** |
| Epoch 2 (~step 86) | 88 | 0.960 | 维持 |
| Epoch 2.3 (~step 99) | 99 | 0.940 | 轻微退化 |
| Epoch 3 (step 129) | 129 | 0.960 | 恢复（但MCQ↑有些意外） |

**结论**: 约2个epoch (step 77-88) 是最优停止区间。3个epoch训练并未带来Val GA提升，但MCQ有轻微提升(需独立验证step77 MCQ)。

---

## 六、优化方向建议

### P0: 高优先级

#### 1. Learning Rate与Epoch数协同优化
- 当前lr=1e-6 × 3epoch 导致过拟合
- **建议**: lr=1e-6 × 1.5-2 epochs, 或 lr=5e-7 × 3 epochs
- 或引入cosine decay schedule

#### 2. MCQ联合训练/多任务
- TFQ单任务训练对MCQ泛化不足
- **建议**: 在reverse curriculum中混入MCQ形式的训练样本(例如20% MCQ + 80% TFQ)
- 或设计two-stage: Stage1 MCQ→Stage2 TFQ curriculum

#### 3. KL约束修复
- kl_coef=0.01完全无效，需大幅提高
- **建议**: 尝试 kl_coef=0.5 或更高，或改用adaptive KL (KL target=0.01, 动态调节coef)
- 目标: 使PPO_KL至少达到0.01-0.1量级

### P1: 中优先级

#### 4. Individual vs Grouped Gap缩小
- 40pp Group Acc差距说明模型依赖题目间上下文
- **建议**: 训练时加入individual-style prompt (单题独立推理), 强制提升单题判断能力
- 或在training loss中加入individual consistency regularization

#### 5. 早停策略
- 实现validation-based early stopping, 在Val GA连续N次checkpoint无提升时停止
- 当前实验显示 step 44-77 区间是最优，之后边际收益为0

#### 6. Entropy Bonus增大
- 当前eb=0.005对维持探索性几乎无效
- **建议**: eb=0.02-0.05, 防止模型过早collapse到确定性策略

### P2: 探索性

#### 7. MCQ step88推理完整性排查
- MCQ step88仅加载32/50条数据, 可能是推理中断
- 需检查推理脚本和step88 merged model是否完整

#### 8. Step 77 专项评测
- 验证集最优点step 77尚未在external benchmark上评测
- 建议补充step 77的TFQ-G/TFQ-I/MCQ-G评测

#### 9. 对比Grouped vs Individual训练策略
- 当前训练用的是Grouped format的TFQ
- 探索: 是否用Individual format训练能提升TFQ-I performance

---

## 七、结论

本次reverse curriculum SQ-GRPO e3实验在**TFQ-Grouped任务上取得了里程碑式突破**:
- Group Accuracy从80% → 96%, 提升16个百分点
- Per-Q Accuracy达到99.59%, 接近任务上限
- 训练稳定, 无崩溃

但同时暴露了几个重要问题:
1. **Grouped vs Individual gap巨大(40pp)**, 说明模型学到的更多是题组内一致性推理, 而非真正的单题隐喻理解提升
2. **MCQ迁移有限(+6pp)**, 需要联合训练或专门优化
3. **KL约束完全失效**, 需从机制上修复
4. **3 epochs过度训练**, 最优停止点在约2 epochs

**下一步建议**: 优先(1)补充step 77的完整评测; (2)尝试MCQ联合训练; (3)修复KL约束; (4)探索individual训练策略以缩小Grouped-Individual gap。

---

*分析日期: 2025-03-08*  
*训练日志: reverse_curriculum_sq_grpo_qwenvl2_5_7Bb_e3_n32_lr1.0e-6_kl0.01_eb0.005_20260306_214457.log*
