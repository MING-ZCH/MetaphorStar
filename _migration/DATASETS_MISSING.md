# 缺失数据集 / 模型清单

迁移分支只含代码，本文件列出在新集群运行训练/评测前需要重新获取的数据。

图例：
- 🟢 已在 Hugging Face Hub，新集群直接 `huggingface-cli download` 即可
- 🟡 仅 zch-dev 本地存在，**必须**手动同步或重建
- 🔴 历史快照/中间产物，**优先级低**

---

## EasyR1-latest

源路径：`/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest`

| 类型 | 路径 | 大小 | 状态 | 备注 |
|------|------|------|------|------|
| checkpoints | `save/` | ~294 GB | 🔴 | 训练产物，不迁移 |
| logs | `*.log` `wandb/` | - | 🔴 | 已 exclude |

EasyR1 数据集主要由配置文件中的 HF repo id 指定，运行时按需下载，无本地必须迁移项。
请参照 `examples/*.yaml` 中 `data.train_files` 字段。

---

## MetaphorStar

源路径：`/mnt/shared-storage-user/zhangchenhao/work/MetaphorStar`

| 类型 | 路径 | 状态 | 备注 |
|------|------|------|------|
| 模型 | `MetaphorStar-3B/`, `MetaphorStar-7B/` | 🟡 | 训练得到的本地模型权重 |
| 数据集 | `MetaphorQA/`, `MetaphorQA-SQ/`, `MetaphorQA-SQ-Full/`, `MetaphorQA-SQ-test/`, `MetaphorQA-test/`, `MetaphorQA_Small/` | 🟡 | 自建 QA 数据集 |
| 课程学习中间产物 | `reverse_curriculum_*/` | 🔴 | 历史实验 |
| 评测集 | `evaluation/II-Bench/` | 🟢 | 公开 benchmark，可重新下载 |

⚠️ **新集群运行前**：需要将 `MetaphorQA*` 数据集与 `MetaphorStar-*` 模型上传到 HF Hub 或通过对象存储/外部盘传输；
脚本中所有写死的本地路径需按 PATH_REWRITES.md 改写。

---

## StepcountModel

源路径：`/mnt/shared-storage-user/zhangchenhao/work/StepcountModel`

### 训练后模型权重（`model/`，🟡）
- StepCount-7B-SFT-10k-high
- StepCount-7B-SFT-30k-high
- stepcount_qwen2.5_vl-7b_epoch3_resize_vocab_prompt_full_fix_pixels_fix_points_resized_points
- stepcount_qwen2.5_vl-7b_epoch3_resize_vocab_prompt_full_fix_pixels_fix_points_resized_points_images_resized_one_count_per_time
- stepcount_qwen2.5_vl-7b_epoch3_resize_vocab_prompt_full_resized_points_one_count_per_time_all
- stepcount_qwen2.5_vl-7b_epoch3_resize_vocab_prompt_full_resized_points_without_reasoning_one_count_per_time_all
- stepcount_qwen2.5_vl-7b_epoch3_resize_vocab_prompt_resized_points_all_with_long_ranage_one_count_per_time
- stepcount_qwen2.5_vl-7b_epoch3_resize_vocab_prompt_resized_points_all_with_long_ranage_without_point_reasoning_one_count_per_time
- stepcount_qwen2.5_vl-7b_epoch3_resize_vocab_prompt_resized_points_long_range_one_count_per_time

### 数据集 `dataset/`

| 路径 | 状态 | 推测 HF 位置 | 说明 |
|------|------|--------------|------|
| StepCountQA-RL/ | 🟢 | 见 `upload_merged_rl_hf.py` | RL 主数据集 |
| StepCountQA-RL-Dense-Plus/ | 🟢 | 已上传 | dense reward 变体 |
| StepCountQA-RL-Easy_to_Hard_Small/ | 🟢 | 已上传 | 课程小集 |
| StepCountQA-RL-Eval/ | 🟢 | - | RL 评测集 |
| StepCountQA-RL-Traj_0_10/ 及变体 | 🟢 | - | 轨迹 0-10 |
| StepCountQA-RL-Traj_11_50/ 及变体 | 🟢 | `upload_traj_11_50_*_hf.py` | 轨迹 11-50 |
| StepCountQA-SFT/ | 🟢 | `upload_sft_to_hf.py` | SFT 主数据集 |
| CountBenchQA/ | 🟢 | 公开 | 评测 |
| Robot_bench/ | 🟢 | 公开 | 评测 |
| pixmo-test/ | 🟢 | `allenai/pixmo-points` 子集 | 评测 |
| vlms_bias_data/ | 🟡 | bias 评测集 | |
| stepcount-500/ | 🟡 | 500 样本子集 | |
| filtered_json/ | 🟡 | 中间过滤产物 | |
| grouped_data/ | 🔴 | 分组中间产物 (17 GB) | |
| sam_mask/ | 🔴 | SAM 分割中间产物 | |
| viz_samples/ | 🔴 | 可视化样本 | |
| eval/* 及其它中间产物 | 🔴 | 实验中间产物 | |
| 根目录 `stepcount_reasoning_dataset_*.json` | 🟡 | 训练用 reasoning 数据 (~6 GB) | |

### 已保留迁移内容
- `eval/*.txt` `eval/*.json` 36 个摘要报告（图片缓存子目录已剔除）
- `dataset/*.py` `dataset/*.sh` `dataset/*.md` 全部脚本

### ⚠️ 必查清单
1. 用 HF Hub `MING-ZCH/` 名下 dataset 列表对账 🟢 项实际 repo id 与最新 revision。
2. 模型权重（🟡）若已上 HF，按 `download.py` 拉取；若仅本地，用对象存储或 `huggingface-cli upload` 转运。
3. 所有路径按 PATH_REWRITES.md 在配置中修正。
