# 集群迁移完整手册

> **源集群**：`zch-dev`（PJLab）/ 个人目录 `/mnt/shared-storage-user/zhangchenhao`  
> **目标**：新集群，下文以 `$NEW_HOME` 代替新集群个人目录  
> **迁移日期**：2026-05-15  
> **迁移分支**：`migration/zch-dev-20260514`（main 分支未改动）

---

## 1. 迁移范围

| 项目 | GitHub Repo | 源目录 | 代码分支 | main 状态 |
|------|------------|--------|---------|----------|
| EasyR1-latest | `MING-ZCH/EasyR1` | `work/EasyR1-latest` | `migration/zch-dev-20260514` @ `a43a807` | `410a169`（未动） |
| MetaphorStar | `MING-ZCH/MetaphorStar` | `work/MetaphorStar` | `migration/zch-dev-20260514` @ `6ae8d65` | `70cb669`（未动） |
| StepcountModel | `MING-ZCH/StepcountModel`（新建） | `work/StepcountModel` | `migration/zch-dev-20260514` @ `b8d72df` | 无（新 repo） |

**不在本次迁移范围**：模型权重（`model/`、`save/`）、训练数据集大文件、eval 图片缓存、wandb logs。

---

## 2. 代码拉取

```bash
# 先确认代理可用（新集群若需要）
export http_proxy=http://<代理地址>
export https_proxy=http://<代理地址>
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy

git config --global http.proxy $http_proxy
git config --global https.proxy $https_proxy

# 存储凭证（替换为新 PAT）
git config --global credential.helper store
echo "https://MING-ZCH:<NEW_PAT>@github.com" > ~/.git-credentials
chmod 600 ~/.git-credentials
git config --global user.name "MING-ZCH"
git config --global user.email "MING-ZCH@users.noreply.github.com"

mkdir -p $NEW_HOME/work && cd $NEW_HOME/work

# 拉取三个 repo
for repo in EasyR1 MetaphorStar StepcountModel; do
  git clone -b migration/zch-dev-20260514 https://github.com/MING-ZCH/$repo.git
done

# 建议本地目录名与原集群一致
mv EasyR1 EasyR1-latest
```

---

## 3. 目录结构

### 3.1 EasyR1-latest（`MING-ZCH/EasyR1`）

```
EasyR1-latest/
├── _migration/                   # 迁移文档（此手册等）
├── examples/
│   ├── config.yaml               # 通用训练配置模板
│   ├── format_prompt/            # prompt 模板 jinja
│   ├── reward_function/          # 奖励函数
│   ├── qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj.sh   # StepCount 主训练脚本
│   ├── qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj_v*.sh # 历史版本
│   ├── qwen2_5_vl_3b_geo3k_grpo.sh
│   ├── qwen2_5_vl_7b_math_grpo.sh
│   └── launch_v17_experiments.sh
├── scripts/                      # 启动辅助脚本
├── tools/                        # 工具脚本
├── verl/                         # verl 框架源码（已修改）
│   ├── trainer/
│   │   ├── core_algos.py         # GRPO/BoK 算法
│   │   └── ray_trainer.py        # 主训练器
│   └── workers/
│       ├── actor/dp_actor.py
│       ├── fsdp_workers.py
│       └── reward/function.py    # 奖励函数封装
├── requirements.txt
├── setup.py
└── pyproject.toml
```

### 3.2 MetaphorStar（`MING-ZCH/MetaphorStar`）

```
MetaphorStar/
├── _migration/
├── dataset/
│   ├── README.md
│   └── construct_TFQ.py          # 数据集构建脚本
├── docs/
├── evaluation/
│   ├── eval_MCQ.py               # MCQ 评测
│   ├── eval_TFQ.py               # TFQ 评测（主要）
│   ├── eval_OSQ.py
│   ├── eval_cli.py
│   ├── eval_grouped.py
│   ├── qwen_eval_batch.py
│   ├── qwen_eval_single.py
│   ├── baseline_eval_batch.py
│   └── run_grouped_eval.sh
├── train/
│   ├── setup.py
│   ├── requirements.txt
│   ├── verl/                     # verl 框架（MetaphorStar 版本）
│   ├── scripts/
│   │   └── model_merger.py
│   └── examples/
│       ├── config.yaml
│       ├── format_prompt/
│       ├── reward_function/
│       ├── qwen2_5_vl_7b_TFQ_Data_Lite_TFQ_GRPO.sh    # 7B 主训练脚本
│       ├── qwen2_5_vl_3b_TFQ_Data_Lite_TFQ_GRPO.sh
│       └── qwen2_5_vl_32b_TFQ_Data_Lite_TFQ_GRPO.sh
├── assets/
└── README.md
```

### 3.3 StepcountModel（`MING-ZCH/StepcountModel`）

```
StepcountModel/
├── _migration/
├── dataset/                      # 数据脚本（无数据本体）
│   ├── MODIFICATION_LOG_11_50.md
│   ├── build_*.py                # 数据集构建脚本
│   ├── clean_*.py                # 数据清洗脚本
│   ├── convert_*.py              # 格式转换脚本
│   ├── eval_fix_pixels_one_count_per_time.py   # 主评测脚本
│   ├── eval_multi_task.sh        # 多任务评测脚本（含路径配置）
│   ├── pipeline_11_50_convert_and_merge.py
│   ├── upload_merged_rl_hf.py    # 上传 StepCountQA-RL-SFT-Merged
│   ├── upload_sft_to_hf.py       # 上传 StepCountQA-SFT
│   ├── upload_traj_11_50_combined_hf.py
│   ├── upload_traj_11_50_numeric_hf.py
│   ├── skills.md
│   └── missing_in_masks_metadata_stepcount_rl.json
└── eval/                         # 评测摘要报告（36 个 .txt/.json）
    ├── eval_SFT_StepCount_all_*.txt
    ├── eval_StepCount-7B-SFT-*.txt
    └── ...（图片缓存子目录已剔除）
```

---

## 4. 环境配置

### 4.1 源集群环境快照

| 项目 | 版本 |
|------|------|
| OS | Ubuntu 22.04 LTS |
| Python | 3.10（`/root/miniconda3/bin/python`） |
| torch | 2.6.0 |
| torchvision | 0.21.0 |
| transformers | 4.52.3 |
| flash-attn | 2.7.3+cu12torch2.6cxx11abiFALSE |
| vllm | 0.8.2 |
| accelerate | 1.13.0 |
| numpy | 2.4.1 |
| ray | (with default extras) |
| deepspeed | 见 `_migration/pip-base.txt` |

完整包列表见 `_migration/pip-base.txt`，conda 导出见 `_migration/env-base.yml`。

### 4.2 安装步骤

```bash
# 1. 创建 conda 环境（Python 3.10，与源端一致）
conda create -n main python=3.10 -y
conda activate main

# 2. 安装 torch + CUDA（根据新集群 CUDA 版本调整）
#    若新集群 CUDA 12.4~12.8：
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. 安装 flash-attn（必须匹配 torch 版本，找对应 wheel）
#    方式 A：从 flash-attention releases 下载预编译 whl
#    source whl: flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
#    方式 B：源码编译（慢，约 30-60 min）
pip install flash-attn --no-build-isolation

# 4. 安装 vllm
pip install vllm==0.8.2

# 5. 安装各项目依赖
cd $NEW_HOME/work/EasyR1-latest
pip install -r requirements.txt

cd $NEW_HOME/work/MetaphorStar/train
pip install -r requirements.txt

# 6. 安装 EasyR1 verl（开发模式）
cd $NEW_HOME/work/EasyR1-latest
pip install -e .

# 7. 安装 MetaphorStar verl（若与 EasyR1 冲突，考虑独立 env）
cd $NEW_HOME/work/MetaphorStar/train
pip install -e .
```

> ⚠️ EasyR1 与 MetaphorStar 都内含定制版 `verl`，安装顺序或 editable install 可能冲突，建议各自独立 conda env：
> - `conda create -n easyr1 python=3.10`
> - `conda create -n metaphorstar python=3.10`

---

## 5. 路径适配

### 5.1 全局替换

所有脚本中硬编码的旧路径前缀需替换：

```bash
OLD=/mnt/shared-storage-user/zhangchenhao
NEW=$NEW_HOME   # 例如 /home/zhangchenhao 或 /data/user/zhangchenhao

for proj in EasyR1-latest MetaphorStar StepcountModel; do
  grep -rln "$OLD" $NEW_HOME/work/$proj \
    --include='*.py' --include='*.sh' --include='*.yaml' \
    --include='*.yml' --include='*.json' \
  | xargs -r sed -i "s|$OLD|$NEW|g"
done
```

### 5.2 EasyR1-latest 关键路径清单

| 文件 | 变量/字段 | 旧值（zch-dev） | 新集群改写 |
|------|----------|----------------|-----------|
| `examples/qwen2_5_vl_7b_StepCount_0_10_grpo_interleaved_traj.sh` | `LOG_DIR` | `/mnt/.../EasyR1-latest/logs/train` | `$NEW_HOME/work/EasyR1-latest/logs/train` |
| 同上 | `STEPCOUNT_MASKS_METADATA` | `/mnt/.../StepCount-RL_masks_output/masks_metadata.json` | 见下方 masks 说明 |
| 同上 | `STEPCOUNT_MASKS_DIR` | `/mnt/.../StepCount-RL_masks_output/masks` | 见下方 masks 说明 |
| 同上 | `data.train_files` | `/mnt/.../StepcountModel/dataset/StepCountQA-RL-Traj_0_10` | 本地路径 or HF repo id |
| 同上 | `data.val_files` | `/mnt/.../StepcountModel/dataset/pixmo-test` | 本地路径 or HF repo id |
| `examples/config.yaml` | `output_path` | `'./'` | 建议改为绝对路径 `$NEW_HOME/work/EasyR1-latest/outputs` |
| `examples/config.yaml` | `ckpt_path` | `'./ckpt'` | `$NEW_HOME/work/EasyR1-latest/ckpt` |
| 所有 `*.sh` | `WANDB_API_KEY` | 硬编码 key | 改用 `wandb login` 或环境变量 |

**StepCount Masks（必须）**：  
训练时依赖 `StepCount-RL_masks_output/` 目录中的 mask 文件。该目录原在源集群 `$OLD/StepCount-RL_masks_output/`。需要：
1. 将 `masks_metadata.json` 和 `masks/` 子目录复制到新集群（约数百MB）
2. 设置环境变量：
   ```bash
   export STEPCOUNT_MASKS_METADATA=$NEW_HOME/StepCount-RL_masks_output/masks_metadata.json
   export STEPCOUNT_MASKS_DIR=$NEW_HOME/StepCount-RL_masks_output/masks
   ```

### 5.3 MetaphorStar 关键路径清单

| 文件 | 变量/字段 | 旧值（zch-dev） | 新集群改写 |
|------|----------|----------------|-----------|
| `train/examples/qwen2_5_vl_7b_TFQ_Data_Lite_TFQ_GRPO.sh` | `MODEL_PATH` | `QwenVL/Qwen2.5-VL-7B-Instruct`（HF id） | 保持 or 替换为本地缓存路径 |
| 同上 | `data.train_files` | `MING-ZCH/TFQ-Data-Lite@train`（HF id） | **已在 HF**，无需改（需 HF 登录） |
| 同上 | `data.val_files` | `MING-ZCH/TFQ-Bench-Lite@test`（HF id） | 同上 |
| `evaluation/eval_TFQ.py` | 输入 json 路径 | 运行时参数传入 | 改写为新集群路径 |
| `evaluation/run_grouped_eval.sh` | `checkpoint`/`model_path` 路径 | 若含 `/mnt/...` | 全局替换 |

### 5.4 StepcountModel 关键路径清单

| 文件 | 变量/字段 | 旧值（zch-dev） | 新集群改写 |
|------|----------|----------------|-----------|
| `dataset/eval_multi_task.sh` | `DEFAULT_MODEL_PATH` | `/mnt/.../EasyR1-latest/save/<ckpt>/actor/huggingface` | 新集群 ckpt 路径 |
| 同上 | `PIXMO_TEST_PATH` | `$STEPCOUNT_ROOT/dataset/eval/eval_dataset.json` | 保持相对路径（自动解析） |
| 同上 | `PIXMO_IMAGES_DIR` | `/mnt/.../datasets/Datasets--allenai--pixmo-count/processed_data/images` | 需迁移该图片目录 or 重新下载 |
| `dataset/upload_*.py` | `HF_TOKEN` | 硬编码 `hf_...` | 改用 `huggingface-cli login` 或环境变量 `HF_TOKEN` |
| `dataset/upload_*.py` | `_PROXY` | `http://httpproxy-headless.kubebrain.svc.pjlab.local:3128` | 改为新集群代理或删除 |
| `dataset/upload_sft_to_hf.py` | `LOCAL_DIR` | `/mnt/.../StepcountModel/dataset/StepCountQA-SFT` | `$NEW_HOME/work/StepcountModel/dataset/StepCountQA-SFT` |

---

## 6. 数据集与模型权重恢复

### 6.1 EasyR1-latest

EasyR1 的数据集通过 Hugging Face Hub 在运行时自动下载，训练脚本中以 HF repo id 指定，无本地必须迁移的数据文件。

- `data.train_files` / `data.val_files` 字段若仍是 HF repo id（如 `hiyouga/math12k@train`），确保新集群能访问 HF 即可。
- 若字段已改为本地路径（如 `StepCountQA-RL-Traj_0_10`），参照 6.3 节下载数据集。

**额外需复制**：`StepCount-RL_masks_output/`（几百 MB，不在 GitHub 代码中）

```bash
# 在 zch-dev 上打包 masks（在源集群操作）
tar -czf masks_output.tar.gz -C /mnt/shared-storage-user/zhangchenhao StepCount-RL_masks_output/
# 传输到新集群（rsync / scp / 对象存储）
rsync -avz masks_output.tar.gz <new-cluster>:$NEW_HOME/
# 在新集群解压
tar -xzf $NEW_HOME/masks_output.tar.gz -C $NEW_HOME/
```

### 6.2 MetaphorStar

| 资产 | HF Repo / 位置 | 下载命令 |
|------|---------------|---------|
| 训练数据 `TFQ-Data-Lite` | `MING-ZCH/TFQ-Data-Lite`（HF，🟢） | 训练时自动拉取 |
| 评测集 `TFQ-Bench-Lite` | `MING-ZCH/TFQ-Bench-Lite`（HF，🟢） | 训练时自动拉取 |
| 底座模型 `Qwen2.5-VL-7B-Instruct` | HF `Qwen/Qwen2.5-VL-7B-Instruct`（🟢） | `huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct` |
| 训练后模型 `MetaphorStar-3B` / `MetaphorStar-7B` | 🟡 仅 zch-dev 本地 | 见下方 |
| 数据集 `MetaphorQA*` 系列 | 🟡 仅 zch-dev 本地 | 见下方 |
| `evaluation/II-Bench/` | 🟢 公开 benchmark | 参照 II-Bench 官方仓库重新下载 |

**本地 🟡 资产迁移方案**（选其一）：

```bash
# 方案 A：通过 HF Hub 中转（推荐，需要私有 repo 权限）
# 在 zch-dev 上传（设 DRY_RUN=False 后运行）
huggingface-cli upload MING-ZCH/MetaphorStar-7B \
    /mnt/shared-storage-user/zhangchenhao/work/MetaphorStar/MetaphorStar-7B \
    --repo-type model --private

# 在新集群下载
huggingface-cli download MING-ZCH/MetaphorStar-7B \
    --repo-type model \
    --local-dir $NEW_HOME/work/MetaphorStar/MetaphorStar-7B

# 方案 B：rsync 直接传输
rsync -avz --max-size=2M \
    /mnt/shared-storage-user/zhangchenhao/work/MetaphorStar/MetaphorQA/ \
    <new-cluster>:$NEW_HOME/work/MetaphorStar/MetaphorQA/
```

### 6.3 StepcountModel

#### 已上传 HF 的数据集（🟢，按 repo_id 下载）

```bash
HF_HOME=$NEW_HOME/datasets  # 统一存放路径

huggingface-cli login  # 输入 HF token

for dataset in \
    "MING-ZCH/StepCountQA-RL" \
    "MING-ZCH/StepCountQA-RL-Traj_11_50_NumericOnly" \
    "MING-ZCH/StepCountQA-RL-Traj_11_50_Combined" \
    "SI-Lab/StepCountQA-SFT" \
    "SI-Lab/StepCountQA-RL-SFT-Merged"; do
  repo_name=$(basename $dataset)
  huggingface-cli download $dataset \
      --repo-type dataset \
      --local-dir $NEW_HOME/work/StepcountModel/dataset/$repo_name
done
```

> 以 `MING-ZCH/` 或 `SI-Lab/` 为命名空间，用 `huggingface-cli repo list-datasets` 确认实际 repo id 后再下载。

#### 本地 🟡 资产

| 资产 | 说明 | 大小 |
|------|------|------|
| 模型权重 `model/StepCount-7B-SFT-*` | 训练后 ckpt，需通过 HF 中转或 rsync | 数十 GB |
| 模型权重 `model/stepcount_qwen2.5_*` | 同上 | 数十 GB |
| `dataset/stepcount_reasoning_dataset_merged_7500_plus_11_50.json` | 最常用训练 json | ~1.3 GB |
| `dataset/vlms_bias_data/` | bias 评测集 | 中 |
| `dataset/stepcount-500/` | 500 样本子集 | 小 |
| pixmo-count 评测图片 `PIXMO_IMAGES_DIR` | 原路径 `.../datasets/Datasets--allenai--pixmo-count/processed_data/images` | 大 |

```bash
# 评测所需的 pixmo-count 图片（从 HF 重新下载）
huggingface-cli download allenai/pixmo-count \
    --repo-type dataset \
    --local-dir $NEW_HOME/datasets/Datasets--allenai--pixmo-count

# 最新 reasoning json（按需，从 zch-dev rsync）
rsync -avz \
    /mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/stepcount_reasoning_dataset_merged_7500_plus_11_50.json \
    <new-cluster>:$NEW_HOME/work/StepcountModel/dataset/
```

#### EasyR1 训练产出的模型（用于 StepCount 评测）

StepcountModel 评测脚本 `eval_multi_task.sh` 的 `DEFAULT_MODEL_PATH` 指向 EasyR1 `save/` 下的 checkpoint。需在新集群完成 EasyR1 训练后，将新 ckpt 路径写入该变量。

---

## 7. 环境变量速查

在新集群 `~/.bashrc` 或 `~/.profile` 中追加：

```bash
# ── 通用 ──────────────────────────────────────────
export NEW_HOME=<新集群个人目录>          # 按实际修改
export HF_HOME=$NEW_HOME/cache/huggingface
export TRANSFORMERS_CACHE=$NEW_HOME/cache/huggingface/hub
export HF_TOKEN=<new-hf-token>           # huggingface-cli login 后可省

# ── 代理（若需要）────────────────────────────────
export http_proxy=http://<代理地址>
export https_proxy=http://<代理地址>
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy

# ── Weights & Biases ─────────────────────────────
export WANDB_API_KEY=<wandb-key>
export WANDB_PROJECT=StepCount    # 按项目设

# ── StepCount Masks（EasyR1 训练专用）────────────
export STEPCOUNT_MASKS_METADATA=$NEW_HOME/StepCount-RL_masks_output/masks_metadata.json
export STEPCOUNT_MASKS_DIR=$NEW_HOME/StepCount-RL_masks_output/masks

# ── GPU / CUDA ────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

---

## 8. 非代码文件清单（额外需要传输）

以下文件不在 GitHub 迁移分支中，需要手动从 zch-dev 传输或重新获取：

| 文件/目录 | 源集群路径 | 大小 | 优先级 |
|----------|-----------|------|--------|
| masks 文件 | `$OLD/StepCount-RL_masks_output/` | 数百 MB | 🔴 高（EasyR1 训练必须） |
| StepCount 模型权重 | `$OLD/work/StepcountModel/model/` | 数十 GB | 🟡 中 |
| MetaphorStar 模型 | `$OLD/work/MetaphorStar/MetaphorStar-{3B,7B}/` | 数十 GB | 🟡 中 |
| EasyR1 最终 ckpt | `$OLD/work/EasyR1-latest/save/<最新>` | 数十 GB | 🟡 中（训练后重新生成亦可） |
| pixmo-count 图片 | `$OLD/datasets/.../processed_data/images` | 大 | 🟡 中（可从 HF 重新下载） |
| MetaphorQA 数据 | `$OLD/work/MetaphorStar/MetaphorQA*/` | 中 | 🟡 中 |
| reasoning dataset json | `$OLD/work/StepcountModel/dataset/stepcount_reasoning_dataset_merged_7500_plus_11_50.json` | ~1.3 GB | 🟡 中 |

---

## 9. 安全提醒

> ⚠️ 以下敏感信息在脚本中**明文硬编码**，迁移后必须清理：
>
> - `WANDB_API_KEY` — 见 `examples/*.sh`，改为 `wandb login` 或 secrets 管理
> - `HF_TOKEN` — 见 `dataset/upload_*.py`，改为 `HF_TOKEN` 环境变量
> - GitHub PAT — 原 `~/.git-credentials` 中旧 token 请**立即在 GitHub Settings 撤销**并重新生成
