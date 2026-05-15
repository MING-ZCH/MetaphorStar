# 跨集群迁移说明（zch-dev → 新集群）

本目录是 zch-dev 上 3 个项目向 GitHub 私有 repo 推送的 staging 区。
策略：**仅代码迁移**，不动主分支；所有改动推到新分支 `migration/zch-dev-20260514`。

## 项目

| 项目 | 远端 repo | 源目录 | 新分支 |
|------|----------|--------|--------|
| EasyR1-latest | `MING-ZCH/EasyR1` | `/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest` | `migration/zch-dev-20260514` |
| MetaphorStar | `MING-ZCH/MetaphorStar` | `/mnt/shared-storage-user/zhangchenhao/work/MetaphorStar` | `migration/zch-dev-20260514` |
| StepcountModel | `MING-ZCH/StepcountModel`（新建私有） | `/mnt/shared-storage-user/zhangchenhao/work/StepcountModel` | `migration/zch-dev-20260514` |

## 排除规则

参见 `exclude-common.txt` 与各 `exclude-<project>.txt`。
- `.git/`、`__pycache__/`、`*.pyc/.so/.parquet/.safetensors/.bin/.pt/.ckpt` 等大文件
- `logs/ wandb/ outputs/ checkpoints/ runs/ save/`
- 秘钥 `.env *.pem *.key id_rsa*`
- 各项目自有大数据子目录（详见 DATASETS_MISSING.md）

## 关键元数据

- ENVIRONMENT.md — OS / GPU / CUDA / Python / conda 快照
- env-base.yml / env-rebuttal.yml — conda 环境 export
- pip-base.txt — pip freeze
- DATASETS_MISSING.md — 各项目缺失数据列表
- PATH_REWRITES.md — 新集群需要修改的绝对路径列表
