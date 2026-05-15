# 新集群路径改写指引

zch-dev（源）→ 新集群（目标）的路径基础前缀替换。

## 基础前缀

| 用途 | zch-dev | 新集群（建议） |
|------|---------|---------------|
| 个人工作区 | `/mnt/shared-storage-user/zhangchenhao/` | `<NEW_HOME>/` |
| 项目根 | `/mnt/shared-storage-user/zhangchenhao/work/<project>/` | `<NEW_HOME>/work/<project>/` |
| 模型 | `<project>/model/` | `<NEW_HOME>/models/<project>/` 或 HF 缓存 |
| 数据 | `<project>/dataset/` | `<NEW_HOME>/datasets/<project>/` 或 HF 缓存 |
| HF cache | `~/.cache/huggingface/` | 同左（注意磁盘空间） |
| Conda | `/root/miniconda3/` | 视新集群安装而定 |

## 推荐改写流程

1. 在新集群 clone 该分支 `migration/zch-dev-20260514`。
2. 设置环境变量：
   ```bash
   export OLD_PREFIX=/mnt/shared-storage-user/zhangchenhao
   export NEW_PREFIX=<新集群个人目录>
   ```
3. 用 `grep -rn "$OLD_PREFIX"` 列出所有写死路径：
   ```bash
   grep -rn "/mnt/shared-storage-user/zhangchenhao" \
     --include="*.py" --include="*.sh" --include="*.yaml" \
     --include="*.yml" --include="*.json" .
   ```
4. dry-run 后整体替换：
   ```bash
   grep -rl "$OLD_PREFIX" --include="*.py" --include="*.sh" \
     --include="*.yaml" --include="*.yml" . \
     | xargs sed -i "s|$OLD_PREFIX|$NEW_PREFIX|g"
   ```
5. 检查 conda env：参考 env-base.yml / env-rebuttal.yml 创建。
6. 检查 GPU/CUDA：参考 ENVIRONMENT.md 与 pip-base.txt，注意 torch / flash-attn / vllm 版本对 CUDA 的要求。

## 已知热点

- EasyR1: `examples/*.yaml`（data.train_files / data.val_files）、`scripts/*.sh`
- MetaphorStar: `train/*.sh`、`evaluation/*.py`
- StepcountModel: `dataset/*.py`（特别是 upload_* / convert_* 系列）、`eval_*.sh`
