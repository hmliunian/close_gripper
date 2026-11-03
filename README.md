# close_gripper

从从臂实测开度和同侧腕部图像生成完整的 **0～1夹爪目标开度曲线**：抓前稳定张开、闭合过渡、持物保持0、释放过渡。目标及最终交付以用户的 [Untitled.md](Untitled.md) 为准，实现方法见 [PLAN.md](PLAN.md)，工程边界见 [BOUNDARIES.md](BOUNDARIES.md)，实际证据与限制见 [VALIDATION.md](VALIDATION.md)。

当前主入口是 `clean`，扫描完整 episode 的所有有效侧与候选周期。模型证据通过校验后可生成有效自动标签；有效不代表人工核验。原始观测保持不变，新目标独立保存。已实现小样本派生和隔离副本中的实际 ORION loader 接入；全量三版数据处理、人工质量验收和正式发布尚未完成。

## 安装与离线检查

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[data,plots,test]'
.venv/bin/python -B scripts/check.py
```

统一检查不读取凭据、不访问实际数据集、不联网。数值核心只需标准库；数据、图片和 PNG 依赖按 extras 安装。缺少依赖时会报告跳过项，不能当作全检查通过。HTML 脚本有语法检查；尚无完整浏览器自动化。

所有运行输出必须使用本工作区内**新的** `runs/<run_id>`，已有目录拒绝覆盖。失败产物保留，重试使用新目录。

## 处理完整 episode

准备 JSON 清单，每项指定 LeRobot 子集和 episode：

```json
[
  {"dataset": "/absolute/path/to/lerobot_subset", "episode_index": 11}
]
```

可改用冻结快照：`episode_cache` 为清单目录内的相对 JSON 路径，配套 `episode_id` 和 `episode_sha256`。程序核验完整输入摘要，拒绝路径越界或过期快照；历史代表事件清单中的候选选择不限制新入口的全周期扫描。

```sh
# 检测与报告，不调用模型；未审阅范围保持原值
.venv/bin/python -B -m close_gripper clean --manifest /path/to/manifest.json --config configs/full_cycle.json --output runs/my_full_cycle_inspection

# 固定配置、4并发、所有 episode 共用请求预算
.venv/bin/python -B -m close_gripper clean --manifest /path/to/manifest.json --config configs/full_cycle.json --workers 4 --max-requests 50 --qwen --api-key-file qwen_api.txt --output runs/my_full_cycle_qwen
```

参数集中在 [configs/full_cycle.json](configs/full_cycle.json)。使用固定的 Qwen 模型、零重试、全局请求上限；认证失败后阻止排队请求。预算不足、证据不足、缺少同侧图像和边界矛盾分别记录。图像采样保留过渡附近原始帧，稳定区间默认每0.5秒采样；超过图片上限时明确拒绝，不隐式压缩证据。

默认凭据为工作区 `qwen_api.txt`，也可传入 `--api-key-file`。完整密钥仅进入鉴权头，不进入请求正文、报告或Git。显式文件优先，其次配置中的环境变量、默认文件；只有默认文件不存在时才回退 `DASHSCOPE_API_KEY`。非法凭据文件不静默切换来源。

根目录生成 `review.html`、`summary.json`、配置与实现快照；每条 episode 保存输入、周期、图片引用、模型原始响应、结构化决定、复核模板、结果、目标和单条曲线图。任务汇总记录修改 episode／帧数；请求用量与未知用量分别记录。

打开 `review.html`，切换 episode 并拖动时间滑块：灰线是实测开度，绿色是有效目标，橙线是候选；同时显示同侧腕部图像、逐帧阶段、来源和拒绝原因。覆盖统计表示候选及其上下文经过检查，**不表示每帧都有视觉真值**。根目录 `status=complete` 仅表示运行产物完成。

只调整数值实现、保持原候选和参数时，可用 `scripts/replay_clean_run.py --source-run /path/to/completed_run --manifest /path/to/frozen_manifest.json --output runs/my_numeric_replay` 复用证据。脚本先核验输入、原实现快照和图片摘要，保留原响应与失败记录；新调用数为0，复用的请求与用量单列。这是同一批证据的数值重放，不是独立评估。独立产物检查入口为 `scripts/verify_clean_run.py --run /path/to/run --manifest /path/to/frozen_manifest.json --output runs/my_run_audit`，均使用本地 Python 执行。

## 复核与独立目标

人工复核使用每条 `review_template.json`，保留 `episode_sha256`。填写证据与边界；人工确认需 `review_source: "human"`、`confirmed: true` 和 `reviewer`、`reviewed_at`、`review_note`。将封套按 `episode_id` 放入一个 JSON 对象，再使用 `clean --reviews /path/to/reviews.json` 写入新的运行。模型输出不能通过改一个确认位冒充人工来源。

`targets.json` 的列顺序为**左、右**，与 episode 时间戳和帧号一一对应：

| 字段 | 用途 |
| --- | --- |
| `annotation.gripper_target` | 0～1目标开度；无效帧保存实测原值 |
| `annotation.gripper_target_valid` | 该侧该帧是否可采用新目标 |
| `annotation.gripper_target_source` | 模型／人工等逐帧来源 |
| `annotation.gripper_phase` | 准备、闭合、持物、释放或保持原样 |

目标还绑定完整 episode 摘要、生成方法版本和运行来源。`result.json.preview_target` 是审阅候选，不能代替有效训练目标。来源和阶段通过完整 sidecar 随派生产物保存。

## 派生与实际训练读取

```sh
# 仅派生明确提供标签的 episode，源数据只读
.venv/bin/python -B -m close_gripper.export subset --dataset-root /absolute/path/to/lerobot_subset --targets /path/to/run/episode/targets.json --output runs/my_derived_subset

# 只读生成三版数据的任务分母与派生清单；不是全量处理或发布
.venv/bin/python -B -m close_gripper.export plan --release-root /absolute/path/to/release1 --release-root /absolute/path/to/release2 --release-root /absolute/path/to/release3 --output runs/my_release_plan
```

导出在 Parquet 中新增目标与有效性字段，保留原列、原值、帧与 episode 索引、图像时间关系。新目录包含元数据、训练配置、标签 sidecar、摘要及一致性验证；小样本目录显式标识为部分覆盖，不能当作完整 `-gripcmd` 发布版。

本地 `training.py` 提供20维动作9/19维替换及10→20 Hz标签插值合同。实际接入脚本复制指定 ORION loader 到新 run，生成可审阅补丁，并调用真实 `get_dataloader`／`make_dataloader` 对比 batch：

```sh
# 使用已装有 ORION 依赖的 Python；不修改外部仓库
/gpfs/jiuquyun/projects/youzi/orion-data-pipeline/.venv/bin/python scripts/verify_orion_loader.py --source /gpfs/jiuquyun/projects/youzi/orion-0 --derived-root /absolute/path/to/derived_subset --output runs/my_orion_verification
```

新配置通过 `dataset.gripper_target_source: annotation` 启用标签；默认仍读取观测。标签仅替换动作的两个夹爪通道，物理观测及其他动作通道保留；插值跨无效区间回退，时间戳不符合声明网格时拒绝。实际仓库版本、继承的已有修改、补丁及逐 batch 结果保存到运行目录。此验证不等于训练效果验证，也未把补丁安装到外部仓库。

## 历史入口与证据

`run` 是局部张开候选实验，`batch` 每条只检查一个指定代表事件，`demo` 展示旧方法的合成契约；这些入口和产物用于复现历史基线。完整曲线任务请使用 `clean`。

旧版100条／91任务检查及GUI单位修正的记录保留在 [VALIDATION.md](VALIDATION.md)。[合成曲线](examples/synthetic_curves.png)、[合成审阅页](examples/synthetic_review.html)、[旧100条统计](reports/20261103_real100_summary.json)都不代表新方法质量验收。实际图片、完整响应和运行目录默认不提交Git。

正式发布仍按任务书约定，在三版完整派生产物与验收结果准备好后确认。
