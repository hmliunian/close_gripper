# close_gripper

从左右从臂开度与同侧腕部图像生成独立夹爪训练目标，保留实际观测与原时间轴。算法以 [PLAN.md](PLAN.md) 为准，工程约束见 [BOUNDARIES.md](BOUNDARIES.md)，当前验证见 [VALIDATION.md](VALIDATION.md)。

已实现趋势检测、LeRobot v3 只读加载、Qwen 视觉适配、闭合与张开的端点拟合、原速延伸对照、同步图片与曲线审阅页、独立目标 sidecar 和本地训练适配函数。模型或合成结果只生成候选；正常释放策略仍需审阅，不标为有效训练标签。

## 安装与离线检查

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[data,plots,test]'
.venv/bin/python -B scripts/check.py
```

默认检查不读取凭据、不访问实际数据集、不联网。数值核心只需标准库；数据、图片和 PNG 依赖按 extras 安装。缺少 matplotlib 时仍生成独立 HTML，但没有 PNG；统一检查缺少数据依赖时会报告跳过项，不能当作全检查通过。

## 先查看曲线变化

```sh
.venv/bin/python -m close_gripper demo --output runs/my_demo
```

打开 `runs/my_demo/review.html`，切换 episode，拖动时间滑块。`curves.png` 可直接查看或分享。示例涵盖左侧误开、右侧正常释放、双侧同时误开、空夹和未触发；全部标明合成来源，橙线是候选，绿色仅绘制有效目标。

本次已生成的可移植示例：[曲线图](examples/synthetic_curves.png)、[交互审阅页](examples/synthetic_review.html)。闭合与张开均保留原起止时刻，将闭合终点和张开起点映射到0；蓝色虚线保留旧原速闭合延伸对照。正常释放仍只生成候选。闭合默认 `closing_policy: "fixed_timing"`，配置 `"original_speed"` 可复现旧的延伸方案。

[闭合与张开局部对比](examples/closing_opening_fit.png)可直接查看两个过渡段的变化，数据明确为合成示例。

所有输出必须使用本工作区内**新的** `runs/<run_id>`。已有目录拒绝覆盖，失败后保留审计信息，换新 run 重试。

## 实际数据与 Qwen

```sh
# 默认只检测，不调用服务
.venv/bin/python -m close_gripper run \
  --dataset /absolute/path/to/lerobot_subset --episode 11 \
  --output runs/inspect_ep11

# 显式调用服务；凭据不会进入模型输入或报告
.venv/bin/python -m close_gripper run \
  --dataset /absolute/path/to/lerobot_subset --episode 11 \
  --config configs/episode11_qwen.json \
  --qwen --api-key-file qwen_api.txt --max-requests 3 \
  --output runs/qwen_ep11
```

实际请求默认读取本工作区根目录的 `qwen_api.txt`，也可显式指定 `--api-key-file`。密钥完整读取（支持含点号的格式），仅放入鉴权头，不进入请求正文、报告或Git。优先级为显式文件、显式配置的 `api_key_env`、默认文件；只有默认文件不存在时才回退 `DASHSCOPE_API_KEY` 环境变量，文件无效时不静默换key。默认固定 `qwen3.8-max-2026-09-02`、北京 compatible-mode endpoint；地域必须匹配密钥，可在配置中指定其他官方 endpoint。默认不重试，全部请求共用预算；模型原始输出、解析结果、token 用量与请求摘要独立保存。失败或超预算保持原值。

示例配置只选择 episode 11 中起始帧为97、187、188的三个候选。删除 `candidate_starts` 可按时间顺序检查，仍受请求上限控制。`context_before_s/context_after_s` 设置窗口，保留原始10 Hz帧。默认取同侧主视角，解码失败才尝试同腕另一眼；视觉遮挡需用 `camera_mode: "stereo"` 或人工另视角复核。超出图像上限直接拒绝，不隐式抽稀。

输出每个 episode 有 `episode.json`、`candidates.json`、`frames.json`、`model_responses.json`、`decisions.json`、`review_template.json`、`result.json`、`targets.json`。根目录 `summary.json` 表示流程完成，**不表示视觉与训练验收完成**。未触发、未检查、拒绝与候选状态分别记录。

## 人工复核与训练合同

复制本次 `review_template.json`，保留 `episode_sha256`，在 `decisions` 中记录明确的视觉边界与证据。人工确认的条目需设置 `review_source: "human"`、`confirmed: true`，并填写 `reviewer`、`reviewed_at`、`review_note`，然后：

```sh
.venv/bin/python -m close_gripper run \
  --dataset /absolute/path/to/lerobot_subset --episode 11 \
  --decisions /path/to/reviewed.json --output runs/reviewed_ep11
```

摘要不匹配时拒绝使用过期审阅；重新使用模型输出不能冒充人工确认。持物起点应标注为与视觉证据一致的数值接触/平台连接帧，含糊边界回退。类别与字段要求见 `contracts.py`，业务语义仍以 PLAN 为准。

`targets.json` 包含对齐的 timestamp、帧号及两个新字段：`annotation.gripper_target` 和 `annotation.gripper_target_valid`，列顺序为**左、右**。无效帧保存原夹爪值。模型候选单独在 `result.json.preview_target`，不得用它替代有效目标。

```python
from close_gripper.training import apply_gripper_targets
actions = apply_gripper_targets(original_actions, targets, valid)
```

该函数只替换20维动作的9/19维，不改输入或物理观测。调用者必须在同一时间网格上使用。它是本地接入合同；外部实际 loader、20 Hz 插值、训练与执行效果尚未验证。

## 审阅清单

```sh
.venv/bin/python -m close_gripper sample \
  --dataset /path/to/subset1 --dataset /path/to/subset2 \
  --limit 48 --output runs/sample_manifest
```

按子集和任务分层并确定性选取。没有原始 episode 对应关系的样本标为 `unassigned`，不能当作无泄漏留出集。左右、误开、释放、遮挡等语义覆盖需要实际审阅，工具不会从任务名推造参考标签。

当前不执行全量数据派生、不修改外部训练仓库、不发布 `-gripcmd` 版本。发布仍须遵循原需求的发布前确认。

## 冻结清单的真实批次检查

```sh
.venv/bin/python -m close_gripper batch \
  --manifest runs/20261103_real100_selection/manifest.json \
  --config configs/real100.json --workers 4 --qwen \
  --api-key-file qwen_api.txt --output runs/my_real100
```

清单需包含每条episode的缓存相对路径、完整输入摘要、准确候选ID和侧别。运行前核对缓存与候选，避免两侧同时张开时选错侧。每条episode只检查清单指定的一个代表事件，不将其余事件算作已检查。最大4个并发，每条最多1次真实请求、0重试；认证失败后阻止排队任务继续请求。去掉 `--qwen` 可只运行数值与报告流程。

根目录 `batch_review.html` 提供任务／侧别／状态筛选，逐条链接同步腕部画面和曲线；`case_results.json` 与 `summary.json` 分别保存逐条结果及汇总。模型解析成功、类别、数值回退、候选变化、有效训练目标变化和服务返回用量分别统计，不把调用成功率当作视觉准确率。失败用量未返回时保持未知。
