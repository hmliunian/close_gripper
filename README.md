# close_gripper

从左右从臂开度与同侧腕部图像生成独立夹爪训练目标，保留实际观测与原时间轴。算法以 [PLAN.md](PLAN.md) 为准，工程约束见 [BOUNDARIES.md](BOUNDARIES.md)，当前验证见 [VALIDATION.md](VALIDATION.md)。

已实现趋势检测、LeRobot v3 只读加载、Qwen 视觉适配、保守闭合延伸、两种释放候选比较、同步图片与曲线审阅页、独立目标 sidecar 和本地训练适配函数。模型或合成结果只生成候选；正常释放策略仍需审阅，不标为有效训练标签。

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

也可使用 `DASHSCOPE_API_KEY` 环境变量。默认固定 `qwen3.8-max-2026-09-02`、北京 compatible-mode endpoint；地域必须匹配密钥，可在配置中指定其他官方 endpoint。默认不重试，全部请求共用预算；模型原始输出、解析结果、token 用量与请求摘要独立保存。失败或超预算保持原值。

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
