# 实现与验证记录

更新日期：2026-11-05。任务目标与验收依据见 [Untitled.md](Untitled.md)，当前方法见 [PLAN.md](PLAN.md)。本页先记录最新展示及当前配置的全量数值处理核验，再保留此前证据。软件检查、模型自动标签、主臂参考、真实 loader 读取和训练效果分别报告。

## 主视角20条、动作配置与独立GUI（2026-11-05）

按用户最新指令，选10个动作各2条独立原始录制，共20条episode、22个有效侧；直接用主视角取代腕部作本批判断，没有做视角对照。动作包括拿起刷子、杯子、魔方、海绵擦、芒果、插入缝隙刷、揭开锅盖、盖上锅盖、橱柜门打开和叠衣服。取样沿用这些任务已有冻结清单中的两条来源，没有按本轮是否抓住或配置是否启用筛选。准备输入为 `runs/20261105_head20_inputs_v1/`，原生审阅为 `runs/20261105_head20_reviews_v1/`。

三个子agent分别承担主视角语义查看、GUI和合同检查，另作独立数据审计；主agent完成输入、配置、整合与复看。1,612个时间帧完整查看，双侧合计2,258个侧帧：**held633、free677、uncertain948**。实际查看并绑定SHA的主视角图片2,357张；头部双目各自按视频元数据偏移及解码时间对齐。全部为Codex原生模型观察，人工确认false、外部Qwen请求0；这些计数不是人工真值或准确率。

主视角可见性按实际记录：刷子两例和杯子两例的接触区位于有效画面上沿或之外，未据手臂动作推断持物；魔方、芒果及部分海绵擦有遮挡。锅盖、柜门、缝隙刷和叠衣服可看到部分明确持物与释放过程。盖上锅盖25438末尾的下降半谷是放盖后空夹闭合，叠衣服也有同开度持物和空夹；曲线形状只作查看线索。

最终配置为 [task_gripper_profiles_head20.json](configs/task_gripper_profiles_head20.json)，构建产物在 `runs/20261105_head20_config_v3/`。10动作／11任务侧均保存观察区间及原因；**只有插入缝隙刷右侧满足现有逐值双来源自动匹配条件**，其实际匹配带为 **0.15107005488872527～0.17710481929779054**。另一些任务已观察到持物，但两个来源的开度带没有交集或存在同值空夹，不能记作没有抓住。配置和曲线都使用同批20条，不声称留出效果。

最终数值运行 `runs/20261105_head20_eval_v3/` 使用 `task-profile-rule-v3-hold-first-reviewed-v1`。原数值稳定窗与主视角held段相交后重新检查时长，保留原数值候选和被拒绝子段；固定时序拟合遇到未知边界保留原值。**2条、2段、48帧发生数值改变**：13838右侧0～26帧、1012右侧0～20帧。两个例子均起始已经持物，释放边界不确定，未拟合该过渡；因此不能声称这20条已完成完整曲线清洗。原观测和训练目标保持，训练有效位全false。

独立审计 [verification_final.json](runs/20261105_head20_audit_v1/verification_final.json) **29,673项检查通过、0失败**：精确发布来源映射及SHA、20个独立source、原Parquet状态／时间／索引、实际证据摘要、逐点双来源校准、图像与40个主视角视频引用及各目偏移一致；948个uncertain与677个free侧帧全部原值保留，48个改写帧全部held且有合格稳定窗支持。安装配置与config／eval／GUI配置字节一致。审计为数据及规则核验，不能代替人工语义参考。

新GUI在 **[8770主视角审阅](http://127.0.0.1:8770/review.html)**，目录 `runs/20261105_head20_gui_v2/`；原8766页继续可用。新页包含20条、40路主视角引用、逐手状态、凹谷／半段和语义区间跳转、同步曲线及配置下载。真实Chromium153最终 **14项通过**，20条视频均实际解码；JS、控制台、HTTP、网络失败和外部请求均0，见 [browser verification](runs/20261105_head20_browser_v1/attempt-003/verification.json)。涵盖2倍速播放／暂停切目、时间偏移、逐帧、双臂状态、Range206和配置原字节下载。

实际命令如下，准备、构建、渲染、审计及浏览器命令均退出0；服务持续运行。更多运行记录见 [commands.json](runs/20261105_head20_work_v1/commands.json)。

```sh
.venv/bin/python -B scripts/prepare_head_review.py --manifest runs/20261104_task_profile_verified_inputs_v1/manifest.json --output runs/20261105_head20_inputs_v1
.venv/bin/python -B scripts/run_head_review.py --inputs runs/20261105_head20_inputs_v1 --reviews runs/20261105_head20_reviews_v1 --config-output runs/20261105_head20_config_v3 --output runs/20261105_head20_eval_v3
.venv/bin/python -B scripts/render_video_review.py --source-run runs/20261105_head20_eval_v3 --output runs/20261105_head20_gui_v2
.venv/bin/python -B scripts/serve_review.py --run runs/20261105_head20_gui_v2 --port 8770
.venv/bin/python -B runs/20261105_head20_audit_v1/audit_final.py
node runs/20261105_head20_browser_v1/verify.cjs --gui runs/20261105_head20_gui_v2
.venv/bin/python -B scripts/check.py
```

统一离线检查最终 **399项通过、0跳过，退出0**，日志 [offline_final_v2.log](runs/20261105_head20_work_v1/offline_final_v2.log)。覆盖主视角身份、错目／错时、双侧与稠密语义合同、未知帧保护、短持物／缺帧／独立周期及默认规则兼容；首次统一检查的旧模拟生成结果缺少版本字段，补齐夹具并验证实际版本落盘后通过，初次日志保留。

历史尝试保留：config_v1因重复整段图片元数据导致校准冲突枚举过慢而中止（退出130，无完成产物）；v2生成的数值曲线把6个uncertain帧归零、10个uncertain帧实际改写，已由v3的逐段视觉约束纠正，[旧审计](runs/20261105_head20_audit_v1/verification.json)不作最终验收。浏览器attempt-001被沙箱阻止，获权限后attempt-002发现暂停换目会重置视频倍速，修复后attempt-003通过。旧产物均未覆盖。已完成本轮20条主视角审阅、对应配置和GUI；人工语义评估、更多来源覆盖、训练采用及效果和正式发布未在本轮完成，见 [交付汇总](reports/20261105_head20_summary.json)。

## GUI显示对应任务夹持区间（2026-11-05）

按用户要求，`close_gripper/report.py` 在曲线上方增加“当前任务夹持区间”，从已落盘的 `coverage.matching_bands` 读取实际范围，按有效左／右侧分别展示六位小数和每端外扩余量，悬停显示完整精度；多个区间保留为独立的“或”选项。任务、episode及四档外扩切换时更新，无搜索结果时随当前episode隐藏。

四档均从原210条GUI输入重新生成展示，产物为 `runs/20261105_profile_margin010_video_gui_v1/`、`margin020`、`margin030`、`margin040` 同名目录；对比入口为 `runs/20261105_profile_margin_comparison_gui_v1/`，已在原 **`http://127.0.0.1:8766/review.html`** 启动。逐项比较四档各210条完整页面记录及媒体／偏移清单，与此前对应页面完全一致，原曲线、数据、视频和既有实验均保留。数值算法执行0、模型调用0，默认配置仍为0.01。

实际执行 `.venv/bin/python -B scripts/check.py` **退出0，375项通过、0跳过**；日志见 `runs/20261105_task_range_gui_work_v1/offline_check.log`。`.venv/bin/python -B runs/20261105_task_range_gui_work_v1/refresh_gui.py` **退出0**，四次渲染和对比页生成的完整命令及保留性核验见[记录](runs/20261105_task_range_gui_work_v1/preservation.json)。服务入口见 README。

真实Chromium在预览与最终8766地址均 **21项通过**，最终命令为 `node runs/20261105_task_range_gui_work_v1/verify_comparison.cjs --url http://127.0.0.1:8766/review.html`，退出0。检查四档曲线记录、9596任务区间与外扩联动、关微波炉的多个区间、叠衣服左右侧、空搜索恢复，以及视频解码／播放、筛选保留和Range 206；JavaScript错误、失败请求及外部请求均0。见[浏览器核验](runs/20261105_task_range_gui_work_v1/browser-2/verification.json)、[9596截图](runs/20261105_task_range_gui_work_v1/browser-2/margin030_9596_frame25.png)和[交付汇总](reports/20261105_task_range_gui_summary.json)。本轮仅完成显示改动，未新增语义或训练效果结论。

## 外扩0.02／0.03／0.04：全量参数对比与原GUI（当前）

按用户要求分别测试每端外扩0.02、0.03、0.04，保留已完成的0.01作为基线。三档均从 `runs/20261104_profile_merged_catalog_v1/calibration_catalog.json` 独立派生，原配置SHA256为 `d8d684cafcf12f418d6f73d8af21594feb9f5b2c5530ffadebaf62f3a291d696`，没有累计加宽。稳定时长0.6秒、波动阈值0.02、空夹／冲突阻断及任务选择保持；业务包实现摘要均为 `6f8d92a49a76d8b3701e2a16bd5ac6306682e8cb3372c4cf721c895508c878c7`。已安装默认配置仍为0.01，字节未改。

每档均使用原三版7子集、1,961分片、125,236条episode、15,041,374帧；105个启用任务共102,577条执行规则，45个未启用任务的22,659条保持原值。各档源身份、摘要和episode选择相同，未按开度命中筛选输入。命中指episode至少有一段合格夹持区间，分母为102,577，不代表整条episode都已标注。

| 每端外扩 | 命中episode | 命中比例 | 比前一档新增 | 数值改变episode | 数值改变帧 | 应用夹持区间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01（基线） | 42,027 | 40.97% | — | 41,505 | 1,170,132 | 45,369 |
| 0.02 | 47,154 | 45.97% | 5,127 | 46,573 | 1,338,559 | 51,378 |
| 0.03 | 49,538 | 48.29% | 2,384 | 48,957 | 1,430,206 | 54,609 |
| 0.04 | 51,061 | 49.78% | 1,523 | 50,480 | 1,491,824 | 56,864 |

逐episode身份集合比较确认各档命中集合包含前一档，没有丢失原命中。0.01有522条应用后数值与原值相同，其余三档各581条。扩大仍增加覆盖，但相邻档新增数递减。汇总及路径见[四档报告](reports/20261104_margin_sweep_summary.json)、[汇总CSV](reports/20261104_margin_sweep_summary.csv)和[150任务CSV](reports/20261104_margin_sweep_tasks.csv)。

三档各自完成9分片／515条小批次、268条逐episode参考运行及全量生成。小批次与全量的独立核验全部 **PASS、0失败**；每档全量核验覆盖全部1,961分片／125,236条，检查原列、索引、时间、未选任务、训练字段、新范围、全曲线及合格夹持段归零。全量落盘新增六列与独立逐episode运行在210条GUI样本上精确一致。核验分别见 [0.02](runs/20261104_profile_margin020_full_audit_v1/verification.json)、[0.03](runs/20261104_profile_margin030_full_audit_v1/verification.json)、[0.04](runs/20261104_profile_margin040_full_audit_v1/verification.json)。

实际入口如下；逐步骤完整命令、退出码和耗时见 `runs/20261104_profile_margin_study_v1/progress.jsonl`，总日志为同目录 `study.log`，冻结输入为 `frozen_inputs.json`。服务持续运行，其余列出命令退出0；运行脚本拒绝覆盖已有实验，复现须使用新运行目录。

```sh
.venv/bin/python -B runs/20261104_profile_margin_study_v1/run_study.py
.venv/bin/python -B runs/20261104_profile_margin_study_v1/build_comparison.py
.venv/bin/python -B runs/20261104_profile_margin_study_v1/build_comparison.py --output runs/20261104_profile_margin_comparison_gui_v2 --skip-summary-export
.venv/bin/python -B scripts/serve_review.py --run runs/20261104_profile_margin_comparison_gui_v2 --port 8766
node runs/20261104_profile_margin_study_v1/verify_comparison.cjs --url http://127.0.0.1:8766/review.html
```

原地址 **`http://127.0.0.1:8766/review.html`** 现在提供四档切换，默认0.01；四档均保留原105任务／210条固定样本及424个同侧视频引用，相对原观测改变的展示样本分别为155、167、172、175条。切换参数保留筛选、搜索、episode、帧位置、倍速和相机。9596第21–29帧四档均归零，第30–32帧仍不满足新的完整稳定窗；扩大范围不改变该例结论。

最终8766地址真实Chromium核验 **14项通过**，逐项检查全部840条参数版本记录的样本身份、原输入、时间、视频、夹持归零及训练无效位，并抽查9596和双臂例的实际视频解码、连续播放、同侧双目、曲线同步和Range 206。JavaScript错误、失败请求、外部请求均0；见[浏览器结果](runs/20261104_profile_margin_study_v1/browser-4/verification.json)和[0.03档9596截图](runs/20261104_profile_margin_study_v1/browser-4/margin030_9596_frame25.png)。首次浏览器检查发现切换参数后重建播放器会重置倍速，已修复并写入新的GUI目录v2；第二次检查修正了连续视频时钟与离散帧时刻混用的检查逻辑，精确拖帧检查仍保留。预览及最终地址均通过，失败记录与v1产物保留。

本轮未修改业务包或正式配置，下节基线的375项离线测试不计为本轮重跑；本轮新增证据为三档真实生成与独立审计、对比页面浏览器核验、文档链接／语法检查。完成的是参数对比和GUI交付；数值命中率不是语义准确率，尚未补充新增区域的逐episode视觉真值、训练采用及效果。输出仍为evaluation数值曲线，未提升训练有效位、未新增视觉判断或正式发布。

## 夹持区间合并并外扩0.01：全量重跑与GUI（基线）

用户批准合并并适当扩大夹持区间。配置更新为 `task-profiles-20261104-merged-margin010-v4`，SHA256 `b9f412cce8c1c5d5edaaff5399bd3d53ac5390aa31763e077245be283a712d26`。279条原校准带及全部语义证据保留，派生138条实际匹配带；稳定0.6秒、波动0.02、105任务选择和45任务跳过均保持。范围推导规则见 PLAN，逐任务新旧范围见[配置对照](reports/20261104_merged_profile_bands.csv)。扩展部分明确标记为泛化，不冒充新增视觉证据。

全量运行 `runs/20261104_profile_merged_full_v1/` 使用与上一版相同的三版7子集、1,961分片、125,236条episode和15,041,374帧。新冻结清单仅变更配置绑定与生成时间，全部源身份、摘要、任务和episode选择逐项一致。清单 SHA256 为 `1330dbe25f3f7cd32cc2c5e60e073c487569e8419b37ac690a8bf14fbdbf326a`，业务包摘要为 `6f8d92a49a76d8b3701e2a16bd5ac6306682e8cb3372c4cf721c895508c878c7`。

| 发布版 | 执行规则 episode | 上一版数值改变 | 本次数值改变 | 本次改变帧数 |
| --- | ---: | ---: | ---: | ---: |
| priority | 14,528 | 3,890 | 11,262 | 278,484 |
| remaining | 71,084 | 7,567 | 22,751 | 672,011 |
| drawer | 16,965 | 1,541 | 7,492 | 219,637 |
| 合计 | 102,577 | 12,998 | 41,505 | 1,170,132 |

**42,027条episode命中规则**（启用任务数据的40.97%，全部原始episode的33.56%），其中41,505条有数值改动，522条应用后仍与原值相同。共应用45,369个合格夹持区间。剩余60,550条启用任务episode未应用：14,378条未进入匹配带、21,466条带内连续稳定时间不足、24,705条含需区间视觉判断的原因、1条来源对齐疑点。上述原因按episode互斥归类，冲突视觉原因优先；没有把带内短段等同于原曲线完全没有稳定段。另22,659条属于45个未启用任务，全部原值保留。[覆盖统计](runs/20261104_profile_band_expansion_work_v1/coverage_reasons.json)保留完整分母与任务分布。

离线入口 `.venv/bin/python -B scripts/check.py` **退出0、375项通过、0跳过**，日志为 `runs/20261104_profile_band_expansion_work_v1/offline_final.log`。新增检查覆盖合并空隙、外扩边缘、空夹／冲突阻断、禁用侧、9596形态、原证据不变、禁止累计扩展，以及批处理投影和独立核验实际消费新范围。首次检查中新增全量审计夹具仍是legacy格式，补齐合成all-mode证据合同后通过；没有放宽真实配置校验。新匹配范围由独立实现复算，原279条校准带仍按原逐值双来源证据核验。

真实小批次9分片／515条及全量1,961分片／125,236条的独立核验均 **PASS、0失败**，包括原列、索引、时间、未选任务、原训练字段、新范围、全曲线及全部合格夹持段归零。生成、核验和展示的实际命令如下；日志在 `runs/20261104_profile_band_expansion_work_v1/`，服务命令持续运行，其余命令退出0：

```sh
.venv/bin/python -B scripts/expand_task_profile_bands.py --catalog configs/task_gripper_profiles.json --margin 0.01 --version task-profiles-20261104-merged-margin010-v4 --output runs/20261104_profile_merged_catalog_v1
.venv/bin/python -B scripts/process_profile_dataset.py --manifest runs/20261104_profile_merged_pilot_scope_v1/manifest.json --catalog runs/20261104_profile_merged_catalog_v1/task_profiles.json --workers 4 --output runs/20261104_profile_merged_pilot_v1
.venv/bin/python -B scripts/verify_profile_dataset.py --run runs/20261104_profile_merged_pilot_v1 --output runs/20261104_profile_merged_pilot_audit_v1 --workers 4
.venv/bin/python -B -m close_gripper.cli profile-clean --manifest runs/20261104_task_profile_verified_inputs_v1/manifest.json --profiles runs/20261104_profile_merged_catalog_v1/task_profiles.json --workers 4 --output runs/20261104_profile_merged_examples_v1
.venv/bin/python -B scripts/process_profile_dataset.py --manifest runs/20261104_profile_merged_scope_v1/manifest.json --catalog configs/task_gripper_profiles.json --workers 4 --output runs/20261104_profile_merged_full_v1
.venv/bin/python -B scripts/verify_profile_dataset.py --run runs/20261104_profile_merged_full_v1 --output runs/20261104_profile_merged_full_audit_v1 --workers 4
.venv/bin/python -B runs/20261104_profile_105_gui_extract_tool_v2/extract_from_bulk.py --bulk-run runs/20261104_profile_merged_full_v1 --reference-run runs/20261104_profile_merged_examples_v1 --output runs/20261104_profile_merged_gui_source_v1
.venv/bin/python -B scripts/render_video_review.py --source-run runs/20261104_profile_merged_gui_source_v1 --output runs/20261104_profile_merged_video_gui_v1
.venv/bin/python -B scripts/serve_review.py --run runs/20261104_profile_merged_video_gui_v1 --port 8766
node runs/20261104_profile_merged_browser_v1/verify_browser.cjs --url http://127.0.0.1:8766/review.html
```

第一条命令运行时配置仍是原校准版；原输入字节保存在 `runs/20261104_profile_merged_catalog_v1/calibration_catalog.json`，复现时应使用该原版本和新的输出目录，不能给已扩展版本累计加宽。通过小批次后，新配置已安装到 `configs/task_gripper_profiles.json`。

原GUI地址 **`http://127.0.0.1:8766/review.html`** 已切到 `runs/20261104_profile_merged_video_gui_v1/`。固定105任务／210条样本和424个同侧视频引用保持；相对原观测有变化的样本从55条增加到155条。全量落盘新增六列与本轮逐episode运行精确一致；新旧GUI间119条曲线不同，共2,565帧，原输入、视频及时间均相同。页面增加实际夹持开度范围显示。

9596（拿起刷子）现在使用 **0.01577398–0.18106512** 匹配范围，第21–29帧稳定段归零，闭合拟合从第10帧连接至第21帧；第25帧实测0.12375191、目标0。第30–32帧未满足新的完整稳定窗，仍保留原值，没有声称整个尾段均已归零。3093第18帧仍为0。真实Chromium在最终8766地址 **20项通过**，含9596数值与范围展示、3093回归、视频解码／同步、筛选和Range 206，JavaScript错误、失败请求与外部请求均0。结果见 [浏览器核验](runs/20261104_profile_merged_browser_v1/attempt-002/browser_verification.json)及[9596截图](runs/20261104_profile_merged_browser_v1/attempt-002/brush_9596_frame25_expanded_hold_zero.png)。

本次完成配置外扩、全量数值重跑、独立核验和原GUI更新；[汇总](reports/20261104_merged_profile_summary.json)、[任务对比](reports/20261104_merged_profile_task_results.csv)、[全量核验](runs/20261104_profile_merged_full_audit_v1/verification.json)可复查。数值覆盖提升不等于语义准确率提升，训练有效位仍未启用；原数据、历史运行和证据保留。当前产物仍为evaluation数值staging与只读源视频展示，尚未新增语义真值、独立媒体交付、训练效果或正式发布。

## 稳定夹持优先：全量重跑与原GUI更新（上一版）

按用户确认的稳定夹持优先方案，生成方法改为 `task-profile-rule-v3-hold-first`，业务细则见 PLAN。仍使用原105任务配置、全部1,961个源分片及原GUI固定样本，没有重新采样或合并配置区间。配置 SHA256 为 `d8d684cafcf12f418d6f73d8af21594feb9f5b2c5530ffadebaf62f3a291d696`，完整范围摘要为 `54e4f613b78499fb222cba99e516b7bfd6420e61580ca5ada096bc34d93bec1f`，本次业务包摘要为 `709d7563e83719ebb25e000aef271308706947ad36c2cb5f87d1294c77f6978f`。全量产物为 `runs/20261104_profile_hold_first_full_v1/`；以前的运行均保留。

| 发布版 | 原始 episode | 执行规则 | 跳过未启用任务 | 数值改变 episode | 改变帧数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| priority | 20,586 | 14,528 | 6,058 | 3,890 | 89,708 |
| remaining | 87,685 | 71,084 | 16,601 | 7,567 | 186,302 |
| drawer | 16,965 | 16,965 | 0 | 1,541 | 39,635 |
| 合计 | 125,236 | 102,577 | 22,659 | 12,998 | 315,645 |

105任务执行规则，45任务在派生表中原值透传，候选及修改数均0。14,307个合格夹持区间全部应用，独立核验逐一确认其完整稳定段为0；旧方法为6,172个应用区间、5,681条变化episode／152,692变化帧。本次358,331个短候选及219,518个仍需区间视觉判断的候选保留原值，未算作成功处理。

统一离线入口 `.venv/bin/python -B scripts/check.py` **退出0，366项通过、0跳过**，日志为 `runs/20261104_profile_hold_first_work_v1/offline_final.log`。新增契约覆盖3093形态、小幅张开、缺少过渡、重叠过渡与合格稳定段不可被覆盖；独立审计另有篡改目标且更新文件摘要仍须失败的检查。真实小批次9分片／515条episode和全量1,961分片／125,236条episode的独立数值核验均 **PASS、0失败**，确认15,041,374帧的原列、索引、时间戳、无效侧及未选任务保持。

实际执行的生成和核验命令如下，均退出0；完整日志在 `runs/20261104_profile_hold_first_work_v1/`：

```sh
.venv/bin/python -B scripts/process_profile_dataset.py --manifest runs/20261104_profile_105_pilot_scope_v1/manifest.json --catalog configs/task_gripper_profiles.json --workers 4 --output runs/20261104_profile_hold_first_pilot_v1
.venv/bin/python -B scripts/verify_profile_dataset.py --run runs/20261104_profile_hold_first_pilot_v1 --output runs/20261104_profile_hold_first_pilot_audit_v1 --workers 4
.venv/bin/python -B -m close_gripper.cli profile-clean --manifest runs/20261104_task_profile_verified_inputs_v1/manifest.json --profiles configs/task_gripper_profiles.json --workers 4 --output runs/20261104_profile_hold_first_examples_v1
.venv/bin/python -B scripts/process_profile_dataset.py --manifest runs/20261104_profile_105_scope_v1/manifest.json --catalog configs/task_gripper_profiles.json --workers 4 --output runs/20261104_profile_hold_first_full_v1
.venv/bin/python -B scripts/verify_profile_dataset.py --run runs/20261104_profile_hold_first_full_v1 --output runs/20261104_profile_hold_first_full_audit_v1 --workers 4
.venv/bin/python -B runs/20261104_profile_105_gui_extract_tool_v2/extract_from_bulk.py --bulk-run runs/20261104_profile_hold_first_full_v1 --reference-run runs/20261104_profile_hold_first_examples_v1 --output runs/20261104_profile_hold_first_gui_source_v1
.venv/bin/python -B scripts/render_video_review.py --source-run runs/20261104_profile_hold_first_gui_source_v1 --output runs/20261104_profile_hold_first_video_gui_v1
.venv/bin/python -B scripts/serve_review.py --run runs/20261104_profile_hold_first_video_gui_v1 --port 8766
node runs/20261104_profile_hold_first_browser_v1/verify_browser.cjs --url http://127.0.0.1:8766/review.html
```

服务命令持续运行，其余命令退出0。GUI仍在 **`http://127.0.0.1:8766/review.html`**，保留原105任务／210条、424个同侧视频引用。全量落盘的六个新增列与本次逐episode独立运行完全一致。对照旧GUI，210条的身份、状态、时间、视频与候选清单完全一致，其中25条的曲线发生变化，共548帧；当前55条相对实测值有数值改动。对照记录见 [gui_comparison.json](runs/20261104_profile_hold_first_work_v1/gui_comparison.json)。

用户指出的 priority / baseline-v4-pushcart-right-v1 / **3093**：第18帧（1.80秒）右侧实测 `0.1556805223`，本次目标为 **0**；第0–41帧稳定段全部为0，释放拟合连接第41–46帧，保留第46帧实际张开端点。真实Chromium检查在原8766地址 **19项通过**，包括此数值、页面展示、任务筛选、时间拖动、连续视频解码、同侧双目、双臂同步和Range 206；JavaScript错误、失败请求及外部请求均0。证据为 [browser_verification.json](runs/20261104_profile_hold_first_browser_v1/attempt-003/browser_verification.json) 和 [3093截图](runs/20261104_profile_hold_first_browser_v1/attempt-003/spoon_3093_frame18_hold_zero.png)。首次启动被OS沙箱限制，升级执行后临时预览端口与最终8766端口均通过，失败记录保留。

本次完成全量数值重跑、独立核验和原GUI更新；结果汇总见[重跑汇总](reports/20261104_hold_first_summary.json)、[逐任务对比](reports/20261104_hold_first_task_results.csv)和[全量核验](runs/20261104_profile_hold_first_full_audit_v1/verification.json)。输出仍为evaluation曲线，未新增视觉判断、训练有效标签或正式发布。当前全量目录是数值staging，GUI只读原视频；此前独立媒体副本及ORION loader结果属于旧运行，不能记为新版训练或完整媒体交付。逐episode语义质量、训练采用及训练效果仍待相应验收。

## 旧方法v2的三版全量处理与再次核验

此前用户要求用当前配置跑全部数据集，没有对应可用动作的跳过。检查发现 `runs/20261104_profile_105_full_eval_v1/` 已有完整全量结果，其配置 SHA256 与 `configs/task_gripper_profiles.json` 完全相同：`d8d684cafcf12f418d6f73d8af21594feb9f5b2c5530ffadebaf62f3a291d696`；业务包实现摘要与当时的v2实现一致。该次复用完成产物，并重新独立核验全部源数据、候选、曲线和任务统计，没有重跑生成。以下数字属于旧方法，当前v3重跑见上节；区间配置仍未合并。

| 发布版 | 原始 episode | 执行规则 | 跳过未启用任务 | 数值改变 episode | 改变帧数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| priority | 20,586 | 14,528 | 6,058 | 2,707 | 64,877 |
| remaining | 87,685 | 71,084 | 16,601 | 2,425 | 70,796 |
| drawer | 16,965 | 16,965 | 0 | 549 | 17,019 |
| 合计 | 125,236 | 102,577 | 22,659 | 5,681 | 152,692 |

105 个启用任务执行数值规则，45 个未启用任务的候选、规则应用和修改数均为0。三版7子集、1,961个Parquet、15,041,374帧全部保留原字段、行序和索引；跳过任务在派生结构中保留原值。实际应用6,172个区间，数值改变数量不代表抓持准确率；没有命中或被边界条件拒绝的范围不算清洗成功。

本次实际执行以下命令，**退出0、PASS、0失败**，完整核验1,961个分片和125,236条episode：

```sh
.venv/bin/python -B scripts/verify_profile_dataset.py --run runs/20261104_profile_105_full_eval_v1 --output runs/20261104_profile_105_confirmation_audit_v1 --workers 4
```

完整数值复核见 [verification.json](runs/20261104_profile_105_confirmation_audit_v1/verification.json)，按任务的执行／跳过／修改统计见 [task_results.csv](runs/20261104_profile_105_confirmation_audit_v1/task_results.csv)，本次命令、摘要与产物路径见 [completion_summary.json](runs/20261104_profile_105_confirmation_audit_v1/completion_summary.json)。既有完整派生结构为 `runs/20261104_profile_105_delivery_v1/staging/` 下的三个 `-gripcmd` 发布版；本次另核对全部2,003份数值与支持文件的源／副本SHA一致。视频复制记录来自既有交付，本次未重新核验视频内容。

新增曲线保存在 evaluation 字段，原有数据及训练目标保持原合同；本次未提升训练有效位、未新增视觉判断，亦未正式发布。完成的是当前配置全量数值处理及其核验，逐episode语义质量、训练采用与训练效果仍需另行验收。

## 稳定平台语义复核与逐值二源配置（当前）

按用户要求检查稳定时间内是否实际夹持，4个agent使用Codex原生视觉复核三个发布版、150任务／151有效侧别。完整数值扫描沿用下节所述125,236条episode、772,042个至少0.6秒的稳定窗口；视觉检查覆盖全部 **2,377个数值模式的代表**，不把稳定、lower或接近0直接当作夹持。逐例查看平台首、中、末及上下文，保留持物、未持物、不确定和混合子段。

最终 `runs/20261104_task_profile_second_sources_v2/normalization_final_v3/` 保存 **2,685条审阅记录／1,439个独立原始录制**，包括305条二源补证（该阶段123例补另一目），以及两条刷子诊断的边界补证。平台状态为held671、free1298、uncertain684、mixed32；不确定平台仍可能含已确认子段，不能把这些数直接当episode类别或模型准确率。实际帧和原图SHA、原始判断、同帧重看修订及双目证据分别保留；最终同帧语义冲突与合同错误均0，12,519个证据文件SHA核验通过。全部是模型证据，人工确认均false。

原发布版全量模式外还有18,396个稳定窗口（2.3828%），归入4,722个数值尾桶；当前仅8个尾桶有代表复核、**4,714个仍未审**。代表复核不验证该模式全部成员窗口。叠衣服一条录制的信号与图像语义存在疑点：逐值核对原MCAP及实际图像时间后仍无法消歧，26条相关审阅记录排除数值校准；来源及精确episode身份进入运行回退，不能据此排除其他同编号episode。

当前任务配置版本为 `task-profiles-20261104-all-modes-pointwise-two-source-v3`，SHA256 `d8d684cafcf12f418d6f73d8af21594feb9f5b2c5530ffadebaf62f3a291d696`。**105任务／106任务侧有279条可用带**；每个可用数值都由至少两个独立来源的实际持物子段支持。另3侧为已审样本不支持、42侧需继续视觉判断，合计45任务未自动启用；可用任务也只在其已校准区间匹配。3个二源检索请求没有第二来源，明确记为不足，没有复用原阳性源凑数。

配置构建纠正了两处实质问题：局部held/free重叠只切除实际交集，保留其余有证据部分；连通带内只有一源覆盖的边缘不再借用整带来源数。缝隙刷实际持物约0.164，旧配置只覆盖约0.06；新版保留两种有支持的区间，并依据同源可见释放建立局部小幅释放规则，全局噪声、0.6秒确认和固定时序公式保持不变。

独立几何审计 `runs/20261104_task_profile_catalog_audit_final_v3/report.json` **PASS，24,738项、0失败**：从保存的H/F帧、实际trace与子段重新计算持物带、逐值二源及真实交集，不导入builder；核对排除源未贡献，刷子约0.164可用、已知自由0.20269不可当持物，并验证13838整段f0..29被同一带包含。软件统一检查 `.venv/bin/python -B scripts/check.py` **328项通过、0跳过，退出0**，日志 `runs/20261104_task_profile_second_sources_v2/builder_pointwise_check_v1.log`；配置CLI校验退出0。

实际规则运行 `runs/20261104_task_profile_verified_eval_v2/` **退出0**：保留原266条比较样本的缓存字节、身份与内容，另给新增可用任务“挂毛巾”补2个未参加V3校准的独立来源，共268条／134任务／14,803帧／270有效episode侧，所有105个可用任务均有两例。新增来源使用固定种子20261104，排除校准来源、同源优先最长裁剪，不按命中带选样。当前比较样本中73条已参加新版视觉校准（25条来源支持可用持物带），另195条未参加；其中原266的未校准部分为193条，不能再把全部266或268称为独立留出。

本轮完整枚举251候选／122合格窗口，41区间应用、210候选拒绝。40条有规则应用，其中34条产生801帧实际变化，6条原数值无需变化；其余228条没有应用。拒绝分别为时长不足131、同值语义冲突需逐段视觉54、无对齐闭合边界6、对齐后时长不足1、释放边界不足18。原始观测、时间戳、行数、无效侧及训练目标全部保持，训练有效位全false；规则目标仍是独立evaluation字段，未生成正式训练标签。

缝隙刷两条均应用、共改变58帧；挂毛巾两条均应用但原值已满足目标，变化0帧。第一次新版266试跑 `verified_eval_v1` 曾因13838开头几个点略超二源带而拒绝。保留该结果后，补齐此前已查看诊断录制的首中末双目证据形成V3；没有放宽噪声、来源要求或拟合边界，其余149任务配置与V2逐对象相等。

独立落盘审计 `runs/20261104_task_profile_verified_eval_audit_v2/verification.json` **PASS，172,348项、0失败**：穷举原曲线窗口，核对全部候选／事件、0.6秒回填、固定时序公式、逐值二源、冲突拒绝、mask并集与原字段保留；211个源Parquet的SHA、大小与mtime均符合记录。二源材料另核验16,935项身份／图片／时间关联及241个涉及源Parquet与原census摘要一致，未写源数据。

实际入口：

```sh
.venv/bin/python -B -m close_gripper.task_profiles validate configs/task_gripper_profiles.json
.venv/bin/python -B -m close_gripper profile-clean --manifest runs/20261104_task_profile_verified_inputs_v1/manifest.json --profiles configs/task_gripper_profiles.json --workers 4 --output runs/20261104_task_profile_verified_eval_v2
.venv/bin/python -B runs/20261104_all_mode_evaluation_audit_tools_v2/audit_output.py --run runs/20261104_task_profile_verified_eval_v2 --inputs runs/20261104_task_profile_verified_inputs_v1 --catalog configs/task_gripper_profiles.json --output runs/20261104_task_profile_verified_eval_audit_v2
```

这些输出目录已存在，复现应换新目录。执行记录是本次真实结果，不表示原数据全部稳定窗已完成语义审查，也不表示人审准确率、实际训练效果或三版全量派生已完成。按任务的区间与拒绝状态、逐条例子及GUI核验见本节交付表；旧结果与失败记录继续保留。

| 交付 | 当前产物与核验 |
| --- | --- |
| 任务配置与状态 | [150任务／151侧的区间和原因表](reports/20261104_verified_task_profiles.csv)，包括105个自动任务及45个未启用任务 |
| 实际逐条例子 | [268条应用／未应用与来源重叠表](reports/20261104_verified_profile_examples.csv)，所有105个自动任务各有两例 |
| 机器可读汇总 | [配置、覆盖、运行与检查摘要](reports/20261104_verified_task_profiles_summary.json) |
| 稳定平台证据GUI | [8767审阅页](http://127.0.0.1:8767/review.html)，目录 `runs/20261104_task_profile_verified_evidence_gui_v2/`；2685条记录、150任务、11,842个去重原图字节副本；图片摘要核对通过，新增外部模型请求0 |
| 视频与曲线GUI | [8766审阅页](http://127.0.0.1:8766/review.html)，目录 `runs/20261104_task_profile_verified_video_gui_v2/`；268例、540个同侧视频引用，不复制源视频 |

两套页面均完成真实Chromium检查：模式证据页 `runs/20261104_all_mode_browser_tools_v1/attempt-002/verification.json` **8项通过**；连续视频页 `runs/20261104_verified_video_browser_v1/attempt-001/browser_verification.json` **18项通过**。两者JavaScript错误、失败请求和外部请求均0。检查覆盖筛选、搜索、曲线／状态来源、视频真实解码、非零偏移、双臂同步、同侧双目切换、倍速及HTTP 206；另外核对缝隙刷两例的原始非零持物段与归零曲线、挂毛巾两例的无数值变化应用。实际目视复核两张刷子页面截图，未用截图检查代替逐episode语义真值。视频交互抽查范围与原始失败尝试分别保留，不声称540段均完整播放。

当前服务入口如下；远程使用需转发服务器8766及8767端口。已运行的服务应继续复用，停止后才重新启动：

```sh
.venv/bin/python -B scripts/serve_review.py --run runs/20261104_task_profile_verified_video_gui_v2 --port 8766
.venv/bin/python -B scripts/serve_review.py --run runs/20261104_task_profile_verified_evidence_gui_v2 --port 8767
```

二源材料首次准备因run内 `select.py` 遮蔽标准库而在导入阶段失败；改名后在新的v2目录按**完全相同冻结选样**恢复，305例材料全部成功，失败未记成视觉复核。一次联系图帧号疑虑经原样重绘及SHA核验解除，没有改源视频对齐；实际需要修正的语义均另存并重新派生子段，未覆盖原判断。

## 首版：133个任务各两条的规则曲线试跑

2026-11-04 用户授权开始按任务开度配置运行，并要求每个可用任务两条与视频GUI。新增 `profile-clean` 生成真实规则曲线；来源为 `task_profile_rule`，不调用模型，也不伪造逐episode视觉决定。结果中的 `evaluation.gripper_target` 保存新曲线；训练目标保持原观测、有效位为false，供本轮审阅后继续验证。

冻结输入为 `runs/20261104_task_profiles_two_per_task_inputs_v1/manifest.json`：133任务各两个不同来源，共266条episode、266个独立录制、14,729帧、268个有效episode侧。排除全部301个开度校准来源，交集0；随机种子20261104，不按是否命中夹持带筛选。相同来源在不同发布版中的裁剪先取最长，再对来源随机抽样。发布版分布为priority202、remaining57、drawer7，未发生加载失败或缺同侧视频。

实际运行 `runs/20261104_task_profiles_two_per_task_eval_v1/` **退出0**：完整枚举389候选、280个合格稳定窗口；215个区间应用、174个候选明确拒绝。186条episode生成规则应用区间，其中168条有数值变化、共4,246帧；18条的规则应用未产生数值变化。其余80条未应用。按任务计，63任务的两条均改变、42任务有一条改变、28任务两条均未改变。数值改动数不是抓持准确率；本轮没有逐条视觉真值。

174个拒绝候选分别为：稳定时间不足115、释放边界不明确45、闭合边界未对齐6、边界对齐后持物时间不足4、相互冲突4。没有应用的80个有效episode侧分别为：边界条件未通过39、未进入配置开度带27、稳定时长不足14。全部例子保留在GUI和[逐条结果表](reports/20261104_task_profile_two_per_task_results.csv)，没有只展示成功样本。

独立审核 `runs/20261104_task_profiles_two_per_task_audit_v1/verification.json` **PASS，163,440项检查、0失败**：从原曲线穷举窗口并核对全部候选／事件；逐项核验真实接触起点、0.6秒稳定确认、同带持物、固定时序公式、区间外与无效侧保留、原20维state／时间戳／行数、训练目标未提升及应用mask。210个源Parquet重新计算SHA256、大小、mtime，全部未变。没有修改源视频或源发布版。

执行命令：

```sh
.venv/bin/python -B scripts/check.py
.venv/bin/python -B -m close_gripper profile-clean --manifest runs/20261104_task_profiles_two_per_task_inputs_v1/manifest.json --profiles configs/task_gripper_profiles.json --workers 4 --output runs/20261104_task_profiles_two_per_task_eval_v1
```

统一离线检查 **305项通过、0跳过，退出0**，日志 `runs/20261104_task_profiles_two_per_task_checks_v1/offline_check.log`；真实运行日志为同目录 `evaluation.log`。实现快照在运行中未变，SHA256为 `9e35611a00df923182061a5b274ea8283dfc013f6a35877b3989dc66c20fce3c`。开发中补充了实际接触时刻后移后的确认窗稳定性回归，避免只重算时长而错误确认跨漂移区间。

连续视频GUI为 `runs/20261104_task_profiles_two_per_task_gui_v1/`，536个同侧源视频引用、266例全部保留；没有复制视频字节。`scripts/serve_review.py` 在本机8766端口提供审阅页面及媒体Range，请访问 `http://127.0.0.1:8766/review.html`（远程使用时转发服务器8766端口）。

实际Chromium核验 `runs/20261104_task_profile_video_browser_v1/attempt-003/browser_verification.json` **16项通过**：266条／133任务元数据、每任务两例筛选、搜索恢复、规则曲线显示、区间跳转、非零视频偏移、真实像素解码与播放推进、HTTP 206、切换episode停止旧视频、双臂同步、两侧各自双目切换和倍速。页面JavaScript错误、失败请求、外部请求均0；连续视频播放抽查非零偏移例和双臂例，未声称536段均完整播放。截图为同目录 `bilateral_rule_curves.png`。

首次浏览器启动因OS沙箱限制失败，升级执行后实际启动；第二次检查脚本因对SVG使用`innerText`失败，修为`textContent`后第三次全部通过，原失败记录保留。两次均未发生页面JavaScript错误，也没有因此重跑或更改规则生成结果。

完整数值和展示交付见[汇总](reports/20261104_task_profile_two_per_task_summary.json)。本次是规则曲线试跑，不是全量派生发布或训练效果验收。

## 任务开度配置：全量统计与原生视觉校准

分支 `feat/task-gripper-profiles-20261103`，基线 `84c6895`。已交付[配置](configs/task_gripper_profiles.json)、[全部任务表](reports/20261103_task_profiles.csv)、[汇总](reports/20261103_task_profiles_summary.json)和[精简视觉证据](reports/20261103_task_profiles_evidence.json)。业务规则及启用语义只在 PLAN 定义；本节记录实际结果。

| 核查项 | 实际完成 |
| --- | --- |
| 全量数值扫描 | 三发布版、7子集、1,961个Parquet、125,236条发行episode、15,041,374帧；108,472个独立原始录制；150任务、151有效任务侧 |
| 数值窗口 | 1,230,356个按包含关系最大的稳定窗，其中772,042个满足0.6秒；窗口可相互重叠，不能当作抓取次数 |
| Codex 原生视觉 | 302个任务侧样本，来自301个独立录制／发行episode；每任务侧两个不同来源。242支持夹持、40不支持、20不确定；均为模型来源、非人工真值 |
| 补充复核 | 28例查看同侧另一目，8例由另一agent交叉复核；Qwen API请求0。开度由对齐的state读取，未从画面猜测数字 |
| 可用配置 | 133任务、134任务侧允许数值匹配；166个夹持带、249个局部张开参考带 |
| 禁用配置 | 17任务：9个已审样本不支持，8个需要补证；不等于确认这些任务的全部episode都无抓取 |
| 支持强度 | 27个可用任务侧只有一条正例，166个夹持带中91个仅一来源支持。没有把单例校准称为双来源一致验证 |
| 来源与区间审核 | 独立执行16,018项依据断言，核验302样本、151任务侧、1,216个实际帧文件摘要、原曲线、模式归属与反例阻断，全部通过 |
| 输入保留 | 再计算全部1,961个源Parquet的SHA256、大小、mtime及元数据摘要，变化数0 |

数值分布在 `runs/20261103_task_profiles_census_v2/`，逐样本视觉记录在 `runs/20261103_task_profiles_build_v1/visual_reviews_v3.json`。全量扫描的v1曾因先消耗短平台而遗漏跨边界的稳定窗；已改为线性时间枚举最大的稳定窗口，并在新的v2全量重扫。v1的视觉样本都能在v2中找到包含其完整区间的窗口，身份、数值和精确模式归属另经核验，未覆盖旧证据。

全数据整窗数值匹配覆盖 **71,959 / 108,472独立录制（66.34%）**、82,845 / 125,236发行episode，合计336,570个窗口。它只表示至少有一个保存的完整稳定窗落入可用带，不是抓持准确率或清洗成功率；没有穷举从更长窗口中截出的全部可匹配子区间。覆盖与源文件复核记录在 `runs/20261103_task_profiles_coverage_v2/coverage.json`，依据审核在同目录 `catalog_basis_audit.json`。

真实缓存回读 `runs/20261103_task_profiles_runtime_verification_v2/`：302例全部解析到正确中文任务／侧别并通过输入摘要核对；241例匹配，34例因配置禁用拒绝，27例在可用带外拒绝。匹配例全为本轮支持夹持的校准样本；一条支持样本因同任务近零混合持物／空夹反例而禁用。此结果不是留出集评估。确认时间为原平台起点加0.6秒，返回起点保持不变。以“拿起杯子”实际帧33～56为例，原开度约0.6906、确认帧39，未写目标字段。

实际成功执行的入口（退出码均0）：

```sh
.venv/bin/python -B scripts/survey_task_grippers.py --output runs/20261103_task_profiles_census_v2 --workers 4
.venv/bin/python -B scripts/build_task_profiles.py --census runs/20261103_task_profiles_census_v2 --scope runs/20261103_task_profiles_build_v1/scope_and_aliases.json --reviews runs/20261103_task_profiles_build_v1/visual_reviews_v3.json --output runs/20261103_task_profiles_catalog_v3
.venv/bin/python -B -m close_gripper.task_profiles validate configs/task_gripper_profiles.json
.venv/bin/python -B -m close_gripper.task_profiles match configs/task_gripper_profiles.json --episode runs/20261103_task_profiles_visual_group_b_v1/cases/108f6b8c3c374e70/episode.json --side right --start 33 --end 56
.venv/bin/python -B scripts/check.py
```

统一离线检查 **283项通过、0跳过**，日志 `runs/20261103_task_profiles_build_v1/final_offline_check.log`；真实数据核验单独执行。配置SHA256为 `3a29ea422befb087278bd2480ff2e27fba6d602e8f306156b6d13f6a63186552`，与独立审核版本逐字节一致。原生视觉的完整图片保留在本地run，没有提交Git。

本次完成任务配置、数值读取与匹配，并保留原拟合方法。`clean` 尚未接入这个任务配置；本轮未生成新的全量目标、派生数据或训练效果。独立语义验收、未审模式及17项禁用配置仍有待验证；已有小样本训练读取证据不能当作这版配置的训练接入结果。

## 当前阶段：完整周期实现与分层验证

`clean` 已实现对完整输入 episode 的有效侧、闭合周期及剩余观察窗口逐项处理，输出独立目标、有效标记、阶段、来源和覆盖信息。模型判断通过视觉结构与数值门控后可以生成 `source=model` 的自动标签；`valid=true` 不代表人工确认，候选窗口已审阅也不代表每帧视觉语义正确。闭合与释放采用固定时序，主臂仅供离线参考。原始观测和外部源项目保持只读。

最终执行 `.venv/bin/python -B scripts/check.py`：**退出0，227 项测试通过，0 跳过**，日志 `/tmp/close_gripper_full_cycle_final_check.log`。该数字证明所覆盖的软件契约，不是任务质量指标。[精简汇总](reports/20261103_full_cycle_delivery_summary.json) 仅包含数值、状态与证据摘要，不包含真实相机图像、模型原文或凭据。

### 8 条发布 episode 的完整周期 pilot

运行目录：`runs/20261103_full_cycle_pilot_v1/`，配置版本 `full-cycle-v1`。8 条来自三个发布版、5 个子集，含左／右有效侧及双臂多周期例子。处理全部 43 个检测候选，实际模型请求 43 次、0 重试，服务均报告用量：输入 312,564、输出 15,014、合计 327,578 tokens。

| 结果口径 | 数量 | 含义 |
| --- | ---: | --- |
| 候选／请求 | 43／43 | 请求覆盖候选，不等同于有效标签覆盖 |
| 有效结构化判断 | 38 | 其余 5 个未形成满足契约的判断，仍有未审阅范围 |
| 自动接受周期 | 17 | 模型与数值规则接受，未经独立人工确认 |
| 数值拒绝／无抓取判断 | 11／10 | 无抓取结论只针对相应候选，不能扩展为整段无抓取 |
| 有效目标帧侧 | 776 | 按帧与左右侧计数，不能当作唯一 episode 帧数 |
| 变化 episode 帧／变化帧侧 | 450／452 | 同一帧两侧变化在前者中只计一次 |

6 条的有效侧均有完整候选窗口判断；2 条双臂 episode 仍有未审阅范围，不能声称 8 条都已完整清洗。此 pilot 的标签随后用于派生及 loader 接入验证；后续 `full-cycle-v2` 的独立实验不覆盖或改写该批记录。

### 5 条同源主从／真实 VLM 对照

从历史 accepted280 的冻结来源分组选择短录制：3 calibration、2 heldout，按相同 source_id 提取右腕双目和从臂。输入在 `runs/20261103_paired_reference_inputs_v1/`；原始主从数组与历史 NPZ 精确一致。相机话题、frame_id 和消息时间字段逐项核对，使用原始 header／相机 timestamp 的共同绝对时间域，无误差优化平移。不可解码的开头 GOP 与时间覆盖边界明确裁剪；126 个 10 Hz 时间点的从臂最近采样差最大 5.983 ms、相机最大 7.484 ms。独立硬件曝光同步误差未被证明。

模型运行：`runs/20261103_paired_full_cycle_v2/`，`full-cycle-v2`、提示词 `gripper-full-cycle-v2.1`，5 次真实请求、0 重试，服务报告 35,605 tokens。5 个闭合周期自动接受，126／126 右侧帧为 model valid；没有释放周期。输入仅 state[19] 来自从臂，其他维为明示占位零，`valid_sides=['right']`、`not_for_training=true`；主臂另存参考，未输入生成器。这不是正式发布数据或训练数据。

独立评估：`runs/20261103_paired_reference_evaluation_v2/metrics.json`，全部帧与 model-valid 帧分别计算，本批分母恰好相同；无有效帧的指标为 `null`，不补零。

| 冻结分组 | model valid | 对主臂参考的逐帧加权 MAE，原从臂→新目标 | 模型声称持物区间的距零残余，原→新 |
| --- | ---: | ---: | ---: |
| calibration，3 条 | 75／75 | 0.149519→0.015579 | 29 帧，0.336751→0 |
| heldout，2 条 | 51／51 | 0.097097→0.060776 | 20 帧，0.090605→0.007261 |

上述 MAE 与残余均按对应帧逐帧加权，单位为 0～1 开度；不是简单的来源等权平均，主臂指令也不是语义真值。生成器 `phase=hold` 区间归零是构造性质，不能据此证明持物判断准确。兔子例 `ef64d52b` 的模型持物起点为 14 帧，数值接触点为 18 帧，相差 0.4 秒，因此模型声称持物的前 4 帧仍在目标闭合段中。

闭合局部幅度 10%→90% 的来源等权平均时长，原从臂／新目标／主臂分别为：calibration **0.2025／0.2002／0.2210 秒**，heldout **0.2508／0.2493／0.1526 秒**。新目标基本保留从臂时序，不能声称闭合速度已匹配主臂。跨点线性插值只是估计，原观测分辨率仍为 0.1 秒；绝对 0.9→0.1 穿越未发生时记为未知。

可分享的[同源真实曲线](examples/paired_full_cycle_v2_reference.png)只含数值曲线，标题列出真实模型运行来源、任务名和“无独立语义真值”。`runs/20261103_paired_reference_audit_v1/audit.json` 从原始 NPZ 独立复算同网格数值，确认原评估及分组指标，保留 v2 全部原件；分享图基于复核数据独立绘制。5 条是短闭合便利样本，缺少释放、双臂、多次抓放和独立人工语义真值；参考距离降低不能外推为总体准确率或训练收益。

后续 `full-cycle-v3` 抓前准备段修复对这 5 条进行了数值不变性核验：`runs/20261103_paired_numeric_v3_invariance/verification.json` 逐条核对 episode 指纹和候选相等，再复用相同 decisions 重算，target、preview、valid、phase、source 均与 v2 完全相等，新增模型请求 0。上述数值因而仍适用于这 5 条输入；这只是同证据数值复核，不是新的独立 VLM 或质量评估。

### 小样本派生与实际 ORION loader

当前派生记录为 `runs/20261103_full_cycle_delivery_v3/delivery_index.json`：工作区内 **5 子集、8 episode、1,143 帧**，有效目标帧侧左463、右452。它采用最终 `full-cycle-v3` 数值重放的标签；8 条输入与原 pilot 相同，但视觉判断来自后续100条真实模型运行 `runs/20261103_full_cycle_100_v3/`，不混用早期43次 pilot 判断。原列及 episode 指纹经过核对，新目标、valid、来源、元数据与训练配置独立保存；新增模型请求0，`publication_ready=false`。

当前实际 loader 验证位于 `runs/20261103_orion_real_cycle_verify_v3/verification.json`。目标仓库核实为 `/gpfs/jiuquyun/projects/youzi/orion-0`，基线提交 `e08dea45b92198c6b94c1b94236215ac30669640`；其已有未提交改动另存差异，补丁基于“该提交＋已有差异”，不冒充干净上游版本。接入只在工作区隔离副本完成，18 个源文件逐字节核对未改变。

真实 `get_dataloader` 读取 5 个派生子集，共检查 **27 个 batch（batch size=1）、396 个时间×侧别的夹爪动作值**：97 个使用有效标签、299 个回退，62 个动作值实际改变。原20维布局的9／19维表示夹爪；验证在其经 ORION 统一动作映射后的128维 batch 夹爪通道上完成，20 Hz 输出与基于实际源时间戳的插值参考一致。物理观测、其余126维动作、视频时间与 mask 保持不变。真实 loader 也拒绝了非均匀时间戳回归夹具。该结果证明隔离接入与具体 batch 读取，未部署至外部原仓库、未训练模型，也未测量实际抓持改善。

历史 v2 产物保留于 `runs/20261103_full_cycle_delivery_v2/` 与 `runs/20261103_orion_real_cycle_verify_v2/`：同样8条、1,143帧，但标签来自早期 `full-cycle-v1` pilot；当时检查29个样本、436个动作值，其中121个采用标签、315个回退、74个实际改变。它们证明当时的接入，不与当前 v3 统计合并。

### 三版清单与 100 条完整周期结果

`runs/20261103_full_cycle_delivery_v3/release_plan/release_plan.json` 已只读盘点 **3 发布版、7 子集、125,236 episode、15,041,374 帧**，并登记100条最终标签产物。此清单是元数据及待处理范围；登记的标签数量不代表全量已清洗、已逐行校验或已派生。实际数据派生仍限于上述8条。全量数据、完整质量验收及正式发布仍未完成。

100 条冻结发布 episode 共 8,655 帧，来源覆盖三发布版、七子集。其原始录制身份仍不足以建立无泄漏的 calibration／heldout 分组；这是冻结配置样本检查，不能混同于前述按原始 source_id 分组的5条主从对照。

首次 `runs/20261103_full_cycle_100_v2/` 在受限网络环境中有 316 次 `network_error` 尝试、0 个服务响应、0 条有效模型判断；没有返回用量，**token 用量未知**，汇总器的零累计数不能解释为已知零消耗。该失败不作为算法效果。

随后 `runs/20261103_full_cycle_100_v3/` 使用固定 `full-cycle-v2` 数值配置、提示词 `gripper-full-cycle-v2.1`，真实请求 316 次、0 重试，316 次均返回服务用量：输入 **2,178,531**、输出 **98,590**、总计 **2,277,121 tokens**。源码在整个模型运行中未变。294 个结构化判断包括：`no_grasp` 97、`grasp_release` 131、`grasp_hold` 63、`uncertain` 3；22 个解析拒绝分别为非抓取证据矛盾13、抓前准备证据矛盾9。结构判断通过比例不是视觉准确率，`no_grasp` 仅针对相应候选。

数值审阅发现茄子 episode1025 的已确认空闲准备段为0～15帧，旧规则仅从14帧的最高峰开始清理，保留了之前的半开半合。`full-cycle-v3` 对从 episode 第0帧开始、视觉明确确认的准备段补全稳定化；非0起点仍保留连接边界，并分别报告完整、部分清理及上下文保留。修复后的最终运行是 **`runs/20261103_full_cycle_100_replay_v4/`**：复用原始图像、模型响应及 decisions，新增模型请求 **0**，原316次用量只列为复用，不重复计为新调用或独立评估。

| 最终重放口径 | 数量 |
| --- | ---: |
| 自动接受周期／候选总数 | 134／316 |
| 有变化的 episode | 75／100 |
| 有变化的唯一 episode 帧 | 3,543／8,655 |
| 有变化的帧侧 | 3,604：左388、右3,216 |
| 有效目标帧侧 | 5,805：左1,355、右4,450 |
| 所有有效侧候选窗口审阅完整的 episode | 83／100 |
| 仍缺结构化证据、候选审阅不完整的 episode | 17／100 |

134 个接受周期的准备段状态分别为 `cleaned` 44、`partially_cleaned` 36、`unresolved_context_preserved` 9、`not_modified` 45。源数值结果的唯一变化帧为3,328，最终为3,543，相差215；这是同一视觉证据下数值规则改变的影响，不是新增独立效果证据。接受数、模型类别和未审阅范围保持不变。归零、修改量、valid 与候选窗口覆盖均不等同于人工语义正确率，83条候选窗口审阅完整也不表示每帧均已清洗或逐帧验证。

`runs/20261103_full_cycle_100_replay_v4_audit_v1/verification.json` 独立审计100条输入快照、实现快照、同侧图像摘要、原始响应来源、目标与有效标记、覆盖及用量，错误0；审计本身调用模型0次。源和重放实现摘要分别绑定，原始输入、响应和失败证据保留。最终审阅产物为 `runs/20261103_full_cycle_100_replay_v4/review.html`。

最初的逐帧图片 GUI 由该最终产物生成。`runs/20261103_full_cycle_final_gui_check_v1/verification.json` 核对100条原始／目标／valid／phase／source数组和7,546个图片引用，HTTP200；这一轮尚未运行完整浏览器自动化。茄子例最终曲线已目视确认0～15帧稳定在0.995088，15～18帧连续闭合到0，原始观测曲线保留。该目视检查验证展示，不升级为人工抓取语义真值。当前带连续视频的界面见下节。

正式发布前确认来自任务书原约定，目前尚未到正式发布阶段；已完成的是实现、小样本证据、隔离接入和发布准备，不是整个任务验收。

### GUI 连续视频与曲线同步

按用户要求，当前 [8766审阅页](http://127.0.0.1:8766/review.html) 切换到 `runs/20261103_full_cycle_100_video_review_v1/`。100条 episode 与220个有效侧腕部视频逐一绑定，视频和开度曲线并排显示，支持同步播放／暂停、时间条和曲线定位、逐帧、速度、双目切换及全屏；模型证据帧保留在折叠区域。切换 episode 时停止旧视频，播放范围限于该 episode。

本次仅重新展示 `full_cycle_100_replay_v4` 的既有结果，**新模型请求0次，数值算法执行0次**。原JSON逐字节复制，7,546张证据图片按SHA核验。独立 `runs/20261103_video_gui_validation_v1/verification.json` 核对全部100条页面的原值、目标、候选、valid、阶段、来源及220组相机、视频路径和偏移均与原产物一致。源MP4通过白名单只读流式提供，支持HTTP Range；URL摘要由规范路径及stat生成，不是视频内容SHA。源视频前后核对size、mtime、device和inode，未复制或转码视频。

真实 Chromium 153／Playwright 1.64.1 验证位于 `runs/20261103_video_browser_check_v1/attempt-002/`：19项检查通过，JS、控制台和网络错误均为0。覆盖非零偏移、实帧解码、左右同步、曲线光标推进、暂停／播放中换视角、切 episode 停止旧视频、Range 206及2倍速。抽查 episode15889 的右腕证据帧192：episode 19.2秒加0.5秒偏移，对应视频 `currentTime` 与解码帧 `mediaTime` 均为19.7秒；实际视频帧与原JPEG的RGB像素MAE为0.9083／255。该内容检查证明抽查画面对应，未逐帧检查全部100条。attempt-001因沙箱禁止浏览器启动而未执行交互，记录保留；获工具权限后完成attempt-002。

统一离线检查 `.venv/bin/python -B scripts/check.py` 退出0：**237项测试、0跳过**，日志 `/tmp/close_gripper_video_gui_check.log`。新增媒体偏移与侧别绑定、HTTP Range／HEAD／越界保护和展示产物保留测试；默认HTTP测试使用内存请求，不绑定端口或联网。精简证据见 [视频审阅汇总](reports/20261103_video_review_summary.json)，真实截图、视频和完整运行记录不提交Git。本次验证不新增视觉判断、人工参考或训练效果结论。

## 历史记录：局部候选流程与开度展示修正

以下按当时状态保留。文中的“当前”“尚未完成”和旧算法限制均指各轮历史运行，不能用来替代上述完整周期阶段的现状，也不作为新的任务边界。

### 实际执行的检查

| 命令／检查 | 结果 | 验证范围 |
| --- | --- | --- |
| `.venv/bin/python -m pip install -e '.[data,plots,test]'` | 退出0 | 包安装与CLI注册 |
| `.venv/bin/python -B scripts/check.py` | 退出0，91 tests，0 skip | 文档本地引用、源码语法、离线契约 |
| `.venv/bin/python -m pytest -q` | 退出0，91 passed、77 subtests passed | 同一测试集的pytest入口 |
| `.venv/bin/close-gripper --help` | 退出0 | 已安装入口 |
| `.venv/bin/close-gripper demo --output runs/20261103_delivery_demo` | 退出0 | 五组合成示例、PNG、HTML与独立目标 |
| `git diff --check` | 退出0 | 本次改动空白格式 |
| 暂存内容与本地实际密钥逐字节比对 | 未发现密钥；凭据文件被排除 | 提交内容检查，不打印密钥 |

契约覆盖单／双侧、短暂张开、空夹、噪声、未触发、近零平台、初始持物、速度与平台、原下降段与观测不变、持物不足、跨释放／下一抓取、未知视觉结果、模型不能自确认、合成不入训练、输入摘要绑定、来源分组、软链接与覆盖保护、共享Parquet与实际临时视频解码、失败与请求预算、401/403停止后续请求。生成HTML的JavaScript由Node做语法检查，数据文本插入攻击有回归测试；未运行完整浏览器自动化。

环境：Python 3.13.15、NumPy 2.5.4、PyArrow 26.0.0、OpenCV 5.0.0.93、Matplotlib 3.11.2、pytest 9.2.0。日志和环境记录位于上述run目录。运行产物默认不入Git；可移植的合成[曲线图](examples/synthetic_curves.png)与[审阅页](examples/synthetic_review.html)已纳入版本控制。

### 真实数据与模型请求

只读加载 remaining 发布版 `baseline-v4-pushcart-bimanual-v1` 的 episode 11：300帧、左右有效、4个腕部相机。验证左侧98→99帧为8.8358→45.2795 mm，视频偏移0.5秒。当前实验配置检测出12个候选；选择左侧97、187帧及右侧188帧开始的3个候选，实际提取177张同侧腕部图片，保留10 Hz。

运行：`runs/20261103_qwen_episode11_v1/`。模型指定 `qwen3.8-max-2026-09-02`，北京接口，0重试，共3次实际请求。**三次均HTTP 401；成功响应0，服务没有返回token用量，因此用量未知，不能记为0。** 未取得任何真实视觉判断；真实训练目标全部沿用原始开度，有效标记全部false。审阅页可查看实际曲线、同侧图片及错误记录。失败后已增加客户端认证拒绝保护，其行为由离线测试验证；没有继续尝试真实请求。

合成曲线示例展示30 mm平台按原下降速度继续降到0、误开候选抑制、正常释放连续衔接。释放固定时序方案仍属候选；原速方案额外所需时间会显示，确认范围不足时不越界。合成示例不是Qwen结果或人工参考，未写出有效训练标签。

### 样本清单与未覆盖范围

`runs/20261103_samples48_v1/manifest.json` 含48条样本，覆盖7子集、32个任务；各子集数量为8、8、8、7、7、3、7。来源仅为元数据分层，尚未完成视觉语义覆盖审阅。缺少可靠的原始episode对应关系，48条均为 `unassigned`，不能声称已建立无泄漏校准／留出划分。

尚未覆盖：真实Qwen成功调用与正确率、人工参考与参数校准、最终释放策略、跨data shard单episode读取、外部实际训练loader与20 Hz网格接入、训练效果、全量处理及派生数据发布。当前提供独立JSON sidecar与本地动作替换合同，未生成或发布 `-gripcmd` 数据集。正式数据发布仍按原约定确认。

### 后续更新：闭合拟合与完整凭据读取

按用户后续要求，闭合默认与张开一样进行固定起止时刻的端点拟合，算法更新已写入 PLAN。旧原速延伸可通过 `closing_policy=original_speed` 选择。审阅页同时提供两种闭合曲线与额外时间对比，并明确显示当前策略；正常释放仍仅生成候选。新的[局部对比图](examples/closing_opening_fit.png)、[完整示例](examples/synthetic_curves.png)和[交互页](examples/synthetic_review.html)均为合成数据。

用户指定的工作区 `qwen_api.txt` 现为默认凭据来源。排查发现该key含点号，旧解析器截断了它；已改为完整匹配，非法后缀直接拒绝。密钥仅用于鉴权头，不写入请求正文、结果或Git；对应测试使用虚构凭据。

修复后实际执行1次请求（0重试），运行目录为 `runs/20261103_closefit_qwen_keyfix/`，输入为 episode 11 左侧97→101帧候选及55张同侧腕部图片。**请求成功**，返回模型 `qwen3.8-max-0902`；服务报告输入10,131 tokens、输出306 tokens、总计10,437 tokens，其中图片输入3,630 tokens。

模型判断为正常释放，给出持物起点89帧、终点97帧。数值核查因 `unstable_platform_or_contact_mismatch` 回退，因此真实标签仍原样保留，有效标记仍false。这是成功调用及回退链路验证，未建立人工真值或测量真实视觉正确率。首轮401记录保留，不覆盖成成功记录；当前真实Qwen连通性已验证，其他研究与训练验收仍未完成。

本次最终检查：`scripts/check.py` 通过112项、0跳过，`pytest -q` 为112 passed；新增覆盖默认闭合拟合、非匀速相位保留、原速对照、已回退释放区间仍受保护、下一抓取边界、两种策略的报告数据和完整含点号密钥。日志及最新合成运行位于 `runs/20261103_closing_fit_final/`。检查确认完整真实密钥未出现在跟踪文件和本次运行产物中，`qwen_api.txt` 没有Git提交历史。

### 100 条真实 episode 的固定配置检查

按用户要求完成100条真实episode，各检查1个明确选定的张开事件；不是100条完整episode的全事件清洗。扫描632条发布episode，选中100条，另有228条可用未选、280条未触发、24条没有合适的事件边界，读取失败0。覆盖91种任务、全部7个子集；包含取物、挂衣物、插放餐具／工具、开关门／抽屉／家电、叠衣服等任务。双臂子集预留10条，左5、右5；本批合计左5、右95。选样以任务多样性为目标，不按发布数据自然比例抽样。

清单、冻结episode快照及筛选脚本：`runs/20261103_real100_selection/`。精确时间戳＋左右夹爪轨迹摘要均不同，但不同裁剪是否来自同一原始录制仍未经验证；全部split为 `unassigned`。数值特征标签只用于选样，不是抓持或误操作真值。67条前窗受episode开头裁剪，24条不足完整1秒后窗，但均保留至少0.3秒后续且不越界补帧。

实际命令：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m close_gripper batch \
  --manifest runs/20261103_real100_selection/manifest.json \
  --config configs/real100.json --output runs/20261103_real100_qwen \
  --workers 4 --qwen --api-key-file qwen_api.txt
```

命令退出0。配置固定、4并发、每条1次请求、0重试；没有根据中途结果调整提示词、阈值或算法。共100次真实请求、100次服务JSON响应，输入4,361张同侧腕部图像，返回模型为 `qwen3.8-max-0902`。服务报告输入839,045 tokens、输出29,627 tokens、合计868,672 tokens；100次均返回用量。

| 模型结果／结构校验 | 数量 |
| --- | ---: |
| 正常释放 | 40 |
| 未抓持 | 30 |
| 不确定 | 14 |
| 疑似误张开 | 4 |
| 未通过证据契约校验 | 12 |

前四项合计88条结构化判断；12条拒绝分别为缺少严格张开后证据9、释放边界矛盾1、抓持状态矛盾2，均有服务响应，网络与认证错误0。**88/100是输出校验通过比例，不是视觉准确率。**

数值处理仅生成1条变化候选：case007（叠衣服、右侧、episode3278），31帧候选变化，约2.08 mm持物平台映射到0。该事件属于正常释放，释放策略仍待复核。全部100条原始观测、时间戳、帧数保持不变；全部有效训练目标沿用原值、valid=false。其余153个检测到的候选未调用模型，不纳入已检查统计。

持物事件的数值处理原因：episode起始已持物／缺初始下降17、模型给定接触点前无有效下降10、平台与接触点不一致6、近零平台保持原值7、无效持物边界2、退出边界未确认1、候选生成待释放复核1。候选产生少不能直接推导视觉错误率，也不能把合理回退视作成功清洗。

独立审阅发现三个需要后续研究的具体问题：

- `hold_start` 的视觉含义与数值接触点混用。例如case008的下降在160–163帧，模型给出164帧，导致无法拟合下降速度；应将视觉持物范围与数值接触定位分开核对。
- 微小正漂移会提前候选起点。case042中61–65帧仅9.060→9.108 mm，66帧才到44.351 mm；模型给定持物结束65帧与检测候选起点61帧冲突。触发窗口起点与确认释放起点需要区别。
- 疑似误开文本包含“调整抓法或误开”等歧义，不能升级为确认误操作；case010还存在模型文字所述开度与实际数值不符，数值门控保留了原值。

交付入口：本地 `runs/20261103_real100_qwen/batch_review.html` 可按任务、左右侧、类别与回退原因筛选，并逐条查看同步腕部证据和曲线。汇总图与JSON见[统计图](examples/real100_summary.png)、[统计数据](reports/20261103_real100_summary.json)。实际候选局部放大为run内 `case007_curve_detail.png`，原图证据为 `case007_evidence.png`。真实图片、完整模型响应与密钥均未纳入Git。

完成证据：独立审阅 `audit_final.json` 和 `artifact_verification.json` 核对100条精确候选ID、侧别、同侧图像、输入摘要与标签不变，所有请求使用同一个实现摘要；本轮离线入口与pytest均122项通过、0跳过。仅在计算结束后修正PNG过长标题并重绘展示，未改变模型响应、算法结果或本批统计。仍无人工参考、真实准确率或训练效果验证。

### GUI 开度展示核对

对照 `Untitled.md` 与数据转换代码确认：输入9/19维及独立目标字段均为归一化夹爪开度。原审阅图把数值乘95后标成 `Width (mm)`；现将HTML、PNG和逐帧读数统一为0～1，并明确标注实测开度、候选目标开度和有效目标开度。语义及未完成范围见 PLAN。

新审阅入口为 `runs/20261103_real100_opening_review/batch_review.html`。使用缓存重绘100条真实样本，复制4361张已有证据图片；本次模型请求0次。`render_summary.json` 记录源JSON摘要，重绘前后摘要一致，HTML内原值、候选值、目标值与有效标记逐一对应原结果。原批次的100条模型结论和候选变化数量保持不变。

合成示例重绘于 `runs/20261103_opening_units/`，并同步到 `examples/`。静态图已目视核对0～1坐标及闭合／释放两端的原始与候选曲线。执行 `.venv/bin/python -B scripts/check.py`：122项通过、0跳过，包含HTML脚本语法与已有业务契约；`git diff --check` 退出0。本次仅验证展示与数值保留，未新增真实VLM、人工参考或训练效果验证，未运行完整浏览器自动化。
