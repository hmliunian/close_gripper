"""Offline batch review index. Reads no credentials, images, or model services."""

from collections import Counter
import html
import json
import math
from pathlib import Path, PurePosixPath
from urllib.parse import quote


def _escape(value):
    return html.escape(str(value), quote=True)


def _labels(value):
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return list(dict.fromkeys(str(item) for item in values if item is not None and str(item)))


def _count(value):
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else 0


def _usage(row):
    value = row.get("usage") or {}
    if not isinstance(value, dict):
        return 0, 0, 0, False
    known = any(key in value for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens"))
    inputs = _count(value.get("prompt_tokens", value.get("input_tokens", 0)))
    outputs = _count(value.get("completion_tokens", value.get("output_tokens", 0)))
    total = _count(value["total_tokens"]) if "total_tokens" in value else inputs + outputs
    return inputs, outputs, total, known


def _links(relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        return "—"
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("..", ".") for part in path.parts) or ":" in relative:
        return "—"
    prefix = quote(path.as_posix(), safe="/")
    return (f'<a href="{_escape(prefix)}/review.html">同步审阅</a> '
            f'<a href="{_escape(prefix)}/curves.png">开度曲线图</a>')


def _counter_table(title, counts):
    items = "".join(f"<tr><td>{_escape(key)}</td><td>{count}</td></tr>"
                    for key, count in counts.most_common())
    return f'<section class="distribution"><h2>{_escape(title)}</h2><table><tbody>{items or "<tr><td>无记录</td><td>0</td></tr>"}</tbody></table></section>'


def write_batch_report(run_dir, rows):
    """Write batch_review.html from completed per-episode summaries, returning Path.

    ``rows`` follows the batch runner contract: source identity and task labels,
    selected candidate, model status/category, core reason, preview versus accepted
    changed-frame counts, request attempts/usage, and a relative episode directory.
    Counts describe this selected sample, not dataset accuracy or training gains.
    The caller owns the new run directory and any summary JSON.
    """
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("batch report rows must be a list of dictionaries")
    run_dir = Path(run_dir)
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError("batch report requires an existing run directory")
    statuses = Counter(str(row.get("status") or "unknown") for row in rows)
    sides = Counter(str(row.get("side") or "未选择") for row in rows)
    categories = Counter(str(row.get("category") or "未判定") for row in rows)
    reasons = Counter(str(row.get("reason") or "未记录") for row in rows)
    tasks = {label for row in rows for label in _labels(row.get("task"))}
    source_tasks = {label for row in rows for label in _labels(row.get("source_tasks"))}
    all_tasks = tasks | source_tasks
    usages = [_usage(row) for row in rows]
    inputs, outputs, tokens = [sum(value[i] for value in usages) for i in range(3)]
    requests = sum(_count(row.get("request_attempts")) for row in rows)
    missing_usage = sum(_count(row.get("request_attempts")) > 0 and not usage[3] for row, usage in zip(rows, usages))
    preview_rows = sum(_count(row.get("preview_changed_frames")) > 0 for row in rows)
    accepted_rows = sum(_count(row.get("accepted_changed_frames")) > 0 for row in rows)
    preview_frames = sum(_count(row.get("preview_changed_frames")) for row in rows)
    accepted_frames = sum(_count(row.get("accepted_changed_frames")) for row in rows)
    model_ok = statuses["model_ok"]
    cards = [
        ("样本 episode", len(rows), "每条仅检查一个代表张开事件"),
        ("发布 / 来源任务", f"{len(tasks)} / {len(source_tasks)}", "分别按传入任务名去重"),
        ("左 / 右侧事件", f"{sides['left']} / {sides['right']}", f"未选择或其他：{len(rows)-sides['left']-sides['right']}"),
        ("模型调用与解析成功", model_ok, f"实际请求尝试 {requests} 次；失败样本 {statuses['model_error']} 条"),
        ("候选曲线有变化", preview_rows, f"合计 {preview_frames} 帧；仅供复核"),
        ("有效目标有变化", accepted_rows, f"合计 {accepted_frames} 帧；与候选分开统计"),
        ("已报告 token", f"{tokens:,}", f"输入 {inputs:,} / 输出 {outputs:,}；{missing_usage} 个已请求样本未返回用量"),
    ]
    cards_html = "".join(f'<article><span>{_escape(title)}</span><strong>{_escape(value)}</strong><small>{_escape(note)}</small></article>'
                         for title, value, note in cards)
    rows_html = []
    for ordinal, (row, usage) in enumerate(zip(rows, usages)):
        task_names, source_names = _labels(row.get("task")), _labels(row.get("source_tasks"))
        status = str(row.get("status") or "unknown")
        side = str(row.get("side") or "未选择")
        category = str(row.get("category") or "未判定")
        reason = str(row.get("reason") or "未记录")
        searchable = " ".join(map(str, (row.get("episode_id", ""), row.get("subset", ""),
                                        row.get("release", ""), *task_names, *source_names, side, status, category, reason)))
        attrs = f'data-search="{_escape(searchable.lower())}" data-tasks="{_escape(json.dumps(task_names + source_names, ensure_ascii=False))}" data-side="{_escape(side)}" data-status="{_escape(status)}" data-category="{_escape(category)}"'
        start = row.get("candidate_start")
        end = row.get("candidate_end", row.get("end"))
        interval = f"{start} → {end}" if start is not None and end is not None else "—"
        elapsed = row.get("elapsed_s")
        duration = f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) and math.isfinite(elapsed) else "—"
        task_text = " / ".join(task_names) or "未记录"
        source_text = " / ".join(source_names) or "未记录"
        rows_html.append(f'''<tr {attrs}>
<td>{_escape(row.get('index', ordinal))}</td>
<td class="identity"><b>{_escape(row.get('episode_id', '未记录'))}</b><small>{_escape(row.get('release', ''))}<br>{_escape(row.get('subset', ''))}</small></td>
<td><b>{_escape(task_text)}</b><small>来源：{_escape(source_text)}</small></td>
<td>{_escape({'left':'左','right':'右'}.get(side, side))}<small>帧 {_escape(interval)}</small></td>
<td><span class="badge {'ok' if status == 'model_ok' else 'review'}">{_escape(status)}</span><small>{duration} / {_count(row.get('request_attempts'))} 次尝试</small></td>
<td>{_escape(category)}</td><td class="reason">{_escape(reason)}</td>
<td>{_count(row.get('preview_changed_frames'))} / {_count(row.get('accepted_changed_frames'))}</td>
<td>{f'{usage[2]:,}' if usage[3] else '未返回'}</td><td class="links">{_links(row.get('episode_dir'))}</td></tr>''')
    def options(values):
        return "".join(f'<option value="{_escape(value)}">{_escape(value)}</option>' for value in sorted(values))
    body = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>夹爪标签 · 批次审阅</title>
<style>
:root{color-scheme:light}*{box-sizing:border-box}body{font:15px system-ui,sans-serif;color:#183247;background:#f3f6fa;margin:0;padding:28px}main{max-width:1720px;margin:auto}h1{font-size:28px;margin:8px 0}h2{font-size:17px}p{line-height:1.6}.muted,small{color:#61758a}small{display:block;font-size:12px;margin-top:6px;line-height:1.45}.note{background:#fff3d8;border-left:4px solid #c78217;padding:12px 16px;border-radius:4px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:12px;margin:22px 0}article{background:white;border:1px solid #dee6ee;border-radius:10px;padding:16px}article strong{display:block;font-size:29px;margin:9px 0}article span{font-size:14px}.distributions{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}.distribution{background:white;border-radius:10px;padding:12px 16px;max-height:300px;overflow:auto}.distribution td:last-child{text-align:right}.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:22px 0 12px}label{font-size:13px}input,select,button{font:inherit;background:white;border:1px solid #bbcbd9;border-radius:6px;padding:8px}input{min-width:250px}label select{margin-left:5px}.table-wrap{overflow:auto;background:white;border:1px solid #dce5ec;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:11px 10px;border-bottom:1px solid #e5ebf0;vertical-align:top}th{background:#eaf0f6;white-space:nowrap;position:sticky;top:0}td.identity{min-width:240px;max-width:380px;overflow-wrap:anywhere}td.reason{min-width:180px;max-width:280px;overflow-wrap:anywhere}.badge{display:inline-block;padding:3px 7px;border-radius:5px;white-space:nowrap}.ok{background:#e1f2e9;color:#206348}.review{background:#fff0d6;color:#8a601a}.links{min-width:90px}.links a{display:block;margin-bottom:7px}a{color:#176795}tr[hidden]{display:none}button{cursor:pointer}.footer{margin:18px 0;color:#61758a;font-size:13px}@media(max-width:600px){body{padding:12px}h1{font-size:23px}}
</style></head><body><main>
<p class="muted">CLOSE GRIPPER / BATCH REVIEW</p><h1>真实 episode · 夹爪开度曲线审阅</h1>
<p>同步审阅页和曲线图对比原始实测开度与候选目标开度，纵轴统一为0～1（0 = 全闭，1 = 全开）。</p>
<p class="note">本页没有人工参考，不能据此报告准确率、召回率或训练收益。模型成功仅表示调用及结构化解析成功。候选曲线不等于有效训练标签；未选中的张开事件、未触发的持续持物片段不算完成检查。</p>
<div class="cards">__CARDS__</div>
<div class="distributions">__DISTRIBUTIONS__</div>
<p><b>建议审阅顺序：</b>先看产生候选变化的正常释放和疑似误张开，核对同侧腕部、视频偏移、接触点和释放边界；再看被拒绝或判断不明的事件；最后按任务和左右侧抽取未改样本，检查漏检。将人工修订另存为明确来源的复核记录。</p>
<div class="toolbar"><input id="search" type="search" placeholder="搜索任务、episode、原因…" aria-label="搜索记录">
<label>任务<select id="task"><option value="">全部</option>__TASKS__</select></label>
<label>侧别<select id="side"><option value="">全部</option>__SIDES__</select></label>
<label>状态<select id="status"><option value="">全部</option>__STATUSES__</select></label>
<label>类别<select id="category"><option value="">全部</option>__CATEGORIES__</select></label>
<button id="changed" type="button" aria-pressed="false">只看候选变化</button><span id="visible"></span></div>
<div class="table-wrap"><table id="episodes"><thead><tr><th>#</th><th>Episode / 来源</th><th>任务</th><th>检查侧 / 区间</th><th>请求状态</th><th>模型类别</th><th>数值处理原因</th><th>候选 / 有效变化帧</th><th>已报告 token</th><th>可视化</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p class="footer">筛选只影响表格，顶部汇总始终统计整个批次。未返回的 token 用量标为未知，不推算费用。链接为运行目录内的离线审阅页和图片，不依赖外部服务。</p>
</main><script>
const $=id=>document.getElementById(id), rows=[...document.querySelectorAll('#episodes tbody tr')];let onlyChanged=false;
function apply(){const query=$('search').value.trim().toLowerCase();let count=0;rows.forEach(row=>{const data=row.dataset,tasks=JSON.parse(data.tasks),changed=Number(row.cells[7].textContent.split('/')[0])>0;const show=(!query||data.search.includes(query))&&(!$('task').value||tasks.includes($('task').value))&&(!$('side').value||data.side===$('side').value)&&(!$('status').value||data.status===$('status').value)&&(!$('category').value||data.category===$('category').value)&&(!onlyChanged||changed);row.hidden=!show;if(show)count++});$('visible').textContent=`显示 ${count} / ${rows.length} 条`}
['search','task','side','status','category'].forEach(id=>$(id).addEventListener('input',apply));$('changed').addEventListener('click',()=>{onlyChanged=!onlyChanged;$('changed').setAttribute('aria-pressed',String(onlyChanged));$('changed').textContent=onlyChanged?'显示全部变化状态':'只看候选变化';apply()});apply();
</script></body></html>'''
    replacements = {
        "__CARDS__": cards_html,
        "__DISTRIBUTIONS__": _counter_table("模型请求状态", statuses) + _counter_table("模型事件类别", categories) + _counter_table("数值处理 / 回退原因", reasons),
        "__TASKS__": options(all_tasks), "__SIDES__": options(sides),
        "__STATUSES__": options(statuses), "__CATEGORIES__": options(categories),
        "__ROWS__": "".join(rows_html),
    }
    # Replace placeholders in the template only; inserted user text is never
    # interpreted as another replacement marker or as HTML/script content.
    import re
    body = re.sub(r"__(?:CARDS|DISTRIBUTIONS|TASKS|SIDES|STATUSES|CATEGORIES|ROWS)__", lambda match: replacements[match.group()], body)
    output = run_dir / "batch_review.html"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(body)
    return output
