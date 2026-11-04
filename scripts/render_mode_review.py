"""Render saved native-vision platform evidence; never infer or change labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from close_gripper.pipeline import create_run
from close_gripper.review_server import MEDIA_SCHEMA


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def render(reviews_path, coverage_path, catalog_path, output):
    reviews, coverage, catalog = map(read_json, (reviews_path, coverage_path, catalog_path))
    if reviews.get("schema") != "close_gripper.mode_reviews.v1":
        raise ValueError("requires explicit saved native vision reviews")
    records = reviews["records"]
    if len({r["case_id"] for r in records}) != len(records):
        raise ValueError("duplicate review case")
    root = create_run(output, [reviews_path, coverage_path, catalog_path])
    (root / "images").mkdir()
    copied, displayed = {}, []
    for record in records:
        if (record.get("review_source"), record.get("review_method"), record.get("confirmed")) != (
                "model", "codex_native_vision", False):
            raise ValueError("unexpected review provenance")
        item = {key: record.get(key) for key in (
            "case_id", "task", "side", "source_id", "episode_index", "numeric_role",
            "numeric_mode_id", "plateau_start", "plateau_end", "duration_s", "minimum",
            "maximum", "trace", "frame_states", "held_intervals", "free_intervals",
            "interval_semantics", "notes", "source_alignment_uncertain", "reviewer")}
        item["panels"] = []
        for panel in record["panels"]:
            source = Path(panel["path"]).resolve(strict=True)
            raw = source.read_bytes()
            actual = digest(raw)
            if actual != panel["sha256"]:
                raise ValueError("review frame content changed")
            relative = "images/" + actual + source.suffix.lower()
            if relative not in copied:
                (root / relative).write_bytes(raw)
                copied[relative] = {"source": str(source), "sha256": actual, "size_bytes": len(raw)}
            item["panels"].append({key: panel.get(key) for key in (
                "index", "timestamp", "opening", "camera", "sha256")})
            item["panels"][-1]["path"] = relative
        displayed.append(item)
    inputs = {}
    for name, path in (("reviews", reviews_path), ("coverage", coverage_path), ("catalog", catalog_path)):
        raw = Path(path).read_bytes()
        (root / ("source_" + name + ".json")).write_bytes(raw)
        inputs[name] = {"path": str(Path(path).resolve()), "sha256": digest(raw)}
    payload = {"records": displayed, "coverage": coverage,
               "profiles": catalog["tasks"], "inputs": inputs}
    (root / "data.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    (root / "review.html").write_text(PAGE)
    (root / "media_manifest.json").write_text(json.dumps({"schema": MEDIA_SCHEMA, "media": {}}))
    summary = {"schema": "close_gripper.mode_review_display.v1", "status": "complete",
               "presentation_only": True, "cases": len(displayed),
               "tasks": len({r["task"] for r in records}), "unique_images": len(copied),
               "model_requests": 0, "inputs": inputs, "images": copied,
               "renderer_sha256": digest(Path(__file__).read_bytes())}
    (root / "render_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return {k: v for k, v in summary.items() if k != "images"}


PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>稳定平台的夹持证据</title><style>
body{font:16px system-ui,sans-serif;margin:24px;background:#f3f5f8;color:#182333}h1{font-size:24px}
select,input,button{font:inherit;padding:7px;margin:4px;border:1px solid #bac5d2;border-radius:6px}
.box{background:white;border-radius:10px;padding:18px;margin:14px 0}.muted{color:#526174;font-size:14px}
.warn{color:#9e321c}#panels{display:flex;gap:10px;overflow-x:auto}.panel{min-width:220px;flex:1}
.panel img{width:100%;min-width:220px;background:#111}.held{color:#087f49}.free{color:#175eb0}.uncertain{color:#9b6400}
svg{width:100%;height:220px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px ui-monospace,monospace}
details{margin-top:12px}a{color:#175eb0}#meta{overflow-wrap:anywhere}
</style><h1>稳定平台的夹持证据</h1>
<p class="muted">逐模式代表样本的 Codex 原生视觉记录，含稳定段前、起点、中点、终点、之后的实际帧。H=可见夹持，F=空夹或未夹住，U=无法确定。它们是模型证据，尚未经人工确认；代表帧复核不等于全部 episode 的逐帧真值。</p>
<div class="box"><div id="summary"></div><input id="search" placeholder="搜索任务、来源或 case">
<select id="task"><option value="">全部任务</option></select>
<select id="role"><option value="">全部数值角色</option><option>lower</option><option>upper</option><option>static</option><option>mixed</option></select>
<select id="semantic"><option value="">全部视觉状态</option><option value="held">held</option><option value="free">free</option><option value="mixed">mixed</option><option value="uncertain">uncertain</option></select>
<div><button id="previous">上一例</button><span id="position"></span><button id="next">下一例</button></div>
<select id="case" style="max-width:100%;width:100%"></select></div>
<div class="box"><div id="meta"></div><div id="alignment" class="warn"></div>
<svg id="chart" viewBox="0 0 1000 220" role="img" aria-label="原始开度和稳定区间"></svg>
<p class="muted">灰线：原始开度；浅蓝：被复核的稳定区间；彩色圆点：实际查看帧。点击图片可查看原图。</p>
<div id="panels"></div><p id="notes"></p><details><summary>本任务配置及模式审阅状态</summary><pre id="profile"></pre></details>
<details><summary>本例身份、区间与来源</summary><pre id="record"></pre></details></div>
<script>
const $=id=>document.getElementById(id), names={held:'H · 夹持',free:'F · 未夹持',uncertain:'U · 不确定'};
let data,selected=[],position=0;
function option(value,label){const el=document.createElement('option');el.value=value;el.textContent=label;return el}
function filter(){const q=$('search').value.trim().toLowerCase();selected=data.records.filter(r=>(!$('task').value||r.task===$('task').value)&&(!$('role').value||r.numeric_role===$('role').value)&&(!$('semantic').value||r.interval_semantics===$('semantic').value)&&(!q||[r.task,r.source_id,r.case_id].join(' ').toLowerCase().includes(q)));position=0;$('case').replaceChildren(...selected.map((r,i)=>option(String(i),`${r.task} / ${r.side} / ${r.numeric_role}:${r.numeric_mode_id??'未归类'} / ${r.interval_semantics} / ep ${r.episode_index}`)));show()}
function element(tag,text,cls){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el}
function show(){const r=selected[position];$('position').textContent=`${r?position+1:0} / ${selected.length}`;$('case').value=String(position);$('previous').disabled=position===0;$('next').disabled=position+1>=selected.length;if(!r){$('meta').textContent='无匹配样本';$('panels').replaceChildren();$('chart').replaceChildren();$('record').textContent='';$('profile').textContent='';$('alignment').textContent='';$('notes').textContent='';return}
$('meta').textContent=`${r.task} · ${r.side} · ${r.numeric_role}:${r.numeric_mode_id??'未归类尾部'} · 稳定 ${r.duration_s.toFixed(2)} s · 观测 ${r.minimum.toFixed(4)}–${r.maximum.toFixed(4)} · ${r.interval_semantics}`;
$('alignment').textContent=r.source_alignment_uncertain?'该原始来源的信号与图像语义存在疑点，保留视觉记录，但排除数值校准。':'';
$('notes').textContent=r.notes??'';const states=new Map(r.frame_states.map(f=>[f.index,f]));
$('panels').replaceChildren(...r.panels.map(p=>{const s=states.get(p.index)||{state:'uncertain'},box=element('div',undefined,'panel');box.append(element('div',`${p.index} 帧 · ${p.timestamp.toFixed(2)} s · g=${p.opening.toFixed(4)}`),element('p',names[s.state],s.state));const a=element('a');a.href=p.path;a.target='_blank';a.rel='noopener';const img=element('img');img.src=p.path;img.alt=`${r.task} 第 ${p.index} 帧 ${names[s.state]}`;img.loading='lazy';a.append(img);box.append(a,element('div',`${r.side==='left'?'左腕':'右腕'} · ${p.camera?.endsWith('_left')?'左目':'右目'}`,'muted'));return box}));
const profile=data.profiles.find(t=>t.task===r.task);$('profile').textContent=JSON.stringify(profile?.sides?.[r.side]??profile,null,2);$('record').textContent=JSON.stringify({...r,trace:undefined,panels:r.panels.map(p=>({...p,path:undefined}))},null,2);draw(r,states)}
function draw(r,states){const svg=$('chart'),ns='http://www.w3.org/2000/svg';svg.replaceChildren();function add(tag,attrs,text){const el=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text!==undefined)el.textContent=text;svg.append(el);return el}const ts=r.trace.timestamps,gs=r.trace.openings,lo=ts[0],span=Math.max(.1,ts.at(-1)-lo),x=i=>45+930*(ts[i]-lo)/span,y=g=>180-150*g;
for(const v of [0,.5,1]){add('line',{x1:45,x2:975,y1:y(v),y2:y(v),stroke:'#e4e8ed'});add('text',{x:8,y:y(v)+5,fill:'#526174','font-size':13},String(v))}
add('rect',{x:x(r.plateau_start),y:25,width:Math.max(1,x(r.plateau_end)-x(r.plateau_start)),height:155,fill:'#e1efff'});add('polyline',{points:gs.map((g,i)=>`${x(i)},${y(g)}`).join(' '),fill:'none',stroke:'#697789','stroke-width':2});for(const [i,s]of states){add('circle',{cx:x(i),cy:y(gs[i]),r:5,fill:{held:'#087f49',free:'#175eb0',uncertain:'#c08618'}[s.state]})}
for(const i of new Set([0,r.plateau_start,r.plateau_end,ts.length-1]))add('text',{x:x(i),y:208,'text-anchor':'middle',fill:'#526174','font-size':12},`${ts[i].toFixed(1)}s`)}
fetch('data.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(d=>{data=d;const tasks=[...new Set(d.records.map(r=>r.task))].sort((a,b)=>a.localeCompare(b,'zh'));$('task').append(...tasks.map(t=>option(t,t)));$('summary').textContent=`${tasks.length} 个任务 · ${d.records.length} 个代表平台证据。数值模式外的稳定尾部仍按覆盖清单记录未复核。`;for(const id of ['search','task','role','semantic'])$(id).addEventListener(id==='search'?'input':'change',filter);$('case').addEventListener('change',()=>{position=Number($('case').value);show()});$('previous').onclick=()=>{position--;show()};$('next').onclick=()=>{position++;show()};if(tasks.includes('插入缝隙刷'))$('task').value='插入缝隙刷';filter()}).catch(e=>{$('summary').textContent='证据加载失败：'+e.message});
</script></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.reviews, args.coverage, args.profiles, args.output), ensure_ascii=False))
