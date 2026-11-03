"""Portable review page, synchronized evidence frames, and static plot export."""
import html
import json
import math
from pathlib import Path
import re

from .contracts import SIDE_CAMERAS


def _report_videos(episode, record):
    """Only expose same-side, metadata-aligned videos through local media URLs."""
    videos, seen = [], set()
    for value in record.get("videos", []):
        side, camera = value.get("side"), value.get("camera")
        if side not in episode.get("valid_sides", []) or side not in SIDE_CAMERAS:
            raise ValueError("review video requires an effective gripper side")
        names = {name for short in SIDE_CAMERAS[side] for name in (short, "observation.images." + short)}
        if camera not in names or camera in seen:
            raise ValueError("review video camera must match its side and be unique")
        spec = episode.get("cameras", {}).get(camera, {})
        if not spec or spec.get("valid", True) is not True:
            raise ValueError("review video camera is unavailable")
        url = value.get("url")
        if not isinstance(url, str) or not re.fullmatch(r"/media/[a-zA-Z0-9_-]+\.mp4", url):
            raise ValueError("review videos require a local media URL")
        for key in ("from_timestamp", "to_timestamp"):
            number, original = value.get(key), spec.get(key)
            if (isinstance(number, bool) or not isinstance(number, (int, float))
                    or not math.isfinite(number) or not isinstance(original, (int, float))
                    or not math.isclose(number, original, rel_tol=0., abs_tol=1e-6)):
                raise ValueError("review video timing differs from episode metadata")
        start, end = value["from_timestamp"], value["to_timestamp"]
        if start < 0 or end <= start or start + episode["timestamps"][-1] >= end + 1e-5:
            raise ValueError("review video does not cover the episode")
        videos.append(dict(side=side, camera=camera, url=url, from_timestamp=start, to_timestamp=end))
        seen.add(camera)
    return videos


def write_report(run_dir, records):
    run_dir = Path(run_dir)
    full_cycle = bool(records) and all(r["result"].get("workflow") == "full_cycle" for r in records)
    data = []
    for record in records:
        episode, result = record["episode"], record["result"]
        frames = []
        for frame in record.get("frames", []):
            f = dict(frame)
            f["path"] = Path(f["path"]).resolve().relative_to(run_dir.resolve()).as_posix()
            frames.append(f)
        data.append(dict(id=episode["episode_id"], t=episode["timestamps"],
                         original=[[r[9], r[19]] for r in episode["state"]],
                         target=result["target"], preview=result.get("preview_target", result["target"]),
                         valid=result["valid"], events=result["events"],
                         candidates=record["candidates"], frames=frames,
                         videos=_report_videos(episode, record), valid_sides=episode.get("valid_sides", []),
                         identity=episode["identity"], issues=result.get("pipeline_issues", []),
                         comparisons=result.get("release_comparisons", []),
                         closing_comparisons=result.get("closing_comparisons", []),
                         workflow=result.get("workflow", "opening_review"),
                         phase=result.get("phase", []), source=result.get("source", []),
                         coverage=result.get("coverage", {}),
                         closing_policy=result.get("config", {}).get("closing_policy", "fixed_timing")))
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    page = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>夹爪开度曲线审阅</title><style>
body{font:16px system-ui;margin:24px auto;max-width:1200px;padding:0 20px;background:#f4f7fa;color:#172b40}
h1{font-size:26px}select,input{font:inherit}section{background:white;padding:18px;margin:18px 0;border-radius:12px}
svg{width:100%;height:auto}pre{white-space:pre-wrap;font-size:12px;max-height:360px;overflow:auto}
.legend span{margin-right:22px}.frames{display:flex;gap:14px;flex-wrap:wrap}.frame{width:45%;min-width:250px}.frame img{width:100%}
.note{padding:12px;background:#fff4d5;border-left:4px solid #be7a13}button{margin-left:8px;font:inherit;cursor:pointer}
.review-grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(290px,1fr);gap:20px;align-items:start}
#charts svg text{font-size:21px}#charts svg text:first-child{font-size:24px;font-weight:600}
.video-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}.video-card{min-width:0;background:#f4f7fa;border-radius:8px;padding:10px}
.video-card video{display:block;width:100%;max-height:380px;background:#101820;cursor:pointer;object-fit:contain;border-radius:6px}
.video-card header{display:flex;align-items:center;justify-content:space-between;gap:6px;margin-bottom:8px}.video-card select{max-width:65%;font-size:14px}
.video-status{font-size:13px;color:#5b6977;min-height:1.2em}.video-error{color:#b12c26}.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.controls button{margin-left:0}
#identity{overflow-wrap:anywhere;font-size:13px}#scrub{width:100%;margin:12px 0;cursor:pointer}#time{font-size:14px;line-height:1.7}
details{background:white;padding:18px;margin:18px 0;border-radius:12px}summary{cursor:pointer;font-weight:600}h2{font-size:18px}
@media(max-width:850px){.review-grid{grid-template-columns:1fr}.video-grid{grid-template-columns:repeat(auto-fit,minmax(230px,1fr))}}
</style><h1>双侧夹爪 · 开度指令与视觉证据审阅</h1>
<p>纵轴为归一化夹爪开度（0～1）：0 = 全闭，1 = 全开。灰线是原始实测开度；目标开度为0表示发出全闭指令，持物时实测开度仍可大于0。</p>
<p class="note">__MODE_NOTE__</p>
<label>Episode <select id="episodes"></select></label><section><div id="identity"></div>
<p class="legend"><span style="color:#526577">━ 原始实测开度</span><span style="color:#e59420">┄ 候选目标开度</span><span id="closingLegend" style="color:#668cba">┈ 闭合对照方案</span><span style="color:#12826c">━ 有效目标开度</span></p>
<div class="review-grid"><div><div id="charts"></div><input id="scrub" aria-label="Episode时间定位" type="range" min="0" value="0" step="1">
<div class="controls"><button id="play">播放</button><button id="previous">上一帧</button><button id="next">下一帧</button><label>速度 <select id="rate"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label></div><p id="time"></p></div>
<div><h2>对应腕部视频</h2><p class="video-status">点击视频或播放按钮开始；拖动时间条、点击曲线可同步定位。</p><div class="video-grid" id="videos"></div><p id="videoMessage" class="video-status" role="status"></p></div></div></section>
<details><summary>模型使用的证据帧</summary><p>以下为模型判断时的抽帧；上方视频可查看连续过程。</p><div class="frames" id="frames"></div></details>
<section id="coverageSection"><h2>完整周期与检查覆盖</h2><pre id="coverage"></pre></section>
<section><h2>闭合拟合</h2><div id="closing"></div></section>
<section><h2>释放策略比较</h2><div id="release"></div></section>
<section><h2>事件、平台、速度与回退原因</h2><pre id="events"></pre></section>
<script>const records=__DATA__;let r, timer; const $=id=>document.getElementById(id);
let playing=false,currentEpisodeTime=0,videoEntries=[],master=null,displayedIndex=-1,mediaVersion=0;
let fallbackStart=0,fallbackTime=0;const selectedCameras={};
records.forEach((v,i)=>{let o=document.createElement('option');o.value=i;o.textContent=v.id;$('episodes').append(o)});
function line(values,col,accepted=false){let active=false;return values.map((v,i)=>{if(accepted&&!r.valid[i][col]){active=false;return ''}let op=active?'L':'M';active=true;return op+(55+930*(r.t[i]-r.t[0])/(r.t.at(-1)-r.t[0]||1)).toFixed(2)+','+(190-v[col]*160).toFixed(2)}).join(' ')}
function comparisonValues(c){return r.closing_policy==='original_speed'?c.fixed_timing:c.original_speed}
function comparisonLine(c){return comparisonValues(c).slice(c.start,c.display_end+1).map((v,j)=>(j?'L':'M')+(55+930*(r.t[c.start+j]-r.t[0])/(r.t.at(-1)-r.t[0]||1)).toFixed(2)+','+(190-v*160).toFixed(2)).join(' ')}
function render(){stopPlayback();r=records[+$('episodes').value];currentEpisodeTime=r.t[0];displayedIndex=-1;$('identity').textContent=JSON.stringify(r.identity);$('scrub').max=r.t.length-1;$('scrub').value=0;
$('coverageSection').hidden=r.workflow!=='full_cycle';$('coverage').textContent=JSON.stringify(r.coverage,null,2);
$('charts').innerHTML=[0,1].map(col=>`<svg viewBox="0 0 1020 230"><text x="5" y="20">${col?'右':'左'}夹爪开度（0～1）</text>${[0,.25,.5,.75,1].map(v=>`<path d="M55 ${190-v*160}H985" stroke="#e6ebef"/><text x="12" y="${194-v*160}" font-size="12">${v}</text>`).join('')}<path d="${line(r.original,col)}" stroke="#526577" fill="none" stroke-width="2"/><path d="${line(r.preview,col)}" stroke="#e59420" fill="none" stroke-width="3" stroke-dasharray="6 4"/><path d="${line(r.target,col,true)}" stroke="#12826c" fill="none" stroke-width="1.5"/><line class="cursor" y1="25" y2="195" stroke="#9b3065"/><text x="55" y="218" font-size="12">${r.t[0]} s</text><text x="930" y="218" font-size="12">${r.t.at(-1)} s</text></svg>`).join('');
$('closingLegend').textContent=r.closing_policy==='original_speed'?'┈ 固定时序闭合拟合对照':'┈ 原速闭合延伸对照';
$('charts').querySelectorAll('svg').forEach((svg,col)=>r.closing_comparisons.filter(c=>c.side===(col?'right':'left')&&comparisonValues(c)).forEach(c=>{let p=document.createElementNS('http://www.w3.org/2000/svg','path');p.setAttribute('d',comparisonLine(c));p.setAttribute('stroke','#668cba');p.setAttribute('stroke-dasharray','2 4');p.setAttribute('fill','none');p.setAttribute('stroke-width','2');svg.append(p)}));
$('closing').textContent=`当前闭合策略：${r.closing_policy==='original_speed'?'按原速延伸':'固定起止时刻的端点拟合'}。\n`+(r.closing_comparisons.length?r.closing_comparisons.map(c=>`${c.side} · 原闭合帧 ${c.start}–${c.end}：端点拟合在原结束时刻到0，改变闭合速度；原速延伸需额外 ${c.extra_time_needed_s.toFixed(3)} s。${c.original_speed?'':'原速方案不能在确认持物范围内到0，未绘制。'}`).join('\n'):'无可拟合的非零平台闭合段。');
$('events').textContent=JSON.stringify({pipeline_issues:r.issues,candidates:r.candidates,events:r.events,closing_comparisons:r.closing_comparisons,release_comparisons:r.comparisons},null,2);
$('release').textContent=r.comparisons.length?r.comparisons.map(c=>`${c.side} · 帧 ${c.start}–${c.end}：橙线保留释放起止时刻，改变张开速度；按原估计速度 ${(c.opening_speed_mm_s/95).toFixed(3)} 开度/秒从0开始，额外需要 ${c.extra_time_needed_s.toFixed(3)} s。${c.original_speed?'原速候选也能在确认窗口内完成。':'确认窗口不足，原速方案未延伸到窗口外。'} 两种方案均未批准。`).join('\n'):'本片段无可比较的完整释放候选。';
if(r.workflow==='full_cycle'){$('closingLegend').hidden=true;$('closing').textContent='完整周期按抓前准备、闭合、持物和释放分别生成；数值边界与视觉持物范围分开核对。绿色目标经过当前算法校验，来源在逐帧读数中显示。';$('release').textContent='正常释放采用固定起止时刻的连续映射，校验通过可生成目标。模型来源仍为 model；未覆盖、遮挡和冲突区域保留原值。'}else{$('closingLegend').hidden=false}
$('charts').querySelectorAll('svg').forEach(svg=>{svg.style.cursor='crosshair';svg.onclick=event=>{const box=svg.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,((event.clientX-box.left)*1020/box.width-55)/930));seekTo(r.t[0]+fraction*(r.t.at(-1)-r.t[0]))}});
buildVideos();seekTo(r.t[0]);}
function frameAt(t){let lo=0,hi=r.t.length-1;while(lo<hi){let mid=Math.floor((lo+hi)/2);if(r.t[mid]<t)lo=mid+1;else hi=mid}return lo>0&&Math.abs(r.t[lo-1]-t)<=Math.abs(r.t[lo]-t)?lo-1:lo}
function displayTime(t,force=false){currentEpisodeTime=Math.max(r.t[0],Math.min(r.t.at(-1),t));t=currentEpisodeTime;let i=frameAt(t),x=55+930*(t-r.t[0])/(r.t.at(-1)-r.t[0]||1);$('scrub').value=i;document.querySelectorAll('.cursor').forEach(el=>{el.setAttribute('x1',x);el.setAttribute('x2',x)});
$('time').textContent=`帧 ${i} · episode ${t.toFixed(2)} s · 左/右实测开度 ${r.original[i].map(x=>x.toFixed(3)).join(' / ')} · 候选目标开度 ${r.preview[i].map(x=>x.toFixed(3)).join(' / ')} · 有效目标开度 ${r.target[i].map((x,col)=>r.valid[i][col]?x.toFixed(3):'未确认').join(' / ')}`;
if(r.phase.length)$('time').textContent+=` · 阶段 ${r.phase[i].join(' / ')} · 来源 ${r.source[i].join(' / ')}`;
if(!force&&displayedIndex===i)return;displayedIndex=i;
$('frames').replaceChildren();[...new Set(r.frames.map(f=>f.camera))].forEach(camera=>{let fs=r.frames.filter(f=>f.camera===camera),f=fs.reduce((a,b)=>Math.abs(b.timestamp-t)<Math.abs(a.timestamp-t)?b:a),d=document.createElement('div');d.className='frame';let p=document.createElement('p');p.textContent=`${camera} · 证据帧 ${f.frame_index} · episode ${f.timestamp.toFixed(2)}s · 原视频 ${f.video_timestamp.toFixed(2)}s`;d.append(p);let img=document.createElement('img');img.src=f.path;img.alt=`${camera} 证据帧 ${f.frame_index}`;d.append(img);$('frames').append(d)});
if(!r.frames.length)$('frames').textContent='此示例无视觉证据；请查看来源标记。'}
function setVideoTime(entry,t,force=false){if(entry.failed||entry.video.readyState<1)return;const target=entry.spec.from_timestamp+t;if(force||Math.abs(entry.video.currentTime-target)>.08){try{entry.video.currentTime=target}catch(error){entry.status.textContent='视频正在加载，稍后定位。'}}}
function syncVideos(t,force=false,followersOnly=false){videoEntries.forEach(entry=>{if(!followersOnly||entry!==master)setVideoTime(entry,t,force)})}
function seekTo(t){displayTime(t,true);syncVideos(currentEpisodeTime,true);fallbackStart=performance.now();fallbackTime=currentEpisodeTime;}
function stopPlayback(){playing=false;if(timer!==undefined&&timer!==null)cancelAnimationFrame(timer);timer=null;videoEntries.forEach(entry=>entry.video.pause());$('play').textContent='播放';}
function playEntry(entry){const version=mediaVersion;entry.video.playbackRate=+$('rate').value;const promise=entry.video.play();if(promise)promise.catch(error=>{if(version!==mediaVersion||!playing||error.name==='AbortError')return;entry.status.textContent='视频无法播放，请检查视频连接后重试。';entry.status.classList.add('video-error');entry.failed=true;if(entry===master){master=videoEntries.find(item=>!item.failed)||null;if(!master){stopPlayback();$('videoMessage').textContent='视频播放失败；下方保留模型证据帧。'}}})}
function startPlayback(){if(playing||r.t.at(-1)<=r.t[0])return;if(currentEpisodeTime>=r.t.at(-1)-.001)seekTo(r.t[0]);master=videoEntries.find(entry=>!entry.failed)||null;if(videoEntries.length&&!master){$('videoMessage').textContent='没有可播放的视频，请重新选择episode后重试。';return}playing=true;$('play').textContent='暂停';fallbackStart=performance.now();fallbackTime=currentEpisodeTime;syncVideos(currentEpisodeTime,true);videoEntries.filter(entry=>!entry.failed).forEach(playEntry);timer=requestAnimationFrame(tick);}
function togglePlayback(){if(playing)stopPlayback();else startPlayback();}
function tick(now){if(!playing)return;let t=currentEpisodeTime;if(master){if(master.video.readyState<2||master.video.seeking){timer=requestAnimationFrame(tick);return}t=master.video.currentTime-master.spec.from_timestamp;if(t<r.t[0]-.01){syncVideos(r.t[0],true);timer=requestAnimationFrame(tick);return}}else{t=fallbackTime+(now-fallbackStart)/1000*+$('rate').value}displayTime(t);if(t>=r.t.at(-1)-.001){stopPlayback();seekTo(r.t.at(-1));return}syncVideos(t,false,true);timer=requestAnimationFrame(tick);}
function buildVideos(){mediaVersion++;const version=mediaVersion;videoEntries.forEach(entry=>{entry.video.pause();entry.video.removeAttribute('src');entry.video.load()});videoEntries=[];master=null;$('videos').replaceChildren();$('videoMessage').textContent='';
for(const side of ['left','right']){const specs=(r.videos||[]).filter(value=>value.side===side);if(!specs.length)continue;const spec=specs.find(value=>value.camera===selectedCameras[side])||specs[0];selectedCameras[side]=spec.camera;
const card=document.createElement('div');card.className='video-card';const header=document.createElement('header'),title=document.createElement('strong');title.textContent=side==='left'?'左腕':'右腕';header.append(title);
const select=document.createElement('select');select.className='camera-select';select.dataset.side=side;select.setAttribute('aria-label',`${title.textContent}视角`);specs.forEach(value=>{const option=document.createElement('option');option.value=value.camera;option.textContent=value.camera.endsWith('_left')?'双目左视角':'双目右视角';option.selected=value.camera===spec.camera;select.append(option)});header.append(select);card.append(header);
const video=document.createElement('video');video.muted=true;video.defaultMuted=true;video.playsInline=true;video.preload='metadata';video.tabIndex=0;video.dataset.side=side;video.dataset.camera=spec.camera;video.dataset.fromTimestamp=spec.from_timestamp;video.dataset.episodeStart=spec.from_timestamp+r.t[0];video.dataset.episodeEnd=spec.from_timestamp+r.t.at(-1);video.setAttribute('aria-label',`${title.textContent}视频，点击播放或暂停`);
const status=document.createElement('p');status.className='video-status';status.setAttribute('role','status');status.textContent='正在读取视频…';const entry={video,spec,status,failed:false};videoEntries.push(entry);
video.addEventListener('loadedmetadata',()=>{if(version!==mediaVersion)return;status.textContent='与开度曲线同步';setVideoTime(entry,currentEpisodeTime,true)});
video.addEventListener('error',()=>{if(version!==mediaVersion)return;entry.failed=true;status.textContent='视频加载失败，请检查连接；模型证据帧仍可查看。';status.classList.add('video-error');if(entry===master){master=videoEntries.find(value=>!value.failed)||null;if(!master)stopPlayback()}});
video.addEventListener('ended',()=>{if(version===mediaVersion&&entry===master&&playing){stopPlayback();seekTo(r.t.at(-1))}});
video.onclick=togglePlayback;video.onkeydown=event=>{if(event.code==='Space'){event.preventDefault();togglePlayback()}};video.src=spec.url;card.append(video,status);
const fullscreen=document.createElement('button');fullscreen.textContent='全屏';fullscreen.onclick=()=>{if(video.requestFullscreen)video.requestFullscreen().catch(()=>{status.textContent='当前浏览器不支持此全屏方式。'})};card.append(fullscreen);
select.onchange=()=>{const resume=playing;stopPlayback();selectedCameras[side]=select.value;buildVideos();seekTo(currentEpisodeTime);if(resume)startPlayback()};$('videos').append(card);}
if(!videoEntries.length)$('videoMessage').textContent='本报告尚未关联可播放视频；可查看下方模型证据帧。';}
$('episodes').onchange=render;$('scrub').oninput=()=>seekTo(r.t[+$('scrub').value]);$('play').onclick=togglePlayback;
for(const [id,delta] of [['previous',-1],['next',1]])$(id).onclick=()=>{stopPlayback();seekTo(r.t[Math.max(0,Math.min(r.t.length-1,+$('scrub').value+delta))])};
$('rate').onchange=()=>{videoEntries.forEach(entry=>entry.video.playbackRate=+$('rate').value);fallbackStart=performance.now();fallbackTime=currentEpisodeTime};if(records.length)render();</script></html>'''
    note = ("绿色为通过视觉证据与数值规则校验的目标，可含 model 自动生成来源；不等于人工真值。橙色为候选。覆盖统计区分已检查、拒绝和未检查；合成示例不作为训练标签。"
            if full_cycle else "橙色为待复核候选；绿色仅为通过当前契约的目标。合成示例只验证数学行为。此旧版张开事件流程的释放策略尚未批准，含释放事件不进入有效训练标签。真实模型判断不等于人工参考。")
    (run_dir / "review.html").write_text(page.replace("__MODE_NOTE__", note).replace("__DATA__", encoded), encoding="utf-8")
    return run_dir / "review.html"


def write_plot(run_dir, records):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    count = len(records)
    if not count:
        return None
    fig, axes = plt.subplots(count, 2, figsize=(13, 3 * count), squeeze=False)
    for row, record in enumerate(records):
        ep, result = record["episode"], record["result"]
        index = ep.get("identity", {}).get("episode_index")
        label = f"Episode {index}" if index is not None else str(ep["episode_id"]).rsplit("/", 1)[-1]
        for col, side in enumerate(("left", "right")):
            ax = axes[row, col]
            ax.plot(ep["timestamps"], [s[(9,19)[col]] for s in ep["state"]], color="#526577", lw=2, label="Measured opening")
            ax.plot(ep["timestamps"], [s[col] for s in result.get("preview_target", result["target"])], color="#e59420", ls="--", lw=2, label="Candidate command")
            ax.plot(ep["timestamps"], [s[col] if result["valid"][i][col] else float("nan") for i,s in enumerate(result["target"])], color="#12826c", lw=1, label="Accepted command (valid only)")
            ax.set(title=f'{label} / {side}', xlabel="Episode time (s)", ylabel="Gripper opening (0-1)", ylim=(-.03, 1.05), yticks=[0, .25, .5, .75, 1])
            ax.grid(alpha=.2)
            for comp in result.get("closing_comparisons", []):
                old_policy = result.get("config", {}).get("closing_policy") == "original_speed"
                alternative = comp["fixed_timing"] if old_policy else comp["original_speed"]
                if comp["side"] == side and alternative is not None:
                    start, end = comp["start"], comp["display_end"] + 1
                    ax.plot(ep["timestamps"][start:end], alternative[start:end], color="#668cba", ls=":", lw=1.8, label="Fixed-timing closure (comparison)" if old_policy else "Original-speed closure (comparison)")
            comparisons = [c for c in result.get("release_comparisons", []) if c["side"] == side]
            for comp in comparisons:
                if comp["original_speed"] is not None:
                    ax.plot(ep["timestamps"], comp["original_speed"], color="#7755aa", ls=":", label="Original release speed")
                ax.text(.02, .95, f'Original release speed needs +{comp["extra_time_needed_s"]:.2f}s', transform=ax.transAxes, fontsize=7, va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
            ax.legend(fontsize=7, loc="upper right")
    full_cycle = all(r["result"].get("workflow") == "full_cycle" for r in records)
    subtitle = "Automatic targets retain model provenance; not human ground truth" if full_cycle else "Review candidates are not approved training labels"
    fig.suptitle("Gripper opening commands: 0 = closed, 1 = open\n" + subtitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .98))
    path = Path(run_dir) / "curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
