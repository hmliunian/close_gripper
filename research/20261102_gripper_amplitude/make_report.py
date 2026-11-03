"""Create a standalone Plotly report from measured local artifacts."""
import collections
import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT=Path(__file__).resolve().parent
ORDER=['拿起杯子','拿起粉色水杯','拿起盐罐子','拿起洗洁剂','拿起耗油','拿辣椒油','拿起玩偶兔','拿起星星挂件','叠衣服']
LABELS={
    'raw_follower':'原始副臂', 'task_affine':'任务统一端点映射',
    'episode_affine':'每段端点映射', 'episode_affine_shift':'每段端点 + 相位对齐',
    'event_template':'任务端点 + 经验模板',
    'episode_open_affine_shift':'自适应张开幅度 + 端点映射 + 相位对齐',
    'episode_open_template':'自适应张开幅度 + 经验模板',
}


def main():
    episodes=json.loads((OUT/'accepted_episodes.json').read_text())
    ranges=json.loads((OUT/'accepted_ranges.json').read_text())
    results=json.loads((OUT/'mapping_metrics.json').read_text())
    by_task={r['task']:r for r in ranges}
    fmt=lambda x:f"{x['p50']:.2f} [{x['p05']:.2f}, {x['p95']:.2f}]"
    rows=[]
    for task in ORDER:
        r=by_task[task]
        cells=[task,str(r['usable'])]+[fmt(r[side][stat]) for side in ('leader','follower') for stat in ('max','min')]
        rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>')
    table='<table><thead><tr>'+''.join('<th>'+c+'</th>' for c in ['任务','样本数','主臂最大','主臂最小','副臂最大','副臂最小'])+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    boxes=make_subplots(rows=2,cols=1,subplot_titles=['每条 episode 的最大开度','每条 episode 的最小开度'])
    for row,stat in enumerate(('max','min'),1):
        for side,label,color in [('leader','主臂指令','#2764b1'),('follower','副臂实测','#d77723')]:
            boxes.add_trace(go.Box(x=[e['task'] for e in episodes],y=[e[side+'_mm'][stat] for e in episodes],
                                  customdata=[e['source_id'] for e in episodes],name=label,legendgroup=side,showlegend=row==1,
                                  marker_color=color,boxpoints='all',jitter=.25,pointpos=0,
                                  hovertemplate='%{x}<br>%{y:.2f} mm<br>%{customdata}<extra>'+label+'</extra>'),row=row,col=1)
    boxes.update_xaxes(categoryorder='array',categoryarray=ORDER)
    boxes.update_yaxes(title_text='开口宽度 / mm',range=[-3,100])
    boxes.update_layout(boxmode='group',height=800,template='plotly_white',margin=dict(t=55,b=80),font=dict(family='sans-serif'))

    comparison=go.Figure()
    for method in ('raw_follower','episode_affine','episode_open_affine_shift','episode_open_template'):
        comparison.add_bar(name=LABELS[method],x=ORDER[:-1],y=[results['by_task'][task][method]['mae_mm']['mean'] for task in ORDER[:-1]])
    comparison.update_layout(title='留出验证：每段 MAE 先求均值，再在任务内平均',yaxis_title='相对真实主臂的误差 / mm',
                             template='plotly_white',barmode='group',height=450,legend=dict(orientation='h',y=-.3))

    gallery=go.Figure();buttons=[]
    test_rows=sorted(results['rows'],key=lambda r:(ORDER.index(r['task']),r['source_id']))
    for i,r in enumerate(test_rows):
        a=np.load(OUT/'predictions'/(r['source_id']+'.npz'))
        series=[('leader_raw','主臂真值','#2764b1','solid'),('raw_follower','副臂实测','#d77723','solid'),
                ('episode_open_affine_shift','幅度映射 + 相位对齐','#218c57','solid'),('episode_open_template','经验模板','#914eaa','dash')]
        for key,label,color,dash in series:
            gallery.add_trace(go.Scatter(x=a['leader_time'] if key=='leader_raw' else a['time'],y=a[key]*1000,
                                         mode='lines',name=label,line=dict(color=color,dash=dash,width=2),visible=i==0,
                                         hovertemplate='%{x:.3f} s<br>%{y:.2f} mm<extra>'+label+'</extra>'))
        visible=[False]*(4*len(test_rows));visible[4*i:4*i+4]=[True]*4
        title=f"{r['task']} · {r['source_id'][:8]} · 映射 MAE {r['metrics']['episode_open_affine_shift']['mae_mm']:.2f} mm"
        buttons.append(dict(label=r['task']+' '+r['source_id'][:8],method='update',args=[dict(visible=visible),dict(title=title)]))
    gallery.update_layout(title=buttons[0]['args'][1]['title'],template='plotly_white',height=520,
                          updatemenus=[dict(buttons=buttons,direction='down',x=0,y=1.2,xanchor='left',yanchor='top')],
                          xaxis_title='共同 header 时间轴 / s',yaxis_title='开口宽度 / mm',yaxis_range=[-3,100],
                          legend=dict(orientation='h',y=-.2),margin=dict(t=120))
    metrics_table='<table><thead><tr><th>方法</th><th>平均 MAE / mm</th><th>每段 MAE 的 P90 / mm</th><th>过渡附近 MAE / mm</th></tr></thead><tbody>'
    for method,label in LABELS.items():
        m=results['aggregate'][method]
        metrics_table+=f"<tr><td>{label}</td><td>{m['mae_mm']['mean']:.3f}</td><td>{m['mae_mm']['p90']:.3f}</td><td>{m['transition_mae_mm']['mean']:.3f}</td></tr>"
    metrics_table+='</tbody></table>'
    paragraphs=[
        '本次重点是开闭范围和主副臂幅度关系。单位采用现有数据合同中的开口宽度（米转毫米）；没有夹爪机构的角度标定，不能把这些值称为角度。右臂 joint_command 是指令值，joint_state 是实际宽度。',
        '先调查 280 条分层原始样本，再按已有 Accepted 标签补足每个抓取任务 32 条、叠衣服 24 条，最终主表共 280 条 Accepted；两轮合计读取 322 个不同 MCAP。文件选择由固定哈希决定，原始文件只读。所有数值先按 header 时间排序、重复时间戳保留末条；最大/最小是每段实际记录的极值，不等于硬件机械极限。',
        '表中为中位数 [P5, P95]，分位数在 episode 的极值之间计算。样本覆盖 2026-10-26 这批数据，不是所有日期或三个发布版的总体估计。主臂常见最大值 95 mm；星星挂件最大开度中位数只有 39.95 mm。副臂受物体阻挡后的最小值按物体与抓法变化。',
    ]
    method_text='''幅度映射：r(t)=clip((s(t)−S_closed)/(S_open−S_closed),0,1)，再映射到 M_closed+(M_open−M_closed)·r(t)。研究原型用副臂每段 P5/P95 估计两个端点，只接收从开到闭且无反复的序列。主臂闭合端和开口微小偏置由校准集估计，主臂张开幅度结合本段副臂张开值自适应估计。相位差也仅从校准集拟合；这是形状对齐偏移，不应解释成统一的机械响应延迟。'''
    goal_text='''文档目标是从副臂生成规范化指令：抓前 1、抓后 0、开闭过程参考主臂。若追求真实主臂幅度，使用实测/估计的 M_open、M_closed；若执行文档目标，目标端点就是 1、0。这两个目标不同，不能用复制真实主臂的误差直接代表规范化标签质量。'''
    limits='''验证从 8 个抓取任务各 16 条校准候选和 16 条留出候选开始。119 条校准样本满足输入形态，留出数据有 116 条满足副臂单次闭合条件；12 条初始不明/多次开合/结尾未闭的留出样本未进行本轮映射。未根据留出主臂曲线筛掉难例。结果属于同日期、同任务和设备上的探索性留出比较，没有验证跨任务泛化、无抓取保护、松手或叠衣服多次抓放。'''
    salt_mae=results['by_task']['拿起盐罐子']['episode_open_affine_shift']['mae_mm']['mean']
    other_mae=[v['episode_open_affine_shift']['mae_mm']['mean'] for k,v in results['by_task'].items() if k!='拿起盐罐子']
    ambiguity=f'''明确反例：盐罐子 c8ed15cc 的副臂末尾平台约 70.09 mm，主臂为 0；8e68d671 的副臂平台约 70.50 mm，主臂为 53.30 mm（均取末 0.3 s 中位数）。因此只凭受阻后的副臂平台，无法唯一还原主臂实际闭合端。盐罐子留出 MAE 仍为 {salt_mae:.2f} mm，而其他任务约 {min(other_mae):.2f}–{max(other_mae):.2f} mm。'''
    content='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>夹爪范围与幅度映射调查</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#202530;max-width:1400px;margin:35px auto;padding:0 25px;background:#fafbfd}p{max-width:1150px}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}th,td{border:1px solid #dce1e8;padding:9px;text-align:left}th{background:#eef2f7}h1{font-size:26px}h2{font-size:21px;margin-top:38px}.note{padding:15px;background:#eef6f0;border-left:4px solid #218c57}a{color:#2764b1}</style><body><h1>夹爪范围与幅度映射调查 · 2026-11-02</h1>'
    content+=''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs)+table
    content+=boxes.to_html(full_html=False,include_plotlyjs=True)
    content+='<h2>幅度对应方法与验证</h2><p>'+html.escape(method_text)+'</p><p class="note">'+html.escape(goal_text)+'</p><p>'+html.escape(limits)+'</p>'+metrics_table
    content+=comparison.to_html(full_html=False,include_plotlyjs=False)
    content+='<p>'+html.escape(ambiguity)+'</p><p>经验模板在这批样本上的平均误差没有优于保留副臂形状的幅度映射；因此当前应先以端点映射和相位校准为基线。对多次抓放，应按每次事件重新估计端点，不能把整段最小最大值当成每次抓取的端点。</p>'
    content+='<h2>全部 116 条留出曲线</h2><p>下拉框可查看每一条参与评分的记录，包括失败案例。绿色曲线完全由副臂及校准参数生成，蓝色主臂只用于对照。</p>'+gallery.to_html(full_html=False,include_plotlyjs=False)
    content+='<p>证据：<a href="accepted_episode_ranges.csv">每段范围 CSV</a> · <a href="accepted_ranges.json">分组范围 JSON</a> · <a href="mapping_metrics.json">全部方法与逐段误差</a> · <a href="accepted_manifest.json">采样清单</a></p></body></html>'
    (OUT/'amplitude_report.html').write_text(content)
    lines=['夹爪范围与幅度映射调查（2026-11-02）','',*paragraphs,'','主表单位 mm；每列为 episode 极值的中位数 [P5,P95]。','任务 | n | 主臂最大 | 主臂最小 | 副臂最大 | 副臂最小']
    for task in ORDER:
        r=by_task[task];lines.append(' | '.join([task,str(r['usable'])]+[fmt(r[side][stat]) for side in ('leader','follower') for stat in ('max','min')]))
    lines+=['',method_text,'',goal_text,'',limits,'','探索性留出比较（116 条，先逐 episode 求 MAE，再平均）：']
    for method,label in LABELS.items():
        m=results['aggregate'][method];lines.append(f"{label}：{m['mae_mm']['mean']:.3f} mm；P90={m['mae_mm']['p90']:.3f} mm；过渡附近={m['transition_mae_mm']['mean']:.3f} mm。")
    lines+=['',ambiguity,'','额外发现：第一轮样本中，玩偶兔 25/32 条主从消息计数比异常，全部为 Denied；补取 Accepted 后，32/32 条主臂最小值≤1 mm。必须区分录制异常与真实的非零闭合。杯子、盐罐子 Accepted 样本仍有主臂非零闭合，这不是同一种现象。',
            '','下一步应继续核实各次抓放的平台范围、无事件负例、松手与多次抓放。当前不进行发布或训练 loader 改动。',
            '','复现：用 orion-data-pipeline 的 .venv/bin/python -B 运行 survey.py --accepted-only 和 evaluate_mapping.py；用 data_citdal_v3 的 .venv/bin/python -B 运行 make_report.py。','源数据不改写，所有输出限当前 research 子目录。']
    (OUT/'findings.txt').write_text('\n'.join(lines)+'\n')
    print('wrote',OUT/'amplitude_report.html')


if __name__=='__main__':main()
