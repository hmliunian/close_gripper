"""Held-out comparison on Accepted, follower-detected single-close episodes.

Test inference reads only the follower sequence and a task-level calibration.
Reference leader traces are used for calibration and scoring, never prediction.
This experiment does not evaluate no-grasp detection or repeated grasp cycles.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

OUT=Path(__file__).resolve().parent
METHODS=('raw_follower','task_affine','episode_affine','episode_affine_shift','event_template',
         'episode_open_affine_shift','episode_open_template')
OFFSETS=np.arange(-.8,.801,.005)


def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return {'n':0}
    return dict(n=len(a),mean=float(np.mean(a)),p50=float(np.median(a)),
                p90=float(np.quantile(a,.9)),max=float(np.max(a)))


def nearest(times,values,grid):
    right=np.searchsorted(times,grid).clip(0,len(times)-1)
    left=(right-1).clip(0,len(times)-1)
    ix=np.where(np.abs(times[left]-grid)<np.abs(times[right]-grid),left,right)
    return values[ix],np.abs(times[ix]-grid)


def trace(record):
    a=np.load(OUT/record['cache'])
    # Inference sampling phase, extent, and origin depend only on the follower.
    origin=a['follower_ns'][0]
    lt=(a['leader_ns']-origin)/1e9;st=(a['follower_ns']-origin)/1e9
    grid=np.arange(int(st[-1]*10)+1)/10
    s,dt=nearest(st,a['follower_m'],grid)
    m=np.interp(grid,lt,a['leader_m'])
    return dict(t=grid,s=s,m=m,lt=lt,lm=a['leader_m'],st=st,sm=a['follower_m'],sample_error=dt,
                reference_valid=(grid>=lt[0])&(grid<=lt[-1]))


def characterize(t,signal):
    if len(signal)<10:return None,'too_short'
    low,high=np.quantile(signal,[.05,.95])
    if high-low<.005:return None,'amplitude_below_5mm'
    q=np.clip((signal-low)/(high-low),0,1)
    if np.median(q[:3])<.8:return None,'initial_not_open'
    if np.median(q[-3:])>.2:return None,'final_not_closed'
    down=np.flatnonzero((q[:-1]>.5)&(q[1:]<=.5))
    up=np.flatnonzero((q[:-1]<.5)&(q[1:]>=.5))
    if len(down)!=1 or len(up):return None,'not_single_close'
    i=down[0]
    midpoint=t[i]+(q[i]-.5)/(q[i]-q[i+1])*(t[i+1]-t[i])
    return dict(low=float(low),high=float(high),q=q,midpoint=float(midpoint)),None


def infer(t,s,model):
    features,reason=characterize(t,s)
    if features is None:return None,reason
    low,high=model['leader_low_m'],model['leader_high_m']
    def scale(q):return low+(high-low)*q
    q=features['q']
    fixed=np.clip((s-model['follower_low_m'])/(model['follower_high_m']-model['follower_low_m']),0,1)
    shifted=np.interp(t+model['delay_s'],t,q)
    center=features['midpoint']-model['delay_s']
    template=np.interp(t-center,OFFSETS,model['template'],left=1.,right=0.)
    # Before contact, the follower's observed open plateau is informative about
    # this episode's leader opening; avoid forcing every episode to a task median.
    observed_high=max(low+.005, features['high']+model['open_offset_m'])
    return dict(raw_follower=s,task_affine=scale(fixed),episode_affine=scale(q),
                episode_affine_shift=scale(shifted),event_template=scale(template),
                episode_open_affine_shift=low+(observed_high-low)*shifted,
                episode_open_template=low+(observed_high-low)*template),None


def aggregate(rows):
    return {method:{metric:stats([r['metrics'][method][metric] for r in rows if r['metrics'][method][metric] is not None])
                    for metric in ('mae_mm','transition_mae_mm','phase_mae')}
            for method in METHODS}


def main():
    records=json.loads((OUT/'accepted_episodes.json').read_text())
    records=[r for r in records if r['status']=='ok' and r['task']!='叠衣服']
    fits=collections.defaultdict(list);excluded=[]
    for r in records:
        if r['split']!='calibration':continue
        x=trace(r)
        if np.max(x['sample_error'])>.02:
            excluded.append(dict(source_id=r['source_id'],task=r['task'],split=r['split'],reason='timestamp_gap'));continue
        sf,reason=characterize(x['t'],x['s'])
        if sf is None:
            excluded.append(dict(source_id=r['source_id'],task=r['task'],split=r['split'],reason='follower_'+reason));continue
        mf,reason=characterize(x['lt'],x['lm'])
        if mf is None:
            excluded.append(dict(source_id=r['source_id'],task=r['task'],split=r['split'],reason='calibration_leader_'+reason));continue
        curve=np.interp(OFFSETS+mf['midpoint'],x['lt'],mf['q'],left=1.,right=0.)
        fits[r['task']].append(dict(source_id=r['source_id'],leader_low_m=mf['low'],leader_high_m=mf['high'],
                                   follower_low_m=sf['low'],follower_high_m=sf['high'],
                                   delay_s=sf['midpoint']-mf['midpoint'],template=curve))
    models={}
    for task,rows in fits.items():
        if len(rows)<4:continue
        model={key:float(np.median([r[key] for r in rows])) for key in ('leader_low_m','leader_high_m','follower_low_m','follower_high_m','delay_s')}
        model['open_offset_m']=float(np.median([r['leader_high_m']-r['follower_high_m'] for r in rows]))
        curve=np.median(np.stack([r['template'] for r in rows]),axis=0)
        curve[0]=1.;curve[-1]=0.
        model.update(template=np.minimum.accumulate(np.clip(curve,0,1)).tolist(),
                     n_calibration=len(rows),source_ids=[r['source_id'] for r in rows])
        models[task]=model
    results=[]
    (OUT/'predictions').mkdir(exist_ok=True)
    for r in records:
        if r['split']!='heldout':continue
        if r['task'] not in models:
            excluded.append(dict(source_id=r['source_id'],task=r['task'],split=r['split'],reason='insufficient_calibration'));continue
        assert r['source_id'] not in models[r['task']]['source_ids']
        x=trace(r)
        if np.max(x['sample_error'])>.02:
            excluded.append(dict(source_id=r['source_id'],task=r['task'],split=r['split'],reason='timestamp_gap'));continue
        predictions,reason=infer(x['t'],x['s'],models[r['task']])
        if predictions is None:
            excluded.append(dict(source_id=r['source_id'],task=r['task'],split=r['split'],reason='follower_'+reason));continue
        # No leader-based eligibility filtering of test episodes.
        ml,mh=np.quantile(x['lm'],[.05,.95]);mf,_=characterize(x['lt'],x['lm'])
        qtrue=np.clip((x['m']-ml)/max(mh-ml,1e-9),0,1)
        valid=x['reference_valid']
        transition=(np.abs(x['t']-mf['midpoint'])<=.3)&valid if mf else np.zeros(len(x['t']),dtype=bool)
        model=models[r['task']]
        metrics={}
        for method,pred in predictions.items():
            error=np.abs(pred-x['m'])*1000
            if method=='raw_follower':qpred=np.clip(pred/.095,0,1)
            elif method in ('episode_open_affine_shift','episode_open_template'):
                counterpart='episode_affine_shift' if method=='episode_open_affine_shift' else 'event_template'
                qpred=np.clip((predictions[counterpart]-model['leader_low_m'])/(model['leader_high_m']-model['leader_low_m']),0,1)
            else:qpred=np.clip((pred-model['leader_low_m'])/(model['leader_high_m']-model['leader_low_m']),0,1)
            metrics[method]=dict(mae_mm=float(error[valid].mean()),transition_mae_mm=float(error[transition].mean()) if transition.any() else None,
                                 phase_mae=float(np.mean(np.abs(qpred[valid]-qtrue[valid]))) if mh-ml>=.005 else None)
        row=dict(task=r['task'],source_id=r['source_id'],frames=len(x['t']),metrics=metrics,
                 reference_is_single_close=mf is not None)
        results.append(row)
        np.savez_compressed(OUT/'predictions'/(r['source_id']+'.npz'),time=x['t'],leader=x['m'],reference_valid=valid,
                            leader_time=x['lt'],leader_raw=x['lm'],**predictions)
    payload=dict(scope='Accepted raw episodes, same date/cart/task calibration and holdout; only follower-detected single-close segments; simulated 10 Hz follower input. No cross-task, no-grasp, or cloth-cycle accuracy claim.',
                 metrics='Episode-macro mean absolute error versus true leader command, in mm. Phase error uses separately endpoint-normalized leader reference; not a claim of exact physical command recovery.',
                 candidates=dict(calibration=sum(r['split']=='calibration' for r in records),heldout=sum(r['split']=='heldout' for r in records)),
                 calibration_used=sum(m['n_calibration'] for m in models.values()),heldout_used=len(results),
                 models=models,aggregate=aggregate(results),by_task={task:aggregate([r for r in results if r['task']==task]) for task in models},
                 rows=results,excluded=excluded)
    (OUT/'mapping_metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('calibration used',payload['calibration_used'],'heldout used',payload['heldout_used'])
    for method in METHODS:print(method,payload['aggregate'][method])
    print('exclusions',dict(collections.Counter((r['split'],r['reason']) for r in excluded)))


if __name__=='__main__':main()
