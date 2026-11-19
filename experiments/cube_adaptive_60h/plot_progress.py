"""Plot completed neural-only curriculum validations on the real time axis."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/aero-cube-matplotlib')
from pathlib import Path
from datetime import datetime,timezone
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(goal):
    root=Path(goal);goal_meta=json.loads((root/'goal.json').read_text())
    start=datetime.fromisoformat(goal_meta['started_at']).timestamp();rows=[]
    for axis in ('x','y'):
        for path in (root/axis).glob('segment_*/validation/*.json'):
            report=json.loads(path.read_text())
            if not report['complete'] or any(g['requested_trials']!=20 for g in report['groups'].values()):continue
            rows.append(dict(axis=axis,segment=path.parent.parent.name,elapsed=(path.stat().st_mtime-start)/3600.,report=report,source=str(path)))
    fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True)
    metrics=[('success_rate','Success, worst integrator group (%)',100.),('mean_best_error','Mean per-episode best pose error (deg)',1.),
        ('five_finger_duty','Five-finger valid contact time (%)',100.),('maximum_signed_angle','Observed peak angle, may be a failure (deg)',1.)]
    colors={'x':'#d65140','y':'#246cb5'}
    for panel,(key,title,scale) in zip(axs.flat,metrics):
        for axis in ('x','y'):
            first=True
            for segment,target in sorted(set((r['segment'],r['report']['target_degrees']) for r in rows if r['axis']==axis)):
                group=sorted([r for r in rows if r['axis']==axis and r['segment']==segment and r['report']['target_degrees']==target],key=lambda r:r['elapsed'])
                panel.plot([r['elapsed'] for r in group],[scale*r['report'][key] for r in group],'.-',color=colors[axis],
                    label=axis.upper() if first else None,linewidth=1.4,markersize=5);first=False
        panel.set_title(title,fontsize=10);panel.grid(alpha=.2);panel.set_ylim(bottom=0)
    axs[0,0].set_ylim(0,100);axs[1,0].set_ylim(0,100)
    axs[1,1].axhline(90.,color='#777',linestyle='--',linewidth=1,label='Final90deg goal')
    peak=max((r['report']['maximum_signed_angle'] for r in rows),default=90.)
    axs[1,1].set_ylim(0,max(100.,1.05*peak))
    for panel in axs[1]:panel.set_xlabel('Actual hours from fixed goal start (includes evaluation and restarts)')
    axs[0,0].legend(loc='best');axs[1,1].legend(loc='best')
    fig.suptitle('Completed neural-only validations from original initial grasp',fontsize=13)
    targets=' / '.join(axis.upper()+':'+','.join(f'{v:g}' for v in sorted(set(r['report']['target_degrees'] for r in rows if r['axis']==axis)))+'deg' for axis in ('x','y'))
    fig.text(.01,.01,'Lines break at segment/target changes. Targets '+targets+'. Auxiliary, static-endpoint and planning tests are excluded.',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.96));output=root/'plots';output.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S');path=output/f'progress_{stamp}'
    fig.savefig(path.with_suffix('.png'),dpi=160);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)
    (output/'latest_sources.json').write_text(json.dumps(dict(generated_at=datetime.now(timezone.utc).isoformat(),
        png=str(path.with_suffix('.png')),pdf=str(path.with_suffix('.pdf')),sources=[r['source'] for r in rows]),indent=2)+'\n')
    return path.with_suffix('.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);args=p.parse_args();print(plot(args.goal))
