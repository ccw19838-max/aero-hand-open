"""Bounded review rules. Changes require a new, explicitly recorded segment."""
from dataclasses import replace
from datetime import datetime,timedelta
from .config import validate_profile


def next_review(started_at,now,interval_hours=3):
    start=datetime.fromisoformat(started_at)
    elapsed=max(0.,(now-start).total_seconds())
    index=int(elapsed//(interval_hours*3600))+1
    return start+timedelta(hours=interval_hours*index)


def choose(profile,current,best,history,stage):
    if best and current and current['target_degrees']==best['target_degrees']==stage and current['complete'] and best['complete'] and best['success_rate']>=.2 and current['success_rate']<best['success_rate']-.2:
        updated=replace(profile,name='regression_recovery',learning_rate=max(2e-5,profile.learning_rate*.5),
            entropy=max(.0005,profile.entropy*.7),bc_weight=min(.3,profile.bc_weight+.05))
        return dict(action='rollback_best',reason='validation success rate regressed by more than 20 percentage points',
            profile=validate_profile(updated).dictionary(),target=stage)
    recent=[r for r in history if r and r.get('complete') and r.get('target_degrees')==stage][-2:]
    plateau=len(recent)==2 and current and current['complete'] and current['success_rate']==0. and all(r['success_rate']==0. for r in recent)
    if plateau:
        improvement=recent[0]['mean_best_error']-current['mean_best_error']
        if improvement<2.:
            # Requiring an intermediate stop can be harder for a square cube
            # than a 90deg endpoint. Direct 90deg is a goal change, never a
            # relaxed final criterion, and is recorded as a separate segment.
            direct=stage<90. and len(history)>=3
            updated=replace(profile,name='plateau_contact_recovery',
                contact_weight=min(3.,profile.contact_weight*1.4),loss_weight=min(3.,profile.loss_weight*1.3),
                stall_weight=min(3.,profile.stall_weight*1.25),
                learning_rate=max(2e-5,profile.learning_rate*.8),grip_mix=min(.3,profile.grip_mix+.05))
            return dict(action='new_profile',reason='two review intervals without success or >=2deg error improvement',
                profile=validate_profile(updated).dictionary(),target=90. if direct else stage)
    return dict(action='continue',reason='no bounded adaptation trigger',profile=profile.dictionary(),target=stage)


def final_pass(report):
    return bool(report and report['complete'] and report['target_degrees']==90. and report.get('from_original_initial_grasp')
        and set(report['groups'])=={'0.001','0.002'}
        and all(g['complete'] and g['requested_trials']>=100 and g['successes']==g['requested_trials'] for g in report['groups'].values()))
