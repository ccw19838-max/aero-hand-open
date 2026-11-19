"""Separate observed peaks from successful endpoints in saved validation data."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json
from .storage import atomic_json


def summarize(report,path):
    records=report.get('records',[]);peaks=[];held=[]
    for record in records:
        final=record['final']
        item=dict(degrees=float(final['max_progress_degrees']),seed=record['seed'],dt=record['dt'],
            target_degrees=report['target_degrees'],success=record['success'],
            failure=final.get('failure_reason'),report=str(path))
        peaks.append(item)
        if (record['success'] and final.get('success') and not final.get('failure_reason')
                and final.get('max_stable_seconds',0.)>=1.-1e-9):
            held.append({**item,'degrees':float(final['signed_progress_degrees']),
                'hold_seconds':final['max_stable_seconds']})
    return dict(target_degrees=report['target_degrees'],success_rate=report['success_rate'],
        complete=report['complete'],groups=report['groups'],report=str(path),
        peak=max(peaks,key=lambda item:item['degrees'],default=None),
        qualified_endpoint=max(held,key=lambda item:item['degrees'],default=None))


class RotationEvidence:
    def __init__(self,root):self.root=Path(root);self.cache={}
    def read(self,path,kind):
        modified=path.stat().st_mtime_ns;key=str(path)
        if key not in self.cache or self.cache[key][0]!=modified:
            report=json.loads(path.read_text());parts=[]
            if 'axis' in report:parts=[(report['axis'],kind,'',report)]
            else:
                for axis,entry in report.get('axes',{}).items():
                    for role in ('current','best','final90'):
                        if entry.get(role):parts.append((axis,'review_'+role,f'axes.{axis}.{role}',entry[role]))
            items=[]
            for axis,role,section,value in parts:
                summary=summarize(value,path);summary.update(kind=role,report_section=section)
                for field in ('peak','qualified_endpoint'):
                    if summary[field]:summary[field]['report_section']=section
                items.append((axis,summary))
            self.cache[key]=(modified,items)
        return self.cache[key]
    def write(self):
        result=dict(observed_at=datetime.now(timezone.utc).isoformat(),
            scope='Current goal saved validation only; single successful samples do not establish a new curriculum or90deg verification',axes={})
        groups={axis:[] for axis in ('x','y')}
        inputs=[]
        for axis in groups:
            inputs.extend((path,'curriculum') for path in (self.root/axis).glob('segment_*/validation/*.json'))
            inputs.extend((path,'independent') for path in (self.root/axis).glob('segment_*/independent/*.json'))
        inputs.extend((path,'review') for path in (self.root/'reviews').glob('*_review.json'))
        inputs.extend((path,'verified') for path in (self.root/'reviews').glob('*_verified.json'))
        for path,kind in inputs:
            modified,items=self.read(path,kind)
            for axis,summary in items:groups[axis].append((modified,summary))
        for axis in ('x','y'):
            rows=groups[axis]
            peaks=[v['peak'] for _,v in rows if v['peak']]
            held=[v['qualified_endpoint'] for _,v in rows if v['qualified_endpoint']]
            result['axes'][axis]=dict(
                observed_peak=max(peaks,key=lambda v:v['degrees'],default=None),
                qualified_endpoint=max(held,key=lambda v:v['degrees'],default=None),
                latest_completed_validation=max((row for row in rows if row[1]['complete'] and row[1]['kind'] in ('curriculum','review_current')),
                    key=lambda row:row[0],default=(0,None))[1])
        atomic_json(self.root/'rotation_summary.json',result)
        lines=['# 本轮旋转角度记录','', '峰值可能出现在失败轨迹中；合格单次角度不代表通过更高角度课程。这里只统计当前60小时目标内保存的验证，旧实验和选定初态诊断不混入。','',
            '| 轴 | 本轮瞬时峰值 | 合格单次最终角度 | 最近完成课程 / 成功率 |','|---|---:|---:|---|']
        for axis,evidence in result['axes'].items():
            peak=evidence['observed_peak'];held=evidence['qualified_endpoint'];latest=evidence['latest_completed_validation']
            peak_text=f'{peak["degrees"]:.2f}°' if peak else '尚无记录'
            held_text=f'{held["degrees"]:.2f}°（目标{held["target_degrees"]:g}°）' if held else '尚无合格单次'
            latest_text=f'{latest["target_degrees"]:g}° / {latest["success_rate"]:.1%}' if latest else '尚无完整验证'
            lines.append(f'| {axis.upper()} | {peak_text} | {held_text} | {latest_text} |')
        lines+=['','各值的源报告、种子和物理步长见 rotation_summary.json。最终90°验收状态以控制器状态和独立验收报告为准。']
        (self.root/'rotation_summary.md').write_text('\n'.join(lines)+'\n')
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--goal',required=True);args=p.parse_args()
    print(json.dumps(RotationEvidence(args.goal).write(),ensure_ascii=False,indent=2))
