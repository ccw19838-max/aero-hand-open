"""Bundle videos, metrics and verified training status into an offline HTML."""
import argparse
import base64
from datetime import datetime,timezone
import json
from pathlib import Path

from .evaluate import atomic_json


def data_url(path,mime):
    return 'data:'+mime+';base64,'+base64.b64encode(Path(path).read_bytes()).decode()


def build(folder):
    folder=Path(folder).resolve()
    snapshot=json.loads((folder/'snapshot.json').read_text())
    run_path=Path(snapshot['run_directory'])
    snapshot['run']=json.loads((run_path/'status.json').read_text())
    snapshot['checked_at']=datetime.now(timezone.utc).isoformat()
    pid=snapshot['run'].get('child_pid')
    command=Path('/proc')/str(pid)/'cmdline'
    snapshot['process_alive']=command.exists() and b'experiments.cube_z90.train' in command.read_bytes()
    workers=[]
    for candidate in Path('/proc').iterdir():
        if not candidate.name.isdigit():continue
        try:
            fields=(candidate/'stat').read_text().split()
            args=(candidate/'cmdline').read_bytes()
            if int(fields[3])==pid and b'spawn_main' in args:
                workers.append(int(candidate.name))
        except (OSError,ValueError):pass
    snapshot['worker_pids']=sorted(workers)
    snapshot['validations']=[]
    for path in sorted(run_path.glob('stage_*/progress.jsonl')):
        for line in path.read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            if row.get('validation_complete'):snapshot['validations'].append(row)
    evaluation=json.loads((folder/'checkpoint_evaluation/evaluation.json').read_text())
    manifest=json.loads((folder/'replay.json').read_text())
    frames=json.loads((folder/'frames.json').read_text())
    payload=dict(snapshot=snapshot,manifest=manifest,frames=frames,evaluation={
        k:evaluation.get(k) for k in ('complete','episodes_completed','successes','success_rate','capture_rate',
          'recapture_rate','mean_final_angle_error_degrees','mean_best_angle_error_degrees','mean_max_stable_seconds',
          'contact_duty_cycle','failure_counts','seed_start')})
    template=Path(__file__).with_name('preview_template.html').read_text()
    replacements={'__PAYLOAD__':json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/'),
        '__GRID__':data_url(folder/'four_hands_30s.mp4','video/mp4'),
        '__SINGLE__':data_url(folder/'single_hand_30s.mp4','video/mp4'),
        '__POSTER__':data_url(folder/'four_hands_preview.png','image/png')}
    for key,value in replacements.items():template=template.replace(key,value)
    (folder/'index.html').write_text(template)
    atomic_json(folder/'snapshot.json',snapshot)
    atomic_json(folder/'preview_report.json',dict(snapshot=snapshot,manifest=manifest,evaluation=payload['evaluation']))
    print(json.dumps(dict(html=str(folder/'index.html'),bytes=(folder/'index.html').stat().st_size,
                         actual_training_worker_pids=workers,training_steps=snapshot['run']['training_steps'],
                         replay_checkpoint_steps=manifest['checkpoint_steps']),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    build(parser.parse_args().folder)
