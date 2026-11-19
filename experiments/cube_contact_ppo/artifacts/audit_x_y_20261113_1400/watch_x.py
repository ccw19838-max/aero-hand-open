"""Bounded passive monitoring, no changes to trainers or hyperparameters."""
from datetime import datetime,timezone
from pathlib import Path
import json
import time

OUT=Path(__file__).resolve().parent
RUN=OUT.parent/'contact_ppo_v1'


def main():
    started=time.monotonic();records=0
    with (OUT/'x_monitor.jsonl').open('a') as stream:
        while time.monotonic()-started<1800.:
            status=json.loads((RUN/'x/status.json').read_text());report=status.get('latest_evaluation',{})
            record=dict(observed_at=datetime.now(timezone.utc).isoformat(),
                training_steps=status['training_steps'],status=status['status'],curriculum=status['curriculum'],
                success_rate=report.get('min_integrator_success_rate'),error_degrees=report.get('mean_best_angle_error_degrees'),
                five_finger_duty=report.get('mean_five_finger_duty'),per_finger_zero_contact_fraction=report.get('per_finger_zero_contact_fraction'))
            stream.write(json.dumps(record)+'\n');stream.flush();records+=1
            (OUT/'monitor_status.json').write_text(json.dumps(dict(status='monitoring',records=records,last=record,
                max_duration_seconds=1800,training_unchanged=True),indent=2)+'\n')
            if status['status'] in ('stopped','completed','error'):break
            time.sleep(15.)
    (OUT/'monitor_status.json').write_text(json.dumps(dict(status='finished',records=records,training_unchanged=True),indent=2)+'\n')


if __name__=='__main__':main()
