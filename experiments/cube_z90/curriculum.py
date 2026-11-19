"""A bounded Z-only supervisor, with no time allocated to X/Y training."""
import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .config import ANGLES, CubeReorientConfig, tolerance_for
from .evaluate import atomic_json


def now():
    return datetime.now(timezone.utc)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


class Runner:
    def __init__(self, reference, initial_checkpoint, output, hours=24., steps=10000000,
                 envs=4, resume=False, action_delta_fraction=.01, deadline_at=None):
        if not math.isfinite(hours) or not 0 < hours <= 24:
            raise ValueError('Expected a finite budget of at most 24 hours')
        if not 1 <= envs <= 4 or not 0 < steps <= 10000000:
            raise ValueError('Invalid CPU or step budget')
        if action_delta_fraction not in (.01,.02):
            raise ValueError('Expected the tested 1% or pilot 2% action increment')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / '.supervisor.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError('This run already has a live supervisor')
        self.child = None
        self.stop = None
        self.reference = Path(reference).resolve()
        previous = read_json(self.output / 'status.json')
        if previous and not resume:
            raise ValueError('Choose a new output or use --resume')
        if resume:
            if not previous or previous['reference'] != str(self.reference):
                raise ValueError('Missing run or different reference')
            child_pid = previous.get('child_pid')
            try:
                command = (Path('/proc') / str(child_pid) / 'cmdline').read_bytes()
                if b'experiments.cube_z90.' in command:
                    raise ValueError('Previous child is still alive; do not duplicate it')
            except OSError:
                pass
            self.state = previous
        else:
            start = now()
            deadline = start+timedelta(hours=hours)
            if deadline_at:
                fixed=datetime.fromisoformat(deadline_at)
                if fixed.tzinfo is None or not start < fixed <= deadline:
                    raise ValueError('Absolute deadline must be within the remaining budget')
                deadline=fixed
            self.state = dict(task='cube_z90_aggressive_v1', status='initializing',
                started_at=start.isoformat(), deadline_at=deadline.isoformat(),
                budget_hours=hours, step_budget=steps, envs=envs, reference=str(self.reference),
                action_delta_fraction=action_delta_fraction,
                source_checkpoint=str(Path(initial_checkpoint).resolve()),
                training_axes=['z'], target_degrees=5., passed_degrees=0.,
                training_steps=0, closed_stage_steps=0, stage_index=0, active_stage=None,
                stages={}, milestones={}, verification_attempts=0,
                validation_episodes=20, validation_every_steps=16384,
                checkpoint_every_steps=4096, final_hold_seconds=1., final_tolerance_degrees=5.,
                child_pid=None)
        self.deadline = datetime.fromisoformat(self.state['deadline_at']).timestamp()
        self.state.update(status='running', pid=os.getpid(), stop_reason=None)
        self.write()

    def write(self):
        self.state.update(updated_at=now().isoformat(),
                          remaining_seconds=max(0., self.deadline-time.time()))
        atomic_json(self.output / 'status.json', self.state)

    def request_stop(self, signum=signal.SIGTERM):
        self.stop = signal.Signals(signum).name
        if self.child and self.child.poll() is None:
            self.child.send_signal(signum)

    def finish(self, reason, status='stopped'):
        self.state.update(status=status, stop_reason=reason, finished_at=now().isoformat(),
                          active_stage=None, child_pid=None)
        self.write()
        return self.state

    def run_child(self, name, command, training=True):
        folder = self.output / name
        folder.mkdir(exist_ok=True)
        self.state['active_stage'] = name
        with (self.output / (name+'.log')).open('a') as log:
            self.child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                cwd=Path(__file__).resolve().parents[2], start_new_session=True)
            self.state['child_pid'] = self.child.pid
            self.write()
            stopping = None
            last_write = 0.
            while self.child.poll() is None:
                stamp = time.time()
                if (self.stop or stamp >= self.deadline) and stopping is None:
                    self.child.send_signal(signal.SIGTERM)
                    stopping = time.monotonic()
                if stopping is not None and time.monotonic()-stopping > 60.:
                    try:
                        os.killpg(self.child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.child.wait()
                    self.state['forced_cleanup'] = True
                    break
                if stamp-last_write >= 5.:
                    if training:
                        child = read_json(folder / 'status.json')
                        if child:
                            self.state['training_steps'] = self.state['closed_stage_steps'] + int(child['training_steps'])
                            self.state['latest_checkpoint'] = child.get('latest_checkpoint')
                            self.state['latest_validation'] = child.get('latest_validation')
                            self.state['child_status'] = child['status']
                    self.write()
                    last_write = stamp
                time.sleep(.5)
            code = self.child.wait()
            self.child = None
        self.state.update(child_pid=None)
        self.write()
        return code

    def run(self):
        try:
            while not self.stop and time.time() < self.deadline:
                if self.state['closed_stage_steps'] >= self.state['step_budget']:
                    return self.finish('step_budget')
                angle = self.state['target_degrees']
                name = self.state.get('active_stage')
                if not name or not name.startswith('stage_'):
                    name = f'stage_{self.state["stage_index"]:03d}_z{angle:g}'
                folder = self.output / name
                folder.mkdir(exist_ok=True)
                config = CubeReorientConfig(target_degrees=angle,
                                            angle_tolerance_degrees=tolerance_for(angle),
                                            action_delta_fraction=self.state.get('action_delta_fraction',.02))
                config_path = folder / 'stage_config.json'
                atomic_json(config_path, config.to_dict())
                old = read_json(folder / 'status.json')
                checkpoint = old.get('latest_checkpoint') if old else None
                option = '--resume' if checkpoint else '--init-checkpoint'
                checkpoint = checkpoint or self.state['source_checkpoint']
                already = json.loads((Path(checkpoint)/'metadata.json').read_text())['training_steps'] if option == '--resume' else 0
                remaining = self.state['step_budget']-self.state['closed_stage_steps']-already
                if remaining <= 0:
                    return self.finish('step_budget')
                command = [sys.executable, '-m', 'experiments.cube_z90.train',
                    '--reference', str(self.reference), '--output', str(folder),
                    '--config', str(config_path), '--hours', str(max(.0001, (self.deadline-time.time())/3600.)),
                    '--steps', str(remaining), '--envs', str(self.state['envs']),
                    '--eval-every', '16384', '--eval-episodes', '20', '--baseline-episodes', '0',
                    '--seed', str(12000+self.state['stage_index']), option, str(checkpoint)]
                code = self.run_child(name, command)
                child = read_json(folder / 'status.json')
                if not child:
                    return self.finish('missing_training_status', 'error')
                if code != 0:
                    return self.finish('training_child_failed', 'error')
                self.state['closed_stage_steps'] += child['training_steps']
                self.state['training_steps'] = self.state['closed_stage_steps']
                self.state['source_checkpoint'] = child['latest_checkpoint']
                self.state['latest_checkpoint'] = child['latest_checkpoint']
                self.state['latest_validation'] = child.get('latest_validation')
                self.state['stages'][name] = {k:child.get(k) for k in
                    ('training_steps', 'stop_reason', 'latest_checkpoint', 'latest_validation')}
                self.state['active_stage'] = None
                self.state['stage_index'] += 1
                self.write()
                reason = child.get('stop_reason')
                if reason not in ('angle_stage_ready', 'final_targets_validated'):
                    return self.finish(self.stop or reason or 'stage_stopped')
                check = child.get('latest_validation') or {}
                if not check.get('validation_complete') or check.get('consecutive_axis_passes', {}).get('z', 0) < 2:
                    return self.finish('invalid_promotion', 'error')
                self.state['passed_degrees'] = angle
                self.state['milestones'][str(angle)] = child['latest_checkpoint']
                if angle < 90.:
                    self.state['target_degrees'] = ANGLES[ANGLES.index(angle)+1]
                    self.write()
                    continue

                # A new held-out seed block for each verification attempt.
                attempt = self.state['verification_attempts']
                self.state['verification_attempts'] += 1
                verify_name = f'verify90_{attempt:03d}'
                command = [sys.executable, '-m', 'experiments.cube_z90.evaluate',
                    '--checkpoint', child['latest_checkpoint'], '--output', str(self.output/verify_name),
                    '--episodes', '100', '--seed-start', str(50000000+attempt*1000), '--axis', 'z']
                code = self.run_child(verify_name, command, training=False)
                report = read_json(self.output/verify_name/'evaluation.json')
                self.state['independent_evaluation'] = None if not report else {k:report.get(k) for k in
                    ('complete', 'success_rate', 'capture_rate', 'recapture_rate', 'mean_final_angle_error_degrees', 'seed_start')}
                self.state['active_stage'] = None
                self.write()
                if code == 0 and report and report['complete'] and report['success_rate'] >= .9:
                    # Save a labelled, real-time replay only after an actual pass.
                    video_name = 'verified_replay'
                    command = [sys.executable, '-m', 'experiments.cube_z90.evaluate',
                        '--checkpoint', child['latest_checkpoint'], '--output', str(self.output/video_name),
                        '--episodes', '1', '--seed-start', '70000000', '--axis', 'z', '--video', '30']
                    self.run_child(video_name, command, training=False)
                    return self.finish('z90_independently_verified', 'completed')
                if self.stop or time.time() >= self.deadline:
                    return self.finish(self.stop or 'wall_clock_budget')
                # Keep training Z90 after a failed independent check. Its next
                # verification uses fresh seeds; all failed reports are retained.
                self.write()
            return self.finish(self.stop or 'wall_clock_budget')
        except BaseException as error:
            self.state['error'] = f'{type(error).__name__}: {error}'
            self.finish('supervisor_error', 'error')
            raise
        finally:
            if self.child and self.child.poll() is None:
                self.child.send_signal(signal.SIGTERM)
                try:
                    self.child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(self.child.pid, signal.SIGKILL)
                    self.child.wait()
            self.lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--initial-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--hours', type=float, default=24.)
    parser.add_argument('--steps', type=int, default=10000000)
    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--action-delta', type=float, default=.01)
    parser.add_argument('--deadline-at', help='Optional original absolute deadline, for a revised pilot within one total budget')
    args = parser.parse_args()
    runner = Runner(args.reference, args.initial_checkpoint, args.output,
                    args.hours, args.steps, args.envs, args.resume, args.action_delta, args.deadline_at)
    signal.signal(signal.SIGTERM, lambda signum, frame: runner.request_stop(signum))
    signal.signal(signal.SIGINT, lambda signum, frame: runner.request_stop(signum))
    result = runner.run()
    print(json.dumps({k:result.get(k) for k in ('status', 'stop_reason', 'training_steps', 'passed_degrees', 'deadline_at')}), flush=True)


if __name__ == '__main__':
    main()
