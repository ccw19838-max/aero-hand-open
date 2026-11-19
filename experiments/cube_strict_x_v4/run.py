"""Time-bounded paired PPO experiment. Each checkpoint is taken between updates."""
import argparse
import fcntl
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/aero-cube-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/aero-cube-cache")

from experiments.cube_adaptive_60h.config import Profile
from experiments.cube_adaptive_60h.storage import atomic_json, digest
from .curriculum import ORIGINAL_DEGREES, REVISED_DEGREES, apply_amendment, record_amendment
from .evaluation import evaluate, score

PHASE_NAMES = {"v3_pilot":"V3 对照", "v4_pilot":"V4 对照", "curriculum":"V4 逐级训练",
               "final":"冻结评估和回放", "done":"已交付"}


def beijing(timestamp):
    return datetime.fromtimestamp(timestamp, ZoneInfo("Asia/Shanghai")).strftime("%m-%d %H:%M:%S")


def epoch(value):
    return datetime.fromisoformat(value).timestamp()


def qualifies(report):
    groups = report["groups"]
    return (report["complete"] and set(groups) == {"0.001", "0.002"}
            and all(g["complete"] and g["trials"] == g["requested_trials"] == 20
                    and g["successes"] >= 16 for g in groups.values()))


def plateau(rows):
    window = rows[-4:]
    if len(window) < 4:
        return False
    return (max(r["rate"] for r in window[1:]) <= window[0]["rate"]
            and max(r["median"] for r in window[1:]) - window[0]["median"] < 1.)


class Experiment:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.job = json.loads((self.root / "job.json").read_text())
        self.profile = Profile(**self.job["profile"])
        self.deadline = epoch(self.job["deadline_at"])
        self.freeze = self.deadline - self.job["final_seconds"]
        self.checksum = digest(self.root/"job.json")
        self.stop_requested = False
        statefile = self.root / "state.json"
        if statefile.exists():
            self.state = json.loads(statefile.read_text())
            if self.state["job_sha256"] != self.checksum:
                raise RuntimeError("Job configuration changed after initialization")
        else:
            self.state = dict(job_sha256=self.checksum,phase="v3_pilot",stage=5,arm_deadlines={},
                              milestones={"v3":{},"v4":{}},steps={"v3":0,"v4":0},history={},
                              best={},qualified={},practice={},next_regular={},streak=0,final_quorum=False,
                              last_v4_checkpoint=None)
        apply_amendment(self.root, self.state, self.job)
        self.workers = []
        self.engine = None

    def persist(self, activity):
        s = self.state
        atomic_json(self.root/"state.json", s)
        status = dict(task=self.job["task"],phase=s["phase"],activity=activity,stage=s["stage"],
                      pid=os.getpid(),worker_pids=self.workers,training_steps=s["steps"],
                      started_at=self.job["started_at"],deadline_at=self.job["deadline_at"],
                      updated_at=datetime.now(timezone.utc).isoformat(),best=s["best"],highest_qualified=max(map(int,s["qualified"]),default=0),
                      last_evaluation=s.get("last_evaluation"),last_v4_checkpoint=s["last_v4_checkpoint"],
                      final=s.get("outcome"),next_evaluation=s["next_regular"].get("v3" if s["phase"]=="v3_pilot" else "v4"))
        status["curriculum_degrees"]=s.get("curriculum_degrees", list(ORIGINAL_DEGREES))
        atomic_json(self.root/"status.json", status)
        rows = ["# X 轴魔方训练", "", f'{PHASE_NAMES[s["phase"]]}，当前目标 {s["stage"]}°。{activity}。',
                f'采样步数：V3 {s["steps"]["v3"]:,}，V4 {s["steps"]["v4"]:,}。活动采样进程 {len(self.workers)} 个。',
                f'预算截止：{beijing(self.deadline)}（北京时间）。']
        if "amendment_sha256" in s:
            rows += ["[当前课程 30° → 60° → 90°](amendment.json) · [切换前的训练记录](state_before_30_60_90.json)"]
        if status["last_evaluation"]:
            entry = status["last_evaluation"]
            rows += ["",f'最近评估：{entry["arm"].upper()}，{entry["target"]}°，{entry["steps"]:,} 步。',
                     f'1 ms 成功 {entry["counts"][0]}/20，2 ms 成功 {entry["counts"][1]}/20。首次违规前峰值的中位数 {entry["median"]:.2f}°，最佳目标保持 {entry["hold"]:.2f} s。',
                     f'[详细评估]({entry["report"]})']
            for dt,g in entry.get("groups",{}).items():
                losses = "，".join(f'{label} {count} 次' + (f'（中位 {g["median_first_violation_angle_by_finger"][finger]:.2f}°）' if count else "")
                                   for finger,label in (("index","食指"),("middle","中指"),("ring","无名指"))
                                   for count in [g["first_violation_finger_counts"][finger]])
                lead = g["warning_median_lead_ms"]
                lead_text = "无匹配记录" if lead is None else f'中位提前 {lead:g} ms'
                rows.append(f'{1000*float(dt):g} ms：有效转角中位 {g["median_valid_peak_degrees"]:.2f}°，'
                            f'瞬时峰值 {g["maximum_signed_angle"]:.2f}°，到位保持最长 {g["maximum_stable_seconds"]:.2f} 秒。'
                            f'首次违规：{losses}。预警 {g["warning_count"]} 次，{lead_text}，'
                            f'50 ms 内未失力 {g["warning_false_count"]} 次，观察窗口未结束 {g["warning_censored_count"]} 次。')
        if (self.root/"comparison.json").exists():
            rows += ["","[相同更新步数下的对照结果](comparison.json)"]
        if status["final"]:
            rows += ["","[冻结评估与回放](final/REPORT.md)"]
        tmp = self.root/"status.md.tmp"
        tmp.write_text("\n\n".join(rows)+"\n")
        tmp.replace(self.root/"status.md")
        active_file=self.root.parents[2]/"ACTIVE_EXPERIMENT.json"
        active=dict(task=self.job["task"],root=str(self.root),status_file=str(self.root/"status.json"),
                    human_status=str(self.root/"status.md"),active_axes=["x"],suspended_axes=["y","z"],
                    trainer_pid=os.getpid(),worker_pids=self.workers,live_parallel_environments=len(self.workers),
                    allocation={"x":4,"y":0,"z":0},deadline_at=self.job["deadline_at"],runtime_status=activity,
                    preserved_y_manifest=self.job["preserved_y_manifest"],updated_at=status["updated_at"])
        atomic_json(active_file,active)

    def assess(self,arm,target,limit,kind):
        engine=self.engine
        token=f"{arm}_{int(target):02d}_{engine.steps:09d}"
        report_path=self.root/"evaluations"/(token+".json")
        report=json.loads(report_path.read_text()) if report_path.exists() else None
        if report is None or not report["complete"]:
            saved=engine.save("validation")
            self.persist("评估中")
            report=evaluate(engine.model,self.root/"inputs",self.profile,target,20,self.job["validation_seed"],
                            limit,lambda:self.stop_requested,arm=arm)
            report["checkpoint"]=str(saved)
            atomic_json(report_path,report)
        if not report["complete"]:
            return None
        row=dict(arm=arm,target=target,steps=engine.steps,epochs=engine.epochs,
                 counts=[report["groups"][str(dt)]["successes"] for dt in (.001,.002)],
                 rate=report["success_rate"],median=report["median_valid_peak_degrees"],
                 hold=report["maximum_stable_seconds"],score=list(score(report)),
                 checkpoint=report["checkpoint"],report=str(report_path.relative_to(self.root)),
                 groups=report["groups"])
        self.state["last_evaluation"]=row
        key=str(int(target))
        if arm=="v4":
            self.state["last_v4_checkpoint"]=row["checkpoint"]
            old=self.state["best"].get(key)
            if old is None or tuple(row["score"])>tuple(old["score"]):
                self.state["best"][key]=row
        if kind=="milestone":
            self.state["milestones"][arm][str(engine.steps)]=row
            self.write_comparison()
        if kind=="regular":
            while self.state["next_regular"][arm]<=time.time():
                self.state["next_regular"][arm]+=self.job["eval_seconds"]
            if self.state["phase"]=="curriculum":
                history=self.state["history"].setdefault(key,[])
                if not history or history[-1]["steps"]!=engine.steps:
                    history.append(row)
                    self.state["streak"]=self.state["streak"]+1 if qualifies(report) else 0
                    if self.state["streak"]>=2:
                        self.state["qualified"][key]=row
                        self.state["streak"]=0
                        if target==90:
                            self.state["final_quorum"]=True
                        else:
                            ladder=self.state.get("curriculum_degrees", ORIGINAL_DEGREES)
                            self.state["stage"]=ladder[ladder.index(target)+1]
                            engine.set_demonstration(None)
                            engine.set_stage(self.state["stage"])
                    elif plateau(history) and key not in self.state["practice"]:
                        self.run_search(key,limit)
        self.persist("评估完成")
        return row

    def run_search(self,key,limit):
        from .stall import search, verify_practice
        path=self.root/"practice"/key
        self.persist("搜索失触前的三指动作")
        report=search(self.engine.model,self.root/"inputs",self.profile,int(key),path,
                      self.job["diagnostic_seed"]+int(key)*100,limit,lambda:self.stop_requested,seconds=600.)
        demo=verify_practice(path,int(key)) if report["qualified"] else None
        if demo is not None:
            self.engine.set_demonstration(demo)
            self.state["last_v4_checkpoint"]=str(self.engine.save("verified_practice"))
        self.state["practice"][key]=dict(qualified=bool(demo),path=str(path))

    def write_comparison(self):
        arms=self.state["milestones"]
        common=sorted(arms["v3"].keys() & arms["v4"].keys(),key=int)
        if common:
            pairs=[dict(steps=int(k),v3=arms["v3"][k],v4=arms["v4"][k]) for k in common]
            atomic_json(self.root/"comparison.json",dict(validation_seed=self.job["validation_seed"],pairs=pairs))
            lines = ["# 5° 配对训练结果", "",
                     "每次比较都使用相同的 PPO 更新次数、原始握姿和 20 个验证初态。"
                     "有效峰值截止于三根稳定指的首次违规。保持计时要求五指受力、三维姿态和速度同时合格。", "",
                     "|采样步数|策略|步长|严格成功|有效峰值中位|有效峰值最高|瞬时峰值最高|最长目标保持|",
                     "|---:|---|---:|---:|---:|---:|---:|---:|"]
            for pair in pairs:
                for arm in ("v3", "v4"):
                    row = pair[arm]
                    for dt, group in row["groups"].items():
                        lines.append(f'|{pair["steps"]:,}|[{arm.upper()}]({row["report"]})|{1000*float(dt):g} ms|'
                                     f'{group["successes"]}/{group["trials"]}|{group["median_valid_peak_degrees"]:.2f}°|'
                                     f'{group["maximum_eligible_peak_degrees"]:.2f}°|{group["maximum_signed_angle"]:.2f}°|'
                                     f'{group["maximum_stable_seconds"]:.2f} s|')
            temporary = self.root / "comparison.md.tmp"
            temporary.write_text("\n".join(lines)+"\n")
            temporary.replace(self.root / "comparison.md")

    def run_arm(self,arm,pilot):
        from .engine import Engine
        s=self.state
        end=self.freeze
        if pilot:
            end=min(end,s["arm_deadlines"].setdefault(arm,time.time()+self.job["pilot_arm_seconds"]),
                    epoch(self.job["started_at"])+6*3600)
        if time.time()>=end:
            return
        s["next_regular"].setdefault(arm,time.time()+self.job["eval_seconds"])
        self.persist("加载策略")
        alias=self.root/"arms"/arm/"latest"
        engine=Engine(self.root,arm,stage=s["stage"],checkpoint=alias if alias.exists() else None,
                      stop=lambda:self.stop_requested)
        self.engine=engine
        self.workers=list(engine.worker_pids)
        try:
            if engine.stage!=s["stage"]:
                engine.set_demonstration(None)
                engine.set_stage(s["stage"])
            if ("amendment_sha256" in s and not s.get("entry_evaluation") and
                    arm=="v4" and s["stage"]==REVISED_DEGREES[0]):
                entry=self.assess(arm,s["stage"],end,"entry")
                if entry is None:
                    return
                s["entry_evaluation"]=entry["report"]
                self.persist("已完成 30° 起点评估")
            last_save=time.time()
            while not self.stop_requested and time.time()<end and not s["final_quorum"]:
                s["steps"][arm]=engine.steps
                if pilot and engine.steps in self.job["pilot_milestones"]:
                    if str(engine.steps) not in s["milestones"][arm]:
                        if self.assess(arm,5,end,"milestone") is None:
                            break
                if time.time()>=s["next_regular"][arm]:
                    if self.assess(arm,s["stage"],end,"regular") is None:
                        break
                if pilot and engine.steps>=self.job["pilot_max_steps"]:
                    break
                if s["final_quorum"]:
                    break
                self.persist("采样和更新")
                if not engine.advance(deadline=end):
                    break
                s["steps"][arm]=engine.steps
                if engine.steps==1024 or time.time()-last_save>=self.job["save_seconds"]:
                    path=engine.save("periodic")
                    if arm=="v4":s["last_v4_checkpoint"]=str(path)
                    last_save=time.time()
            path=engine.save("segment_end")
            if arm=="v4":s["last_v4_checkpoint"]=str(path)
            s["steps"][arm]=engine.steps
        finally:
            engine.close()
            self.engine=None
            self.workers=[]
            self.persist("已保存进度")

    def run(self):
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        def stop(*_):
            self.stop_requested=True
        for sig in (signal.SIGTERM,signal.SIGINT):
            signal.signal(sig,stop)
        s=self.state
        while not self.stop_requested and s["phase"]!="done":
            if time.time()>=self.deadline:
                self.persist("预算已截止")
                return
            if time.time()>=self.freeze or s["final_quorum"]:
                s["phase"]="final"
            phase=s["phase"]
            if phase=="final":
                from .finalize import finalize
                self.persist("冻结候选并独立评估")
                s["outcome"]=finalize(self.root,s,self.deadline,lambda:self.stop_requested)
                if self.stop_requested:
                    self.persist("已暂停")
                    return
                if not s["outcome"].get("complete") and s["outcome"].get("error") and time.time()<self.deadline:
                    self.persist("冻结评估将从已保存的初态或录像重试")
                    raise RuntimeError(s["outcome"]["error"])
                s["phase"]="done"
                self.persist("已完成" if s["outcome"].get("complete") else "交付未完整")
                return
            arm="v3" if phase=="v3_pilot" else "v4"
            self.run_arm(arm,phase.endswith("pilot"))
            if self.stop_requested:
                self.persist("已暂停")
                break
            s["phase"]={"v3_pilot":"v4_pilot","v4_pilot":"curriculum","curriculum":"final"}[phase]
            self.persist("切换阶段")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--amend-30-60-90",action="store_true",
                        help="Preserve the current V4 checkpoint and switch the paused run to 30, 60, 90 degrees")
    args=parser.parse_args()
    with (args.root/"experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("The experiment is already active")
        state_path = args.root / "state.json"
        if not args.amend_30_60_90 and state_path.exists():
            if json.loads(state_path.read_text())["phase"] == "done":
                print(f"Completed experiment: {args.root / 'final' / 'REPORT.md'}")
                return
        experiment=Experiment(args.root)
        if args.amend_30_60_90:
            if time.time()>=experiment.freeze:
                raise ValueError("Training window ended before the requested change")
            record_amendment(experiment.root,experiment.state,experiment.job)
            apply_amendment(experiment.root,experiment.state,experiment.job)
            atomic_json(experiment.root/"state.json",experiment.state)
        else:
            experiment.run()


if __name__=="__main__":
    main()
