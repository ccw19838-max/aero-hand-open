"""Export the working source tree, three trained checkpoints and frozen results."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiments/cube_strict_x_v4"
RUN = REPO / EXPERIMENT / "artifacts/strict_x_20261118"
Y_RUN = REPO / "experiments/preserved_policies/y90_best_20261116_155308"
PREFIX = "aero-hand-open"
MODULES = ("sim_rl/mujoco_playground", "sim_rl/simulation")
MENAGERIE = "sim_rl/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie"
SKIP = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "build", "dist"}
SOURCE = {".py", ".c", ".h", ".cpp", ".hpp", ".sh", ".ipynb", ".ino", ".js", ".ts"}
FIXTURES = (
    "experiments/cube_official_baseline/artifacts/source/author_actor_critic.npz",
    "experiments/cube_official_baseline/artifacts/source/author_actor_critic.json",
    "experiments/cube_two_driver_v3/artifacts/thumb_pinky_20261117/job.json",
    "experiments/cube_two_driver_v3/artifacts/thumb_pinky_20261117/inputs/reference.npz",
    "experiments/cube_two_driver_v3/artifacts/thumb_pinky_20261117/inputs/prepare_report.json",
)
REQUIRED = {
    f"{EXPERIMENT}/continue_training.py", f"{EXPERIMENT}/GPU_HANDOFF.md",
    f"{EXPERIMENT}/requirements-gpu.txt", "sim_rl/simulation/mujoco/assets/tetheria_mount.obj",
    "sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/right_hand.xml",
    "experiments/cube_grasp/assets/robogym/rubik_locked.xml", "experiments/cube_grasp/assets/robogym/rubik.png",
    *FIXTURES,
}
POLICIES = {f"checkpoints/{name}/policy.zip" for name in ("x_latest", "x_best", "y_best")}


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def source_files(repo):
    """Include tracked and untracked code plus the local copies of both gitlinks and hand assets."""
    files = {}
    for directory in (repo, *(repo/path for path in MODULES)):
        arguments = ["ls-files", "-z", "--cached", "--others", "--exclude-standard"]
        if directory == repo:
            arguments += ["--", ".", ":(exclude)experiments", ":(exclude)checkpoints"]
        for raw in git(directory, *arguments).split(b"\0"):
            if not raw:
                continue
            path = directory / os.fsdecode(raw)
            relative = path.relative_to(repo)
            if set(relative.parts) & SKIP or path.name == "MUJOCO_LOG.TXT" or path.suffix in {".pyc", ".onnx"}:
                continue
            if path.is_file():
                files[str(relative)] = path

    # The local Menagerie checkout is sparse. Only the installed Aero assets are relevant here.
    for directory in (repo / MENAGERIE, repo / "experiments"):
        for folder, dirs, names in os.walk(directory):
            dirs[:] = [name for name in dirs if name not in SKIP | {"preserved_policies", "checkpoints"}
                       and not name.endswith(".egg-info")]
            for name in names:
                path = Path(folder) / name
                relative = path.relative_to(repo)
                if "artifacts" in relative.parts and path.suffix not in SOURCE:
                    continue
                if path.suffix in {".pyc", ".log", ".so"} or name in {"ACTIVE_EXPERIMENT.json", "MUJOCO_LOG.TXT"}:
                    continue
                if path.is_file() and not path.is_symlink():
                    files[str(relative)] = path
    files.update({key: repo/key for key in FIXTURES})
    return files


def collect(repo, run, y_run):
    repo, run, y_run = (path.resolve(strict=True) for path in (repo, run, y_run))
    files = source_files(repo)
    state = read_json(run/"state.json")
    if state["phase"] != "done":
        raise ValueError("Export a completed run with a frozen evaluation")
    final = run/"final"
    frozen = read_json(final/"frozen.json")
    task = next(task for task in frozen["tasks"] if task["name"] == "challenge_30")
    latest = (run/"arms/v4/latest").resolve(strict=True)
    best = final/"checkpoints"/task["name"]
    y = read_json(y_run/"manifest.json")
    if latest.name != Path(frozen["resumable_checkpoint"]).name:
        raise ValueError("Latest checkpoint differs from the resumable checkpoint in the final report")
    roles = {}
    for role, folder in (("x_latest", latest), ("x_best", best), ("y_best", y_run/"checkpoint")):
        metadata = read_json(folder/"metadata.json")
        names = metadata["files"]
        for name, expected in names.items():
            if Path(name).name != name or sha256(folder/name) != expected:
                raise ValueError(f"Source checkpoint failed integrity check: {role}/{name}")
        if role == "x_best" and names["policy.zip"] != task["policy_sha256"]:
            raise ValueError("Frozen evaluation identifies a different policy")
        if role == "y_best" and any(sha256(folder/name) != expected for name, expected in y["sha256"].items()):
            raise ValueError("Preserved Y checkpoint differs from its manifest")
        output = f"checkpoints/{role}"
        for name in ("metadata.json", *names):
            files[f"{output}/{name}"] = folder/name
        roles[role] = {"path": output, "training_steps": metadata["training_steps"], "policy_sha256": names["policy.zip"]}

    for kind in ("best", "typical_failure"):
        for path in (final/"videos"/kind).iterdir():
            if path.is_file():
                files[f"results/x/videos/{kind}/{path.name}"] = path
    files["results/x/challenge_30/evaluation.json"] = final/"challenge_30/evaluation.json"
    report = (final/"REPORT.md").read_text().replace("(checkpoints/challenge_30)", "(../../checkpoints/x_best)")
    report = report.replace(f"(../arms/v4/checkpoints/{latest.name})", "(../../checkpoints/x_latest)")
    files["results/x/REPORT.md"] = report.encode()
    for name in ("protocol.json", "job.json", "amendment.json"):
        files[f"results/x/{name}"] = run/name
    files["results/y/manifest.json"] = y_run/"manifest.json"
    files["results/y/y_best_policy.mp4"] = y_run/"preview/y_best_policy.mp4"
    # Result pointers are relative to the archive root. The signed checkpoint
    # metadata and the original fixtures retain their source bytes for validation.
    paths = {str(best): "checkpoints/x_best", str(latest): "checkpoints/x_latest",
             str(y_run/"checkpoint"): "checkpoints/y_best",
             str(y_run/"manifest.json"): "results/y/manifest.json",
             str(final): "results/x", str(repo)+"/": ""}
    for name, file in files.items():
        if name.startswith("results/") and name.endswith(".json"):
            contents = file.read_text()
            for old, new in paths.items():
                contents = contents.replace(old, new)
            files[name] = contents.encode()
    sources = {key: git(repo/key, "rev-parse", "HEAD").decode().strip() for key in (".", *MODULES)}
    # Playground can download this extra asset checkout separately from the two gitlinks.
    if (repo/MENAGERIE/".git").exists():
        sources[MENAGERIE] = git(repo/MENAGERIE, "rev-parse", "HEAD").decode().strip()
    files["HANDOFF.json"] = (json.dumps(dict(source_commits=sources, checkpoints=roles,
        guide=f"{EXPERIMENT}/GPU_HANDOFF.md", result_paths_relative_to="archive_root",
        historical_sources="Original policy metadata and test fixtures retain historical source paths as provenance."), indent=2) + "\n").encode()
    validate_names(files)
    return files


def validate_names(names):
    paths = set(names)
    if not REQUIRED <= paths:
        raise ValueError(f"Missing sources or fixtures: {sorted(REQUIRED-paths)}")
    if {name for name in paths if PurePosixPath(name).name == "policy.zip"} != POLICIES:
        raise ValueError("Expected exactly the two selected X policies and one preserved Y policy")


def add_bytes(archive, name, data, mode=0o644):
    entry = tarfile.TarInfo(f"{PREFIX}/{name}")
    entry.size = len(data)
    entry.mode = mode
    archive.addfile(entry, io.BytesIO(data))


def verify(path):
    """Stream all members; reject duplicate, nonportable, unlisted or damaged entries."""
    hashes = {}
    expected = None
    with tarfile.open(path, "r|gz") as archive:
        for entry in archive:
            relative = PurePosixPath(entry.name)
            if (not entry.isfile() or len(relative.parts) < 2 or relative.parts[0] != PREFIX
                    or ".." in relative.parts or str(relative) != entry.name):
                raise ValueError(f"Nonportable archive member: {entry.name}")
            name = str(relative.relative_to(PREFIX))
            if name in hashes or (name == "SHA256SUMS" and expected is not None):
                raise ValueError(f"Duplicate archive member: {name}")
            stream = archive.extractfile(entry)
            if name == "SHA256SUMS":
                expected = {}
                for line in stream:
                    checksum, key = line.decode().rstrip("\n").split("  ", 1)
                    if key in expected:
                        raise ValueError(f"Duplicate checksum for {key}")
                    expected[key] = checksum
                continue
            checksum = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
            hashes[name] = checksum.hexdigest()
    if hashes != expected:
        raise ValueError("File list or content differs from SHA256SUMS")
    validate_names(hashes)
    return len(hashes)


def build(repo, run, y_run, output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_name(output.name+".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Choose a new output path: {output}")
    files = collect(repo, run, y_run)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp)/output.name
        sums = []
        with tarfile.open(staged, "w:gz", compresslevel=4) as archive:
            for name, source in sorted(files.items()):
                if isinstance(source, bytes):
                    add_bytes(archive, name, source)
                    checksum = hashlib.sha256(source).hexdigest()
                else:
                    entry = tarfile.TarInfo(f"{PREFIX}/{name}")
                    entry.size = source.stat().st_size
                    entry.mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
                    with source.open("rb") as stream:
                        archive.addfile(entry, stream)
                    checksum = sha256(source)
                sums.append(f"{checksum}  {name}\n")
            add_bytes(archive, "SHA256SUMS", "".join(sums).encode())
        count = verify(staged)
        checksum = sha256(staged)
        staged_sum = Path(temp)/sidecar.name
        staged_sum.write_text(f"{checksum}  {output.name}\n")
        os.link(staged, output)
        os.link(staged_sum, sidecar)
    print(f"{count} verified files; {output.stat().st_size} bytes; SHA256 {checksum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("build")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--run", type=Path, default=RUN)
    export.add_argument("--y-run", type=Path, default=Y_RUN)
    check = commands.add_parser("verify")
    check.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build(REPO, args.run, args.y_run, args.output)
    else:
        print(f"{verify(args.archive)} verified files")


if __name__ == "__main__":
    main()
