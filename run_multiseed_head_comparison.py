"""Multi-seed gpt_classical vs gpt_quantum: train, freeze, evaluate on Tokyo.

Purpose (Shiva review comment 10): repeat the LFM2.5-350M classical-head vs
quantum-head comparison over several *training* seeds so training variance can
be reported (mean +/- SD) instead of a single seed.

What changes between seeds: only ``--seed`` passed to ``train_modern_lora.py``
(LoRA/head initialisation, dropout, batch order). What does NOT change: the
development split (``dev_split_seed100003``), labels, phase masks, windows, LoRA
config, optimiser, 100-epoch budget, final-epoch checkpoint rule, Tokyo pool.
All of this is re-verified by SHA-256 before training and after every run.

Checkpointing / resume
----------------------
* ``train_modern_lora.py`` already saves a full checkpoint (adapter, head,
  optimiser, scheduler) after every epoch. If the run is interrupted (Ctrl+C,
  power, crash), simply run the same command again: each unfinished
  (seed, head) is resumed from its last completed epoch into a new
  ``__attemptNN`` directory; finished ones are skipped.
* After each (seed, head) finishes, the final checkpoint is hashed and frozen
  (``frozen_checkpoint.json``) *before* Tokyo is opened, then evaluated once.
* When both heads of a seed are evaluated, ``seedNNN/seed.complete.json`` is
  written: the per-seed checkpoint record.
* Old per-epoch checkpoints are pruned while training (keep the newest
  ``--keep-epoch-checkpoints``, default 2) to avoid ~15 GB per run.

Tokyo stays evaluation-only: no Tokyo number is read by training, checkpoint
choice (always the final epoch), or any setting in this script.

Speed on one RTX 4090
---------------------
Training uses batch size 1 (a method setting we must not change), so a single
run keeps the GPU mostly idle. The safe speed-up is to run several (seed, head)
trainings at the same time (``--parallel N``); each still trains exactly as
before. ``benchmark`` measures which N is fastest on your machine in ~10 min.
cuDNN autotuning is also enabled (kernel choice only). Not used on purpose:
larger batches / different accumulation (would change the method), TF32 (would
lower the precision of the fp32 quantum statevector), torch.compile (no Triton
on Windows). Tokyo evaluation always runs alone, so latency stays comparable.

Usage (lab machine, from the repo root)
---------------------------------------
    # 0) see what will run, without running anything
    python run_multiseed_head_comparison.py run --dry-run

    # 1) ~10 min: find the fastest number of concurrent trainings
    python run_multiseed_head_comparison.py benchmark

    # 2) run everything with the recommended N (e.g. 4)
    python run_multiseed_head_comparison.py run --device cuda --parallel 4

    # 3) progress at any time (from another terminal)
    python run_multiseed_head_comparison.py status

    # 4) after all seeds: mean +/- SD table
    python reports/summarize_multiseed_head_comparison.py

Interrupted? Run the same command again: every unfinished run resumes from its
last completed epoch.
"""

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Frozen protocol (copied from the seed-100003 run manifests in
# data/processed/lora_training/four_model_100epoch_v1/lfm2_5_350m_{classical,quantum})
# ----------------------------------------------------------------------------
MODEL_KEY = "lfm2_5_350m"
MODEL_REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"
REFERENCE_SHA256 = {
    "train.pkl": "dae037d4f34c428747763cedb4f2ae9cf1e047b3bceb042638325e188af2a937",
    "validation.pkl": "6d53da7aa0a5c0c994977eb5eb2349fb5f3b2e66c17d1b3f8d55cafdf856d060",
    "split.manifest.json": "a6c5a4c910b894d7dca334cbc7a652f1c468f98c8ba4157a5a21154cab66a109",
}
TOKYO_POOL_SHA256 = "800f1a444fcf1610720b83f6feadf0743fb2acaca81ed8d251f264fad012f27b"

# Seed 100003 already exists (four_model_100epoch_v1) and is reused by the
# summary script; these four are the additional training seeds.
DEFAULT_SEEDS = (200003, 300003, 400003, 500003)
HEADS = ("classical", "quantum")
ROLE = {"classical": "gpt_classical", "quantum": "gpt_quantum"}
EPOCHS = 100

COMMON_TRAIN_ARGS = [
    "--model-key", MODEL_KEY,
    "--dtype", "float16",
    "--rank", "128",
    "--alpha", "32",
    "--dropout", "0.05",
    "--epochs", str(EPOCHS),
    "--sequence-length", "20",
    "--sample-step", "20",
    "--state-feature-dim", "256",
    "--learning-rate", "1e-4",
    "--weight-decay", "1e-4",
    "--grad-accum-steps", "32",
    "--run-purpose", "slm_bbr_modern_backbone_extension",
]
HEAD_TRAIN_ARGS = {
    "classical": ["--head-type", "classical"],
    # 8 qubits, depth 1, RY encoding + trainable RY + CNOT ring, LayerNorm,
    # T = 4, angle scale pi. torch backend = exact statevector (verified equal
    # to the Qiskit path to 1e-5); Tokyo evaluation below uses the Qiskit path,
    # exactly as for seed 100003.
    "quantum": [
        "--head-type", "quantum",
        "--n-qubits", "8",
        "--quantum-depth", "1",
        "--quantum-ansatz", "trainable_ry_layers",
        "--head-input-layernorm",
        "--bottleneck-temperature", "4",
        "--quantum-angle-scale", "pi",
        "--quantum-backend", "torch",
    ],
}
EXPECTED_QUANTUM_HEAD = {
    "n_qubits": 8,
    "depth": 1,
    "ansatz": "trainable_ry_layers",
    "input_layernorm": True,
    "temperature": 4.0,
    "angle_scale": "pi",
}

DEFAULT_TRAIN_ROOT = Path("data/processed/lora_training/multiseed_head_comparison_v1")
DEFAULT_EVAL_ROOT = Path("data/processed/evaluation/multiseed_head_comparison_v1")
DEFAULT_SPLIT_DIR = Path("data/processed/slm_bbr_w10_eq2_3_dev_split_seed100003")
DEFAULT_TOKYO_POOL = Path("data/processed/evaluation/tokyo_8model_comparison_v1/tokyo.pkl")
DEFAULT_FREEZE_MANIFEST = Path(
    "data/processed/evaluation/tokyo_8model_comparison_v1/frozen_checkpoints.manifest.json"
)
UP_GAINS = (1.05, 1.10, 1.15, 1.20, 1.25)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


DRY_RUN = False
_LOG_LOCK = threading.Lock()
_CHILDREN = set()
_CHILDREN_LOCK = threading.Lock()


def log_event(train_root, **event):
    event = {"time": now(), **event}
    if DRY_RUN:
        print("[multiseed][dry-run] " + " ".join("{}={}".format(k, v) for k, v in event.items() if k != "time"))
        return
    with _LOG_LOCK:
        train_root.mkdir(parents=True, exist_ok=True)
        with open(train_root / "runner_log.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        print("[multiseed] " + " ".join("{}={}".format(k, v) for k, v in event.items()), flush=True)


def terminate_children():
    with _CHILDREN_LOCK:
        children = list(_CHILDREN)
    for process in children:
        if process.poll() is None:
            process.terminate()
    for process in children:
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()


def seed_dir(root, seed):
    return Path(root) / "seed{}".format(seed)


def base_name(head):
    return "{}_{}".format(MODEL_KEY, head)


# ----------------------------------------------------------------------------
# run-directory state (supports resume across several attempts)
# ----------------------------------------------------------------------------
def attempt_dirs(train_root, seed, head):
    """All attempt directories for one (seed, head), oldest first."""
    parent = seed_dir(train_root, seed)
    first = parent / base_name(head)
    found = [first] if first.is_dir() else []
    index = 2
    while True:
        candidate = parent / "{}__attempt{:02d}".format(base_name(head), index)
        if not candidate.exists():
            break
        found.append(candidate)
        index += 1
    return found


def next_attempt_dir(train_root, seed, head):
    parent = seed_dir(train_root, seed)
    first = parent / base_name(head)
    if not first.exists():
        return first
    index = 2
    while True:
        candidate = parent / "{}__attempt{:02d}".format(base_name(head), index)
        if not candidate.exists():
            return candidate
        index += 1


def completed_epochs(run_dir):
    progress = run_dir / "progress.manifest.json"
    if not progress.is_file():
        return 0
    try:
        return int(read_json(progress).get("completed_epochs", 0))
    except (ValueError, json.JSONDecodeError):
        return 0


def training_state(train_root, seed, head):
    """Return ('trained', dir) | ('partial', dir, epochs) | ('new',)."""
    attempts = attempt_dirs(train_root, seed, head)
    for run_dir in reversed(attempts):
        if (run_dir / "run.manifest.json").is_file():
            return ("trained", run_dir)
    resumable = [
        (completed_epochs(run_dir), run_dir)
        for run_dir in attempts
        if completed_epochs(run_dir) > 0 and resume_checkpoint_complete(run_dir)
    ]
    if resumable:
        epochs, run_dir = max(resumable, key=lambda item: (item[0], str(item[1])))
        return ("partial", run_dir, epochs)
    return ("new",)


def resume_checkpoint_complete(run_dir):
    try:
        checkpoint = run_dir / read_json(run_dir / "progress.manifest.json")["checkpoint"]
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return False
    return all(
        (checkpoint / name).is_file()
        for name in ("adapter/adapter_config.json", "task_modules.pt", "optimizer.pt")
    )


# ----------------------------------------------------------------------------
# pre-flight checks (fail in seconds, not after 12 hours)
# ----------------------------------------------------------------------------
def preflight(args):
    problems = []
    for name, expected in REFERENCE_SHA256.items():
        path = args.split_dir / name
        if not path.is_file():
            problems.append("missing {}".format(path))
        elif sha256(path) != expected:
            problems.append("{} checksum differs from the seed-100003 runs".format(path))
    if not args.skip_eval:
        if not args.tokyo_pool.is_file():
            problems.append("missing Tokyo pool {}".format(args.tokyo_pool))
        elif sha256(args.tokyo_pool) != TOKYO_POOL_SHA256:
            problems.append("Tokyo pool checksum differs from the frozen protocol")
        if not args.freeze_manifest.is_file():
            problems.append("missing freeze manifest {}".format(args.freeze_manifest))
    if not (REPO / "train_modern_lora.py").is_file():
        problems.append("run this script from the lm-bbr-starlink repo root")
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.strip()
        if dirty:
            print(
                "[multiseed] WARNING: git working tree has uncommitted changes; "
                "run manifests will record git_dirty=true. Commit first if possible.",
                flush=True,
            )
    except (OSError, subprocess.CalledProcessError):
        print("[multiseed] WARNING: could not read git status", flush=True)
    if problems:
        raise SystemExit("Pre-flight failed:\n  - " + "\n  - ".join(problems))


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
# Runs train_modern_lora.py unchanged, after enabling cuDNN autotuning (kernel
# choice only; no change to the model, data, precision or optimiser).
_TRAIN_BOOTSTRAP = (
    "import runpy, sys, torch; "
    "torch.backends.cudnn.benchmark = True; "
    "sys.argv = sys.argv[1:]; "
    "runpy.run_path(sys.argv[0], run_name='__main__')"
)


def build_train_command(args, seed, head, output_dir, resume_from=None):
    if getattr(args, "cudnn_benchmark", True):
        command = [sys.executable, "-u", "-c", _TRAIN_BOOTSTRAP, str(REPO / "train_modern_lora.py")]
    else:
        command = [sys.executable, "-u", str(REPO / "train_modern_lora.py")]
    command += COMMON_TRAIN_ARGS + HEAD_TRAIN_ARGS[head]
    command += [
        "--seed", str(seed),
        "--device", args.device,
        "--split-dir", str(args.split_dir),
        "--output-dir", str(output_dir),
    ]
    if resume_from is not None:
        command += ["--resume-from-run", str(resume_from)]
    return command


class EpochCheckpointPruner(threading.Thread):
    """Delete old epoch_checkpoints/epoch_NNNN while training runs.

    Keeps the newest ``keep`` epochs plus whatever progress.manifest.json
    currently points to, so resume always has a complete checkpoint.
    """

    def __init__(self, run_dir, keep, interval=60.0):
        super().__init__(daemon=True)
        self.run_dir = Path(run_dir)
        self.keep = keep
        self.interval = interval
        self.stop_event = threading.Event()

    def prune_once(self):
        root = self.run_dir / "epoch_checkpoints"
        if self.keep <= 0 or not root.is_dir():
            return
        epochs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("epoch_"))
        protected = set(p.name for p in epochs[-self.keep:])
        try:
            referenced = read_json(self.run_dir / "progress.manifest.json").get("checkpoint", "")
            protected.add(Path(referenced).name)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        for path in epochs:
            if path.name not in protected:
                shutil.rmtree(path, ignore_errors=True)

    def run(self):
        while not self.stop_event.wait(self.interval):
            self.prune_once()
        self.prune_once()

    def stop(self):
        self.stop_event.set()
        self.join(timeout=30)


def run_logged(command, log_path, env, echo=True, on_line=None):
    """Run a child process, tee its output to ``log_path`` (and stdout if echo)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write("\n# {}  {}\n".format(now(), " ".join(command)))
        log.flush()
        process = subprocess.Popen(
            command, cwd=REPO, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        with _CHILDREN_LOCK:
            _CHILDREN.add(process)
        try:
            for line in process.stdout:
                log.write(line)
                if echo:
                    sys.stdout.write(line)
                if on_line is not None:
                    on_line(line)
            return process.wait()
        except KeyboardInterrupt:
            terminate_children()
            raise
        finally:
            with _CHILDREN_LOCK:
                _CHILDREN.discard(process)


def child_env(args, concurrent_jobs=1):
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if getattr(args, "local_model_root", None):
        env["LM_BBR_LOCAL_MODEL_ROOT"] = str(Path(args.local_model_root).expanduser())
    if concurrent_jobs > 1:
        # Avoid CPU thread oversubscription when several trainings share the CPU.
        threads = str(max(1, (os.cpu_count() or 8) // concurrent_jobs))
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(name, threads)
    return env


def train(args, seed, head, concurrent_jobs=1):
    state = training_state(args.train_root, seed, head)
    if state[0] == "trained":
        return state[1]
    resume_from = state[1] if state[0] == "partial" else None
    if state[0] == "partial" and state[2] >= EPOCHS:
        raise RuntimeError(
            "{} reached epoch {} but never wrote run.manifest.json; inspect train.log "
            "(the final save/reload check failed) before retrying".format(state[1], state[2]))
    output_dir = next_attempt_dir(args.train_root, seed, head)
    command = build_train_command(args, seed, head, output_dir, resume_from)
    log_event(
        args.train_root, event="train_start", seed=seed, head=head,
        output_dir=output_dir.as_posix(),
        resume_from=(resume_from.as_posix() if resume_from else None),
        resume_epoch=(state[2] if state[0] == "partial" else 0),
    )
    if args.dry_run:
        print("    " + " ".join(command))
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    pruner = EpochCheckpointPruner(output_dir, args.keep_epoch_checkpoints)
    pruner.start()
    started = time.time()
    try:
        code = run_logged(
            command, output_dir / "train.log", child_env(args, concurrent_jobs),
            echo=(concurrent_jobs == 1),
        )
    finally:
        pruner.stop()
    if code != 0:
        log_event(args.train_root, event="train_failed", seed=seed, head=head, returncode=code)
        raise RuntimeError("training failed for seed {} head {} (exit {}); see {}".format(
            seed, head, code, output_dir / "train.log"))
    log_event(
        args.train_root, event="train_done", seed=seed, head=head,
        hours=round((time.time() - started) / 3600.0, 2),
    )
    return output_dir


def verify_run(run_dir, seed, head):
    """Comparability checks on the finished run manifest."""
    manifest = read_json(run_dir / "run.manifest.json")
    checks = {
        "seed": manifest.get("seed") == seed,
        "head_type": manifest.get("head_type") == head,
        "model_key": manifest.get("model_key") == MODEL_KEY,
        "model_revision": manifest.get("model_revision") == MODEL_REVISION,
        "epochs": manifest.get("epochs") == EPOCHS,
        "tokyo_isolation": manifest.get("tokyo_isolation") == "PASS",
        "checkpoint_reload": manifest.get("checkpoint_reload") == "PASS",
        "train_pool_sha256": manifest.get("train_pool_sha256") == REFERENCE_SHA256["train.pkl"],
        "validation_pool_sha256": manifest.get("validation_pool_sha256") == REFERENCE_SHA256["validation.pkl"],
        "split_manifest_sha256": manifest.get("split_manifest_sha256") == REFERENCE_SHA256["split.manifest.json"],
        "lora_rank": (manifest.get("lora_config") or {}).get("rank") == 128,
        "not_stopped_early": not (manifest.get("early_stopping") or {}).get("stopped_early", False),
    }
    if head == "quantum":
        config = manifest.get("head_config") or {}
        for key, value in EXPECTED_QUANTUM_HEAD.items():
            checks["quantum_" + key] = config.get(key) == value
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise RuntimeError("Run {} failed comparability checks: {}".format(run_dir, failed))
    return manifest


# ----------------------------------------------------------------------------
# freeze + Tokyo evaluation (subprocess so GPU memory is released)
# ----------------------------------------------------------------------------
def eval_dir(args, seed, head):
    return seed_dir(args.eval_root, seed) / base_name(head)


def freeze(args, seed, head, run_dir):
    """Hash the final checkpoint before Tokyo is opened (the per-run checkpoint record)."""
    out_dir = eval_dir(args, seed, head)
    if (out_dir / "frozen_checkpoint.json").is_file():
        return out_dir
    manifest = verify_run(run_dir, seed, head)
    checkpoint = run_dir / "checkpoint"
    frozen = {
        "status": "checkpoint_frozen_before_tokyo",
        "frozen_at": now(),
        "checkpoint_rule": "final_epoch_{}_fixed_before_tokyo".format(EPOCHS),
        "seed": seed,
        "model_role": ROLE[head],
        "head_type": head,
        "run_dir": run_dir.as_posix(),
        "run_manifest_sha256": sha256(run_dir / "run.manifest.json"),
        "checkpoint_files_sha256": {
            name: sha256(checkpoint / name)
            for name in (
                "adapter/adapter_config.json",
                "adapter/adapter_model.safetensors",
                "task_modules.pt",
                "training_state.json",
            )
            if (checkpoint / name).is_file()
        },
        "final_validation": (manifest.get("final_validation_criteria") or {}).get("observed"),
        "git_commit": manifest.get("git_commit"),
        "git_dirty": manifest.get("git_dirty"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "frozen_checkpoint.json", frozen)
    log_event(args.train_root, event="checkpoint_frozen", seed=seed, head=head)
    return out_dir


def evaluate(args, seed, head, run_dir):
    """Tokyo evaluation of a frozen checkpoint. Run alone on the GPU so that the
    latency numbers stay comparable with the seed-100003 evaluation."""
    out_dir = eval_dir(args, seed, head)
    if (out_dir / "result.manifest.json").is_file():
        return out_dir
    if args.dry_run:
        print("    evaluate-one --run-dir {} --output-dir {}".format(run_dir, out_dir))
        return None
    freeze(args, seed, head, run_dir)
    command = [
        sys.executable, "-u", str(Path(__file__).resolve()), "evaluate-one",
        "--run-dir", str(run_dir), "--output-dir", str(out_dir),
        "--seed", str(seed), "--head", head,
        "--tokyo-pool", str(args.tokyo_pool),
        "--freeze-manifest", str(args.freeze_manifest),
        "--device", args.device,
    ]
    log_event(args.train_root, event="tokyo_eval_start", seed=seed, head=head)
    code = run_logged(command, out_dir / "evaluate.log", child_env(args))
    if code != 0:
        raise RuntimeError("Tokyo evaluation failed for seed {} head {}".format(seed, head))
    summary = quick_metrics(out_dir / "predictions.csv")
    log_event(args.train_root, event="tokyo_eval_done", seed=seed, head=head, **summary)
    return out_dir


def evaluate_one(args):
    """Single frozen-checkpoint Tokyo evaluation (runs in its own process)."""
    import pickle

    import torch

    from config import cfg
    from evaluate_tokyo_gpt_quantum import checkpoint_files_sha256, load_policy_from_run
    from evaluate_tokyo_models import write_records
    from plm_special.data.dataset import ExperienceDataset
    from utils.tokyo_evaluation import evaluate_frozen_policy

    frozen = read_json(args.output_dir / "frozen_checkpoint.json")
    checkpoint_dir = args.run_dir / "checkpoint"
    if checkpoint_files_sha256(checkpoint_dir) != frozen["checkpoint_files_sha256"]:
        raise ValueError("Checkpoint changed after it was frozen: {}".format(checkpoint_dir))
    if (args.output_dir / "predictions.csv").exists():
        raise ValueError("Refusing to overwrite existing Tokyo predictions")
    freeze_manifest = read_json(args.freeze_manifest)
    if freeze_manifest.get("tokyo_opened") is not True:
        raise ValueError("Tokyo pool has not passed the frozen protocol gate")
    if sha256(args.tokyo_pool) != TOKYO_POOL_SHA256:
        raise ValueError("Tokyo pool checksum does not match the frozen protocol")
    with open(args.tokyo_pool, "rb") as stream:
        pool = pickle.load(stream)
    if pool.metadata.get("split_role") != "held_out_test":
        raise ValueError("Pool is not marked held_out_test")

    local_model_root = os.environ.get("LM_BBR_LOCAL_MODEL_ROOT", cfg.local_model_root)
    policy, run_manifest = load_policy_from_run(args.run_dir, args.device, local_model_root)
    if run_manifest.get("seed") != args.seed or run_manifest.get("head_type") != args.head:
        raise ValueError("Run manifest seed/head does not match the request")
    dataset = ExperienceDataset(
        pool, gamma=1.0, scale=1000,
        max_length=run_manifest["sequence_length"],
        sample_step=run_manifest["sample_step"],
    )
    metrics, records = evaluate_frozen_policy(policy, dataset, pool, args.device, latency_warmup_batches=5)
    records_path = args.output_dir / "predictions.csv"
    write_records(records_path, records)
    write_json(args.output_dir / "result.manifest.json", {
        "status": "held_out_tokyo_evaluation_completed",
        "evidence_class": "measured_final",
        "experiment": "multiseed_head_comparison_v1",
        "seed": args.seed,
        "model_key": run_manifest["model_key"],
        "model_role": ROLE[args.head],
        "head_type": run_manifest["head_type"],
        "head_config": run_manifest["head_config"],
        "model_id": run_manifest["model_id"],
        "model_revision": run_manifest["model_revision"],
        "run_dir": args.run_dir.as_posix(),
        "checkpoint_files_sha256": frozen["checkpoint_files_sha256"],
        "tokyo_pool_sha256": TOKYO_POOL_SHA256,
        "sequence_length": run_manifest["sequence_length"],
        "sample_step": run_manifest["sample_step"],
        "inference_only": True,
        "phase_mask_applied_before_loss_and_argmax": True,
        "metrics": metrics,
        "predictions": records_path.as_posix(),
    })
    print("accuracy", metrics["accuracy"], "macro_phase", metrics["macro_phase_accuracy"])


def quick_metrics(predictions_csv):
    """BW_UP accuracy / macro-F1 / distinct predictions from predictions.csv."""
    with open(predictions_csv, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    correct = sum(int(r["correct"]) for r in rows)
    up = [(round(float(r["target_gain"]), 2), round(float(r["predicted_gain"]), 2))
          for r in rows if r["phase"] == "BW_UP"]
    f1s = []
    for gain in UP_GAINS:
        tp = sum(1 for y, p in up if y == gain and p == gain)
        fp = sum(1 for y, p in up if y != gain and p == gain)
        fn = sum(1 for y, p in up if y == gain and p != gain)
        if tp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "accuracy": round(correct / len(rows), 4),
        "bw_up_accuracy": round(sum(1 for y, p in up if y == p) / len(up), 4),
        "bw_up_macro_f1": round(sum(f1s) / len(f1s), 4),
        "bw_up_distinct_predictions": len(set(p for _, p in up)),
    }


def write_seed_record(args, seed):
    """Per-seed checkpoint record once both heads are trained and evaluated."""
    record = {"seed": seed, "completed_at": now(), "heads": {}}
    for head in HEADS:
        out_dir = eval_dir(args, seed, head)
        if not (out_dir / "result.manifest.json").is_file():
            return False
        frozen = read_json(out_dir / "frozen_checkpoint.json")
        record["heads"][ROLE[head]] = {
            "run_dir": frozen["run_dir"],
            "checkpoint_files_sha256": frozen["checkpoint_files_sha256"],
            "final_validation": frozen.get("final_validation"),
            "tokyo": quick_metrics(out_dir / "predictions.csv"),
            "tokyo_result": (out_dir / "result.manifest.json").as_posix(),
        }
    write_json(seed_dir(args.train_root, seed) / "seed.complete.json", record)
    log_event(args.train_root, event="seed_complete", seed=seed)
    return True


# ----------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------
def _training_jobs(args):
    """(seed, head) pairs, interleaved so a parallel pool mixes both heads."""
    return [(seed, head) for seed in args.seeds for head in args.heads]


def _train_and_freeze(args, seed, head, concurrent_jobs):
    run_dir = train(args, seed, head, concurrent_jobs=concurrent_jobs)
    if run_dir is None:  # dry run
        return None
    verify_run(run_dir, seed, head)
    if not args.skip_eval:
        freeze(args, seed, head, run_dir)
    return run_dir


def _report_failures(args, failures):
    command_status(args)
    if failures:
        print("Failures:")
        for seed, head, error in failures:
            print("  seed {} {}: {}".format(seed, head, error))
        return 1
    return 0


def _run_sequential(args):
    failures = []
    for seed in args.seeds:
        for head in args.heads:
            try:
                run_dir = _train_and_freeze(args, seed, head, 1)
                if run_dir is not None and not args.skip_eval:
                    evaluate(args, seed, head, run_dir)
            except KeyboardInterrupt:
                log_event(args.train_root, event="interrupted", seed=seed, head=head)
                print("\nInterrupted. Re-run the same command to resume from the last epoch.")
                return 130
            except Exception as error:  # noqa: BLE001
                log_event(args.train_root, event="error", seed=seed, head=head, error=str(error))
                failures.append((seed, head, str(error)))
                if not args.continue_on_error:
                    raise
        if not args.dry_run and not args.skip_eval:
            write_seed_record(args, seed)
    return _report_failures(args, failures)


def _progress_line(args):
    parts = []
    for seed, head in _training_jobs(args):
        state = training_state(args.train_root, seed, head)
        if state[0] == "trained":
            text = "done"
        elif state[0] == "partial":
            text = "{}/{}".format(state[2], EPOCHS)
        else:
            text = "-"
        parts.append("{}:{}={}".format(seed, head[0].upper(), text))
    return "[multiseed] {}  {}".format(_dt.datetime.now().strftime("%H:%M"), "  ".join(parts))


def _run_parallel(args):
    """Up to ``args.parallel`` trainings share the GPU. Batch size is 1, so one
    training leaves most of the RTX 4090 idle; running several at once raises
    total throughput without changing any training setting. Tokyo evaluation is
    deferred and run one model at a time afterwards (clean latency numbers)."""
    import queue

    jobs = queue.Queue()
    for job in _training_jobs(args):
        jobs.put(job)
    failures, lock = [], threading.Lock()
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            try:
                seed, head = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                _train_and_freeze(args, seed, head, args.parallel)
            except Exception as error:  # noqa: BLE001
                log_event(args.train_root, event="error", seed=seed, head=head, error=str(error))
                with lock:
                    failures.append((seed, head, str(error)))
                if not args.continue_on_error:
                    stop.set()

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(args.parallel)]
    for thread in workers:
        thread.start()
        time.sleep(0 if args.dry_run else 20)  # stagger model loading
    try:
        last_report = 0.0
        while any(thread.is_alive() for thread in workers):
            time.sleep(1 if args.dry_run else 15)
            if not args.dry_run and time.time() - last_report >= args.progress_every * 60:
                print(_progress_line(args), flush=True)
                last_report = time.time()
    except KeyboardInterrupt:
        stop.set()
        terminate_children()
        log_event(args.train_root, event="interrupted")
        print("\nInterrupted. Re-run the same command to resume every run from its last epoch.")
        return 130
    if failures and not args.continue_on_error:
        return _report_failures(args, failures)
    if args.skip_eval:
        return _report_failures(args, failures)

    log_event(args.train_root, event="tokyo_eval_phase_start")
    for seed in args.seeds:
        for head in args.heads:
            if any(f[0] == seed and f[1] == head for f in failures):
                continue
            state = training_state(args.train_root, seed, head)
            if args.dry_run:
                print("    evaluate-one seed={} head={}".format(seed, head))
                continue
            if state[0] != "trained":
                continue
            try:
                evaluate(args, seed, head, state[1])
            except KeyboardInterrupt:
                terminate_children()
                return 130
            except Exception as error:  # noqa: BLE001
                log_event(args.train_root, event="error", seed=seed, head=head, error=str(error))
                failures.append((seed, head, str(error)))
        if not args.dry_run:
            write_seed_record(args, seed)
    return _report_failures(args, failures)


def command_run(args):
    global DRY_RUN
    DRY_RUN = args.dry_run
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")
    if not args.dry_run:
        preflight(args)
    log_event(args.train_root, event="runner_start", seeds=list(args.seeds),
              heads=list(args.heads), device=args.device, parallel=args.parallel,
              cudnn_benchmark=args.cudnn_benchmark, dry_run=args.dry_run)
    if args.parallel == 1:
        return _run_sequential(args)
    return _run_parallel(args)


# ----------------------------------------------------------------------------
# throughput benchmark: how many concurrent trainings are fastest on this GPU?
# ----------------------------------------------------------------------------
def _gpu_snapshot():
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
        used, total, util = (float(v) for v in output.split(","))
        return used, total, util
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def command_benchmark(args):
    """Short timed runs (never Tokyo, never the real run directories)."""
    preflight_args = argparse.Namespace(split_dir=args.split_dir, skip_eval=True,
                                        tokyo_pool=None, freeze_manifest=None)
    preflight(preflight_args)
    bench_root = Path(args.bench_root)
    rows = []
    for level in args.levels:
        heads = [HEADS[i % 2] for i in range(level)]
        stamps = [dict() for _ in range(level)]
        peak = {"memory": 0.0, "util": []}
        done = threading.Event()

        def sampler():
            while not done.wait(2):
                snap = _gpu_snapshot()
                if snap:
                    peak["memory"] = max(peak["memory"], snap[0])
                    peak["total"] = snap[1]
                    peak["util"].append(snap[2])

        def launch(index):
            head = heads[index]
            output_dir = bench_root / "level{}".format(level) / "{}_{}".format(index, head)
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
            command = build_train_command(args, 900000 + index, head, output_dir)
            command += ["--epochs", "1", "--max-train-steps", str(args.steps),
                        "--max-validation-steps", "1"]

            def on_line(line):
                if line.startswith("train step "):
                    try:
                        step = int(line.split()[2])
                    except (IndexError, ValueError):
                        return
                    stamps[index][step] = time.time()

            code = run_logged(command, output_dir / "train.log", child_env(args, level),
                              echo=False, on_line=on_line)
            stamps[index]["exit"] = code

        print("[benchmark] {} concurrent training(s): {} ...".format(level, ", ".join(heads)), flush=True)
        threading.Thread(target=sampler, daemon=True).start()
        threads = [threading.Thread(target=launch, args=(i,)) for i in range(level)]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            terminate_children()
            return 130
        finally:
            done.set()
        rates, per_head = [], {}
        for index, stamp in enumerate(stamps):
            steps = sorted(k for k in stamp if isinstance(k, int) and k >= 25)
            if stamp.get("exit") != 0 or len(steps) < 2:
                print("[benchmark]   run {} failed or too short; see {}".format(
                    index, bench_root / "level{}".format(level)), flush=True)
                continue
            rate = (steps[-1] - steps[0]) / (stamp[steps[-1]] - stamp[steps[0]])
            rates.append(rate)
            per_head.setdefault(heads[index], []).append(rate)
        if len(rates) != level:
            rows.append((level, None, None, None, peak))
            continue
        rows.append((level, sum(rates), per_head, peak.get("total"), peak))
        if not args.keep_bench_dirs:
            shutil.rmtree(bench_root / "level{}".format(level), ignore_errors=True)
        if peak["memory"] and peak.get("total") and peak["memory"] > 0.9 * peak["total"]:
            print("[benchmark] GPU memory nearly full; not testing higher levels.", flush=True)
            break

    base = next((r[1] for r in rows if r[0] == 1 and r[1]), None)
    print("\n{:>9} {:>14} {:>9} {:>16} {:>16} {:>11} {:>9}".format(
        "parallel", "windows/s", "speed-up", "classical w/s", "quantum w/s", "peak VRAM", "GPU util"))
    best = None
    for level, total, per_head, _, peak in rows:
        if total is None:
            print("{:>9} {:>14}".format(level, "failed"))
            continue
        util = sum(peak["util"]) / len(peak["util"]) if peak["util"] else float("nan")
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        print("{:>9} {:>14.2f} {:>9} {:>16.2f} {:>16.2f} {:>9.1f}GB {:>8.0f}%".format(
            level, total, "{:.2f}x".format(total / base) if base else "-",
            mean(per_head.get("classical", [])), mean(per_head.get("quantum", [])),
            peak["memory"] / 1024.0, util))
        if best is None or total > best[1] * 1.05:
            best = (level, total)
    if best:
        jobs = len(DEFAULT_SEEDS) * len(HEADS)
        windows = jobs * EPOCHS * 2400 * 1.1  # +~10% for the per-epoch validation pass
        print("\nRecommended: --parallel {}  (about {:.0f} h for {} runs x {} epochs; rough)".format(
            best[0], windows / best[1] / 3600.0, jobs, EPOCHS))
    return 0


def command_status(args):
    print("\n{:>8}  {:<10} {:<22} {}".format("seed", "head", "training", "Tokyo (acc / BW_UP acc / BW_UP F1 / distinct)"))
    for seed in args.seeds:
        for head in HEADS:
            state = training_state(args.train_root, seed, head)
            if state[0] == "trained":
                text = "done ({})".format(state[1].name)
            elif state[0] == "partial":
                text = "epoch {}/{}".format(state[2], EPOCHS)
            else:
                text = "not started"
            predictions = eval_dir(args, seed, head) / "predictions.csv"
            tokyo = "-"
            if predictions.is_file() and (eval_dir(args, seed, head) / "result.manifest.json").is_file():
                q = quick_metrics(predictions)
                tokyo = "{accuracy:.4f} / {bw_up_accuracy:.4f} / {bw_up_macro_f1:.4f} / {bw_up_distinct_predictions}".format(**q)
            print("{:>8}  {:<10} {:<22} {}".format(seed, head, text, tokyo))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p):
        p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
        p.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
        p.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)

    run = sub.add_parser("run", help="train + freeze + Tokyo-evaluate every (seed, head)")
    shared(run)
    run.add_argument("--heads", nargs="+", choices=HEADS, default=list(HEADS))
    run.add_argument("--device", default="cuda")
    run.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    run.add_argument("--tokyo-pool", type=Path, default=DEFAULT_TOKYO_POOL)
    run.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    run.add_argument("--local-model-root", help="sets LM_BBR_LOCAL_MODEL_ROOT for child processes")
    run.add_argument("--keep-epoch-checkpoints", type=int, default=2,
                     help="newest per-epoch checkpoints to keep while training (0 = keep all)")
    run.add_argument("--skip-eval", action="store_true", help="train only; evaluate later")
    run.add_argument("--continue-on-error", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    run.add_argument("--parallel", type=int, default=1,
                     help="trainings to run at the same time on the GPU (see the benchmark command)")
    run.add_argument("--progress-every", type=float, default=10.0,
                     help="minutes between progress lines in --parallel mode")
    run.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")

    bench = sub.add_parser("benchmark", help="time 1..N concurrent trainings (short runs, no Tokyo)")
    bench.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    bench.add_argument("--steps", type=int, default=150, help="training windows per benchmark run")
    bench.add_argument("--device", default="cuda")
    bench.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    bench.add_argument("--local-model-root")
    bench.add_argument("--bench-root", type=Path,
                       default=Path("data/processed/lora_training/multiseed_head_comparison_benchmark"))
    bench.add_argument("--keep-bench-dirs", action="store_true")
    bench.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")

    status = sub.add_parser("status", help="show progress")
    shared(status)

    one = sub.add_parser("evaluate-one", help=argparse.SUPPRESS)
    one.add_argument("--run-dir", type=Path, required=True)
    one.add_argument("--output-dir", type=Path, required=True)
    one.add_argument("--seed", type=int, required=True)
    one.add_argument("--head", choices=HEADS, required=True)
    one.add_argument("--tokyo-pool", type=Path, required=True)
    one.add_argument("--freeze-manifest", type=Path, required=True)
    one.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    os.chdir(REPO)
    args = parse_args(argv)
    if args.command == "run":
        return command_run(args)
    if args.command == "status":
        return command_status(args)
    if args.command == "benchmark":
        return command_benchmark(args)
    if args.command == "evaluate-one":
        evaluate_one(args)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
