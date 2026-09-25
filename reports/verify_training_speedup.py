"""Check the GPU speed-up on the real model: same results, how much faster.

Runs short trainings (2 epochs x 128 windows, seed 100003, real LFM2.5-350M,
real development split) with
  * baseline : the code before the speed-up (reports/speedup_baseline/),
  * baseline again : to measure run-to-run GPU noise,
  * candidate : the current code,
for gpt_classical and gpt_quantum, then compares losses, accuracies and the
saved weights, and reports windows per second. Tokyo is never read.

Run on the lab GPU (about 10-15 min), from the repo root:
    python reports/verify_training_speedup.py
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_multiseed_head_comparison import (  # noqa: E402
    COMMON_TRAIN_ARGS,
    DEFAULT_SPLIT_DIR,
    HEAD_TRAIN_ARGS,
)

BASELINE_FILES = ROOT / "reports" / "speedup_baseline"
CODE_ITEMS = ("config.py", "train_modern_lora.py", "plm_special", "utils")


def build_baseline_code(work):
    code = work / "baseline_code"
    if code.exists():
        shutil.rmtree(code)
    code.mkdir(parents=True)
    for item in CODE_ITEMS:
        source = ROOT / item
        if source.is_dir():
            shutil.copytree(source, code / item, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, code / item)
    for original in BASELINE_FILES.rglob("*.py"):
        target = code / original.relative_to(BASELINE_FILES)
        shutil.copy2(original, target)
    return code


def run(code_dir, head, output_dir, args):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    command = [sys.executable, "-u", str(code_dir / "train_modern_lora.py")]
    command += COMMON_TRAIN_ARGS + HEAD_TRAIN_ARGS[head]
    command += [
        "--seed", "100003", "--device", args.device,
        "--split-dir", str(Path(args.split_dir).resolve()),
        "--output-dir", str(output_dir),
        "--epochs", str(args.epochs),
        "--max-train-steps", str(args.steps),
        "--max-validation-steps", str(args.validation_steps),
    ]
    stamps = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train.log", "w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(command, cwd=code_dir, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   encoding="utf-8", errors="replace")
        epoch = 0
        for line in process.stdout:
            log.write(line)
            if line.startswith("epoch "):
                epoch += 1
            if line.startswith("train step ") and epoch == 1:
                try:
                    stamps[int(line.split()[2])] = time.time()
                except (IndexError, ValueError):
                    pass
        code = process.wait()
    if code != 0:
        raise SystemExit("training failed ({}); see {}".format(code, output_dir / "train.log"))
    steps = sorted(s for s in stamps if s >= 25)
    rate = (steps[-1] - steps[0]) / (stamps[steps[-1]] - stamps[steps[0]])
    return rate


def load_result(output_dir):
    import torch
    from safetensors.torch import load_file

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    scalars = []
    for entry in metrics["epochs"]:
        for split in ("train", "validation"):
            scalars += [entry[split]["loss"], entry[split]["accuracy"]]
    tensors = dict(torch.load(output_dir / "checkpoint" / "task_modules.pt",
                              map_location="cpu", weights_only=True))
    adapter = output_dir / "checkpoint" / "adapter" / "adapter_model.safetensors"
    tensors.update({"adapter." + k: v for k, v in load_file(str(adapter)).items()})
    return scalars, tensors


def difference(a, b):
    scalars_a, tensors_a = a
    scalars_b, tensors_b = b
    scalar = max(abs(x - y) for x, y in zip(scalars_a, scalars_b))
    weight = max(
        float((tensors_a[k].float() - tensors_b[k].float()).abs().max()) for k in tensors_a
    )
    return scalar, weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", nargs="+", default=["classical", "quantum"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--validation-steps", type=int, default=40)
    parser.add_argument("--work-dir", type=Path,
                        default=ROOT / "data/processed/lora_training/speedup_check")
    args = parser.parse_args()

    work = args.work_dir.resolve()
    baseline_code = build_baseline_code(work)
    rows, verdicts = [], []
    for head in args.heads:
        print("[verify] {}: baseline ...".format(head), flush=True)
        rate_base = run(baseline_code, head, work / head / "baseline", args)
        print("[verify] {}: baseline (repeat, noise level) ...".format(head), flush=True)
        run(baseline_code, head, work / head / "baseline_repeat", args)
        print("[verify] {}: candidate (new code) ...".format(head), flush=True)
        rate_new = run(ROOT, head, work / head / "candidate", args)

        base = load_result(work / head / "baseline")
        noise = difference(base, load_result(work / head / "baseline_repeat"))
        change = difference(base, load_result(work / head / "candidate"))
        ok = (change[0] <= max(10 * noise[0], 1e-5)) and (change[1] <= max(10 * noise[1], 1e-4))
        verdicts.append(ok)
        rows.append((head, rate_base, rate_new, noise, change, ok))

    print("\n{:<10} {:>12} {:>12} {:>9} {:>22} {:>22} {:>7}".format(
        "head", "old win/s", "new win/s", "speed-up",
        "noise (metric/weight)", "new-old (metric/weight)", "result"))
    for head, rate_base, rate_new, noise, change, ok in rows:
        print("{:<10} {:>12.2f} {:>12.2f} {:>8.2f}x {:>10.1e} / {:<9.1e} {:>10.1e} / {:<9.1e} {:>7}".format(
            head, rate_base, rate_new, rate_new / rate_base,
            noise[0], noise[1], change[0], change[1], "PASS" if ok else "FAIL"))
    if all(verdicts):
        print("\nPASS: the new code gives the same results (within run-to-run GPU noise).")
        shutil.rmtree(work, ignore_errors=True)
        return 0
    print("\nFAIL: do not use the new code for the multi-seed runs; outputs kept in", work)
    print("Revert with: git checkout -- plm_special/utils/utils.py utils/bbr.py "
          "utils/training_metrics.py plm_special/models/rl_policy.py train_modern_lora.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
