import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import traceback
from itertools import product

# ============================================================
# CONFIG
# ============================================================

SCRIPT = "SmallNet.py"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

EXP_CONF = {
    "FINE_TUNE_EPOCHS": [10, 20],
    "FIT_ONE_CYCLE_EPOCHS": [5, 10],
    "MODEL": ["mobilenet_v3_large", "mobilenet_v3_small"],
    "NUM_SAMPLES": [1000, 2000, 5000, 10000],
    "FILE_SIZE": [25, 35, 50, 75, 100],
    "LOSS": ["ELSE", "CrossEntropyLossFlat"],
    "retries": [0],
}
def cartesian_dict(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    exps = []
    for combo in product(*values):
        d = dict(zip(keys, combo))
        d["name"] = "-".join(f"{k}={v}" for k, v in d.items())
        d["PROJECT_CODE"] = d["name"]
        exps.append(d)

    return exps

EXPERIMENTS = cartesian_dict(EXP_CONF)


def stream_process(proc, log_file):
    with open(log_file, "a") as f:
        for line in proc.stdout:
            print(line, end="")       # stdout to console
            f.write(line)             # stdout to log

        for line in proc.stderr:
            print(line, end="")       # stderr to console
            f.write(line)             # stderr to log


# ============================================================
# RUNNER
# ============================================================

def run_experiment(cfg):
    name = cfg.get("name", "unnamed")
    retries = cfg.get("retries", 0)

    env = os.environ.copy()
    env.update({k: str(v) for k, v in cfg.items()})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{timestamp}_{name}.log"

    start_time = datetime.now()
    attempt = 0

    while attempt <= retries:
        attempt += 1
        print(f"\n🚀 Running [{name}] (attempt {attempt}/{retries+1})")

        attempt_start = datetime.now()

        with open(log_file, "a") as f:
            f.write(
                f"\n=== {name} | attempt {attempt} | start: {attempt_start} ===\n"
            )

        try:
            proc = subprocess.Popen(
                ["python", SCRIPT],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # tee output
            with open(log_file, "a") as f:
                for line in proc.stdout:
                    print(line, end="")
                    f.write(line)

                for line in proc.stderr:
                    print(line, end="")
                    f.write(line)

            returncode = proc.wait()
            duration = datetime.now() - attempt_start

            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, SCRIPT)

            with open(log_file, "a") as f:
                f.write(
                    f"\n=== SUCCESS | attempt_time: {duration} ===\n"
                )

            return True, log_file, datetime.now() - start_time

        except subprocess.CalledProcessError:
            duration = datetime.now() - attempt_start
            print(f"❌ Failed [{name}] after {duration}")

            with open(log_file, "a") as f:
                f.write(
                    f"\n=== FAILED | attempt_time: {duration} ===\n"
                )

            if attempt > retries:
                return False, log_file, datetime.now() - start_time

# ============================================================
# MAIN
# ============================================================

def main():
    results = []
    global_start = datetime.now()

    for cfg in EXPERIMENTS:
        success, log_path, duration = run_experiment(cfg)
        results.append({
            "name": cfg.get("name", "unnamed"),
            "success": success,
            "duration": duration,
            "log": log_path,
        })

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print("\n📊 EXPERIMENT SUMMARY")
    print("-" * 70)
    for r in results:
        status = "✅ SUCCESS" if r["success"] else "❌ FAILED"
        print(
            f"{status:<10} | "
            f"{r['name']:<10} | "
            f"time: {str(r['duration']).split('.')[0]} | "
            f"log: {r['log']}"
        )

    print("-" * 70)
    print(f"Total wall time: {datetime.now() - global_start}")

if __name__ == "__main__":
    main()