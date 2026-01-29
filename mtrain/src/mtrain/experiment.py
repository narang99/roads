import os
import subprocess
from datetime import datetime
from pathlib import Path
from itertools import product


def cartesian_dict(name, param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    exps = []
    for combo in product(*values):
        d = dict(zip(keys, combo))
        d["name"] = f"{name}-" + "-".join(f"{k}={v}" for k, v in d.items())
        d["PROJECT_CODE"] = d["name"]
        exps.append(d)

    return exps


def stream_process(proc, log_file):
    with open(log_file, "a") as f:
        for line in proc.stdout:
            print(line, end="")  # stdout to console
            f.write(line)  # stdout to log

        for line in proc.stderr:
            print(line, end="")  # stderr to console
            f.write(line)  # stderr to log


def _run_experiment(cfg, log_dir, script_run_commands):
    name = cfg.get("name", "unnamed")
    retries = cfg.get("retries", 0)

    env = os.environ.copy()
    env.update({k: str(v) for k, v in cfg.items()})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}_{name}.log"

    start_time = datetime.now()
    attempt = 0

    while attempt <= retries:
        attempt += 1
        print(f"\n🚀 Running [{name}] (attempt {attempt}/{retries + 1})")
        print(f"log file: {log_file}")

        attempt_start = datetime.now()

        with open(log_file, "a") as f:
            f.write(f"\n=== {name} | attempt {attempt} | start: {attempt_start} ===\n")

        try:
            command = script_run_commands
            env_string = " ".join([f"{k}='{str(v)}'" for k, v in cfg.items()])
            print("running command: ", env_string, " ".join(command))
            proc = subprocess.Popen(
                command,
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
                raise subprocess.CalledProcessError(
                    returncode, " ".join(script_run_commands)
                )

            with open(log_file, "a") as f:
                f.write(f"\n=== SUCCESS | attempt_time: {duration} ===\n")

            return True, log_file, datetime.now() - start_time

        except subprocess.CalledProcessError:
            duration = datetime.now() - attempt_start
            print(f"❌ Failed [{name}] after {duration}")

            with open(log_file, "a") as f:
                f.write(f"\n=== FAILED | attempt_time: {duration} ===\n")

            if attempt > retries:
                return False, log_file, datetime.now() - start_time


# ============================================================
# MAIN
# ============================================================


def run_experiments(experiments, log_dir, script_run_commands, extra_env=None):
    """Run your command wiht given experiment configs repeatadly. a very basic runner

    Example usage:
    ```
    EXP_CONF = {
        "FINE_TUNE_EPOCHS": [1],
        "FIT_ONE_CYCLE_EPOCHS": [1],
        "MODEL": ["resnet18"],
        "NUM_SAMPLES": [100],
        "FILE_SIZE": [35],
        "LOSS": ["ELSE"],
        "retries": [0],
    }
    EXPERIMENTS = cartesian_dict(EXP_CONF)
    run_experiments(EXPERIMENTS, Path("./log"), ["uv", "run", "my-script.py"])
    ```

    Pass `extra_env` if it is some base env you need to pass everytime

    Your script would be passed the experiment cfg as environment.
    """
    if extra_env is None:
        extra_env = {}

    results = []
    global_start = datetime.now()

    for cfg in experiments:
        cfg = {
            "PYTHONUNBUFFERED": "1",
            **cfg, 
            **extra_env,
        }
        success, log_path, duration = _run_experiment(cfg, log_dir, script_run_commands)
        results.append(
            {
                "name": cfg.get("name", "unnamed"),
                "success": success,
                "duration": duration,
                "log": log_path,
            }
        )

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
