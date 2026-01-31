import os
import subprocess
from datetime import datetime
from pathlib import Path
from itertools import product


def _add_project_code(exp_dict, name):
    exp_dict["name"] = f"{name}-" + "-".join(f"{k}={v}" for k, v in exp_dict.items())
    exp_dict["PROJECT_CODE"] = exp_dict["name"]
    return exp_dict


def cartesian_dict(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    exps = []
    for combo in product(*values):
        d = dict(zip(keys, combo))
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


def _run_experiment(cfg, script_run_commands, log_dir):
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


class NestedProjectDirsGetter:
    def __init__(self, root_dir: Path):
        self.r = root_dir

    def __call__(self, project_code):
        work_dir = self.r / project_code / "work"
        perm_dir = self.r / project_code / "perm"
        logs_dir = self.r / project_code / "logs"
        work_dir.mkdir(parents=True)
        perm_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        return work_dir, perm_dir, logs_dir


class SeparateProjectDirsGetter:
    def __init__(self, perm_root_dir: Path, work_root_dir: Path):
        self._perm = perm_root_dir
        self._work = work_root_dir

    def __call__(self, project_code):
        work_dir = self._work / project_code / "work"
        perm_dir = self._perm / project_code / "perm"
        logs_dir = self._perm / project_code / "logs"
        work_dir.mkdir(parents=True)
        perm_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        return work_dir, perm_dir, logs_dir


def _print_summary(results, global_start):
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


def prepare_experiment_cfg(name, cfg, project_dirs_getter, extra_env=None):
    if extra_env is None:
        extra_env = {}
    _add_project_code(cfg, name)
    work_dir, perm_dir, logs_dir = project_dirs_getter(cfg["PROJECT_CODE"])
    cfg = {
        **cfg,
        **extra_env,
    }
    cfg["PROJECT_WORK_DIR"] = str(work_dir)
    cfg["PROJECT_PERM_DIR"] = str(perm_dir)
    return cfg, logs_dir


def run_experiments(
    name,
    experiments,
    script_run_commands,
    project_dirs_getter,
    extra_env=None,
):
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
    run_experiments("exp_name", EXPERIMENTS, Path("./log"), ["uv", "run", "my-script.py"])
    ```

    Pass `extra_env` if it is some base env you need to pass everytime

    Your script would be passed the experiment cfg as environment.
    """
    results = []
    global_start = datetime.now()
    for cfg in experiments:
        cfg, logs_dir = prepare_experiment_cfg(name, cfg, project_dirs_getter, extra_env)
        cfg["PYTHONUNBUFFERED"] = "1"
        success, log_path, duration = _run_experiment(
            cfg, script_run_commands, logs_dir
        )
        results.append(
            {
                "name": cfg.get("name", "unnamed"),
                "success": success,
                "duration": duration,
                "log": log_path,
            }
        )
    _print_summary(results, global_start)
