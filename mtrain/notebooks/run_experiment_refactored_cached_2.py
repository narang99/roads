import os
import subprocess
import json
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple, Any

# ============================================================
# CONFIG
# ============================================================

SCRIPT = "SmallNet.py"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

EXPERIMENTS_BASE_DIR = Path("experiments")
EXPERIMENTS_BASE_DIR.mkdir(exist_ok=True)

EXP_CONF = {
    "FINE_TUNE_EPOCHS": [10, 20],
    "FIT_ONE_CYCLE_EPOCHS": [5, 10],
    "MODEL": ["mobilenet_v3_large", "mobilenet_v3_small"],
    "NUM_SAMPLES": [1000, 2000, 5000, 10000],
    "FILE_SIZE": [25, 35, 50, 75, 100],
    "LOSS": ["ELSE", "CrossEntropyLossFlat"],
    "retries": [0],
}


# ============================================================
# CONFIG GENERATION
# ============================================================

def generate_experiment_name(cfg: Dict[str, Any]) -> str:
    """Generate experiment name and project code from config parameters.
    
    Creates a string by joining all k=v pairs from the config with '-'.
    Excludes metadata keys like 'name', 'PROJECT_CODE', and 'retries'.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Generated name string (e.g., "FINE_TUNE_EPOCHS=10-MODEL=mobilenet_v3_large-...")
    """
    exclude_keys = {"name", "PROJECT_CODE", "retries"}
    items = [(k, v) for k, v in cfg.items() if k not in exclude_keys]
    return "-".join(f"{k}={v}" for k, v in items)


def cartesian_dict(param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters from a parameter grid.
    
    Args:
        param_grid: Dictionary mapping parameter names to lists of values
        
    Returns:
        List of configuration dictionaries with all parameter combinations
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    exps = []
    for combo in product(*values):
        d = dict(zip(keys, combo))
        name = generate_experiment_name(d)
        d["name"] = name
        d["PROJECT_CODE"] = name
        exps.append(d)

    return exps


EXPERIMENTS = cartesian_dict(EXP_CONF)


# ============================================================
# PROCESS MANAGEMENT
# ============================================================

def stream_process_output(proc: subprocess.Popen, log_file: Path) -> None:
    """Stream stdout and stderr from a process to console and log file.
    
    Args:
        proc: The subprocess process object
        log_file: Path to write output logs
    """
    with open(log_file, "a") as f:
        for line in proc.stdout:
            print(line, end="")
            f.write(line)

        for line in proc.stderr:
            print(line, end="")
            f.write(line)


def build_process_env(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Build environment variables from configuration dictionary.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Environment dictionary with config values as strings
    """
    env = os.environ.copy()
    env.update({k: str(v) for k, v in cfg.items()})
    return env


def create_experiment_output_dir(project_code: str) -> Path:
    """Create and return the output directory for an experiment.
    
    Args:
        project_code: The PROJECT_CODE identifier for the experiment
        
    Returns:
        Path object for the experiment output directory
    """
    exp_output_dir = EXPERIMENTS_BASE_DIR / project_code
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    return exp_output_dir


def dump_experiment_config(cfg: Dict[str, Any], output_dir: Path, user_param_keys: List[str]) -> None:
    """Save experiment configuration to params.json in the output directory.
    
    Only saves parameters provided by the user, excluding auto-generated metadata.
    
    Args:
        cfg: Full configuration dictionary
        output_dir: Path to the experiment output directory
        user_param_keys: List of parameter keys provided by the user
    """
    params = {k: cfg[k] for k in user_param_keys if k in cfg}
    
    params_file = output_dir / "params.json"
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)


def load_experiment_params(output_dir: Path) -> Dict[str, Any]:
    """Load experiment parameters from params.json.
    
    Args:
        output_dir: Path to the experiment output directory
        
    Returns:
        Dictionary of parameters, or empty dict if params.json doesn't exist
    """
    params_file = output_dir / "params.json"
    if not params_file.exists():
        return {}
    
    with open(params_file, "r") as f:
        return json.load(f)


def params_match(current_cfg: Dict[str, Any], saved_params: Dict[str, Any], user_param_keys: List[str]) -> bool:
    """Check if current config parameters match saved parameters.
    
    Only compares keys provided by the user.
    
    Args:
        current_cfg: Current configuration dictionary
        saved_params: Parameters loaded from params.json
        user_param_keys: List of parameter keys provided by the user
        
    Returns:
        True if all user-provided params match, False otherwise
    """
    for key in user_param_keys:
        current_value = current_cfg.get(key)
        saved_value = saved_params.get(key)
        if current_value != saved_value:
            return False
    return True


def load_all_experiments(user_param_keys: List[str]) -> Dict[str, Path]:
    """Load all existing experiments and their parameters.
    
    Scans EXPERIMENTS_BASE_DIR for all subdirectories with params.json files.
    
    Args:
        user_param_keys: List of parameter keys provided by the user
        
    Returns:
        Dictionary mapping frozenset of params to experiment output directory
    """
    existing_experiments = {}
    
    if not EXPERIMENTS_BASE_DIR.exists():
        return existing_experiments
    
    for exp_dir in EXPERIMENTS_BASE_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
        
        params = load_experiment_params(exp_dir)
        if not params:
            continue
        
        # Create a hashable key from user params only
        params_key = frozenset((k, v) for k in user_param_keys if k in params for v in [params[k]])
        existing_experiments[params_key] = exp_dir
    
    return existing_experiments


def find_matching_experiment(
    cfg: Dict[str, Any],
    existing_experiments: Dict[str, Path],
    user_param_keys: List[str],
) -> Path:
    """Find an existing experiment that matches the current config.
    
    Args:
        cfg: Current configuration dictionary
        existing_experiments: Dictionary of existing experiments from load_all_experiments()
        user_param_keys: List of parameter keys provided by the user
        
    Returns:
        Path to matching experiment directory, or None if no match found
    """
    # Create a hashable key from current config's user params
    current_params_key = frozenset((k, cfg[k]) for k in user_param_keys if k in cfg)
    
    return existing_experiments.get(current_params_key, None)

def generate_log_file(name: str) -> Path:
    """Generate a timestamped log file path.
    
    Args:
        name: Base name for the log file
        
    Returns:
        Path object for the log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"{timestamp}_{name}.log"


def create_subprocess(env: Dict[str, str]) -> subprocess.Popen:
    """Create a subprocess for running the experiment script.
    
    Args:
        env: Environment variables to pass to subprocess
        
    Returns:
        Popen object for the subprocess
    """
    return subprocess.Popen(
        ["python", SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def log_attempt_start(log_file: Path, name: str, attempt: int) -> None:
    """Log the start of an experiment attempt.
    
    Args:
        log_file: Path to log file
        name: Name of the experiment
        attempt: Current attempt number
    """
    with open(log_file, "a") as f:
        f.write(
            f"\n=== {name} | attempt {attempt} | start: {datetime.now()} ===\n"
        )


def log_attempt_result(log_file: Path, success: bool, duration: float) -> None:
    """Log the result of an experiment attempt.
    
    Args:
        log_file: Path to log file
        success: Whether the attempt succeeded
        duration: Time taken for the attempt
    """
    result = "SUCCESS" if success else "FAILED"
    with open(log_file, "a") as f:
        f.write(f"\n=== {result} | attempt_time: {duration} ===\n")
# ============================================================
# RUNNER
# ============================================================

def run_experiment_attempt(
    cfg: Dict[str, Any],
    log_file: Path,
    attempt: int,
) -> Tuple[bool, float]:
    """Execute a single attempt of an experiment.
    
    Args:
        cfg: Configuration dictionary for the experiment
        log_file: Path to write logs
        attempt: Current attempt number
        
    Returns:
        Tuple of (success: bool, duration: timedelta)
    """
    name = cfg.get("name", "unnamed")
    attempt_start = datetime.now()
    
    log_attempt_start(log_file, name, attempt)
    
    try:
        env = build_process_env(cfg)
        proc = create_subprocess(env)
        stream_process_output(proc, log_file)
        
        returncode = proc.wait()
        duration = datetime.now() - attempt_start
        
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, SCRIPT)
        
        log_attempt_result(log_file, success=True, duration=duration)
        return True, duration
        
    except subprocess.CalledProcessError:
        duration = datetime.now() - attempt_start
        name_short = name[:50] if name else "unnamed"
        print(f"❌ Failed [{name_short}] after {duration}")
        log_attempt_result(log_file, success=False, duration=duration)
        return False, duration


def run_experiment(cfg: Dict[str, Any], output_dir: Path, user_param_keys: List[str], skip_if_cached: bool = True) -> Tuple[bool, Path, float, bool]:
    """Run an experiment with retry logic and caching support.
    
    Args:
        cfg: Configuration dictionary for the experiment
        output_dir: Path to the output directory for this experiment
        user_param_keys: List of parameter keys provided by the user
        skip_if_cached: Whether to skip execution if cached params match
        
    Returns:
        Tuple of (success: bool, log_file: Path, total_duration: timedelta, was_cached: bool)
    """
    name = cfg.get("name", "unnamed")
    retries = cfg.get("retries", 0)
    
    # Add output directory to config
    cfg["OUTPUT_DIR"] = str(output_dir)
    
    # Dump configuration to params.json
    dump_experiment_config(cfg, output_dir, user_param_keys)
    
    log_file = generate_log_file(name)
    start_time = datetime.now()
    
    print(f"\n🚀 Running [{name}]")
    print(f"   Output: {output_dir}")
    
    for attempt in range(1, retries + 2):
        success, _ = run_experiment_attempt(cfg, log_file, attempt)
        
        if success:
            total_duration = datetime.now() - start_time
            print(f"✅ Completed [{name}] in {total_duration}")
            return True, log_file, total_duration, False
    
    total_duration = datetime.now() - start_time
    return False, log_file, total_duration, False

# ============================================================
# REPORTING
# ============================================================

def format_result_row(result: Dict[str, Any]) -> str:
    """Format a single experiment result for display.
    
    Args:
        result: Result dictionary from an experiment
        
    Returns:
        Formatted string for display
    """
    status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
    duration_str = str(result["duration"]).split('.')[0]
    cached_indicator = "⚡" if result.get("cached", False) else " "
    return (
        f"{status:<10} | {cached_indicator} | "
        f"{result['name']:<10} | "
        f"time: {duration_str} | "
        f"log: {result['log']}"
    )


def print_summary(results: List[Dict[str, Any]], total_time: float) -> None:
    """Print a summary of all experiment results.
    
    Args:
        results: List of result dictionaries from all experiments
        total_time: Total wall time for all experiments
    """
    print("\n📊 EXPERIMENT SUMMARY")
    print("-" * 70)
    
    for result in results:
        print(format_result_row(result))
    
    print("-" * 70)
    print(f"Total wall time: {total_time}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Main entry point: run all experiments and report results."""
    results = []
    global_start = datetime.now()

    print(f"🎯 Starting {len(EXPERIMENTS)} experiments...")
    print(f"📁 Experiments base directory: {EXPERIMENTS_BASE_DIR.absolute()}")

    # Get user-provided parameter keys from EXP_CONF
    user_param_keys = list(EXP_CONF.keys())
    
    # Load all existing experiments at the start
    existing_experiments = load_all_experiments(user_param_keys)
    print(f"📦 Found {len(existing_experiments)} existing experiments")

    for cfg in EXPERIMENTS:
        # Check if this config matches any existing experiment
        matching_exp_dir = find_matching_experiment(cfg, existing_experiments, user_param_keys)
        
        if matching_exp_dir:
            # Use existing experiment directory
            output_dir = matching_exp_dir
            was_cached = True
            name = cfg.get("name", "unnamed")
            print(f"\n⚡ Cached [{name}]")
            print(f"   Output: {output_dir}")
            log_file = generate_log_file(name)
            results.append({
                "name": name,
                "success": True,
                "duration": 0.0,
                "log": log_file,
                "output_dir": output_dir,
                "cached": was_cached,
            })
        else:
            # Run new experiment
            project_code = cfg.get("PROJECT_CODE", "unnamed")
            output_dir = create_experiment_output_dir(project_code)
            
            success, log_path, duration, was_cached = run_experiment(cfg, output_dir, user_param_keys)
            results.append({
                "name": cfg.get("name", "unnamed"),
                "success": success,
                "duration": duration,
                "log": log_path,
                "output_dir": output_dir,
                "cached": was_cached,
            })

    total_time = datetime.now() - global_start
    print_summary(results, total_time)

if __name__ == "__main__":
    main()