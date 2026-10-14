"""Decide which unfinished regression seeds are worth another cluster job.

short, medium, and long cap a job at 30 GB, and the scheduler bills that
30 GB request as 2 CPUs. largemem bills 1 CPU and is the only partition
that can go above 30 GB, so it is reserved for jobs that already died
of out-of-memory. A timed-out job is retried on medium only when it had
already labeled at least a quarter of the pool.
"""

MIN_PROGRESS_FRACTION = 0.25
ACCOUNT = "clusters_sponsor"
LARGEMEM_MEMORY = "60G"
MEDIUM_MEMORY = "30G"
RETRY_TIME = "48:00:00"


def parse_queue_task(
    job_name: str,
    array_id: str,
    datasets: set[str],
    methods: set[str],
) -> tuple[str, str, int] | None:
    """Read one expanded squeue row. array_id is the SLURM %K field, not the account."""
    if not str(array_id).isdigit() or "_M" not in job_name:
        return None
    dataset, method_number = job_name.removesuffix("_retry").rsplit("_M", 1)
    method = f"M{method_number}"
    if dataset not in datasets or method not in methods:
        return None
    return dataset, method, int(array_id)


def classify_seed(state: str, n_points: int, expected: int, running: bool) -> str | None:
    """Return the partition for a retry, or None when the seed should stay put."""
    if expected <= 0:
        raise ValueError(f"expected curve length must be positive, got {expected}")
    if n_points >= expected or running:
        return None
    parts = (state or "").split()
    if not parts:
        return None
    token = parts[0]
    if token == "OUT_OF_MEMORY":
        return "largemem"
    if token == "TIMEOUT" and n_points * 4 >= expected:
        return "medium"
    return None


def rewrite_sbatch(
    text: str,
    *,
    partition: str,
    mem: str,
    time_limit: str,
    seeds: list[int],
    job_name: str,
    account: str = ACCOUNT,
) -> str:
    """Copy a generated sbatch and replace the resources that determine the charge."""
    if not seeds:
        raise ValueError("refusing to write an sbatch with no seeds")
    array = ",".join(str(seed) for seed in seeds)
    replacements = {
        "job-name": job_name,
        "partition": partition,
        "array": array,
        "time": time_limit,
    }
    lines: list[str] = []
    saw_memory = False
    saw_account = False
    for line in text.splitlines():
        if line.startswith("#SBATCH --mem"):
            if not saw_memory:
                lines.append(f"#SBATCH --mem={mem}")
                lines.append("#SBATCH --cpus-per-task=1")
                lines.append("#SBATCH --ntasks=1")
                saw_memory = True
            continue
        if line.startswith("#SBATCH --cpus-per-task") or line.startswith("#SBATCH --ntasks"):
            continue
        replaced = False
        for key, value in replacements.items():
            prefix = f"#SBATCH --{key}="
            if line.startswith(prefix):
                lines.append(f"{prefix}{value}")
                replaced = True
                break
        if replaced:
            continue
        if line.startswith("#SBATCH --account="):
            lines.append(f"#SBATCH --account={account}")
            saw_account = True
            continue
        lines.append(line)
    if not saw_memory:
        raise ValueError("sbatch file has no memory directive")
    if not saw_account:
        lines.insert(1, f"#SBATCH --account={account}")
    return "\n".join(lines) + "\n"
