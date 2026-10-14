from src.utils.retry_plan import classify_seed, rewrite_sbatch


ORIGINAL = """#!/bin/bash
#SBATCH --job-name=concrete_M5
#SBATCH --partition=short
#SBATCH --array=0-24
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30G
#SBATCH --mail-type=FAIL

echo run
"""


def test_out_of_memory_goes_to_largemem_even_with_no_curve():
    assert classify_seed("OUT_OF_MEMORY", 0, 63, running=False) == "largemem"


def test_timeout_with_a_quarter_of_the_curve_goes_to_medium():
    assert classify_seed("TIMEOUT", 16, 63, running=False) == "medium"
    assert classify_seed("TIMEOUT", 15, 63, running=False) is None


def test_full_curve_and_running_job_are_left_alone():
    assert classify_seed("TIMEOUT", 63, 63, running=False) is None
    assert classify_seed("OUT_OF_MEMORY", 10, 63, running=True) is None


def test_hung_timeout_and_other_failures_are_not_retried():
    assert classify_seed("TIMEOUT", 2, 381, running=False) is None
    assert classify_seed("FAILED", 40, 63, running=False) is None
    assert classify_seed("", 40, 63, running=False) is None


def test_rewrite_sets_the_resources_that_are_charged():
    rewritten = rewrite_sbatch(
        ORIGINAL,
        partition="largemem",
        mem="60G",
        time_limit="48:00:00",
        seeds=[1, 4],
        job_name="concrete_M5_retry",
    )
    assert "#SBATCH --account=clusters_sponsor" in rewritten
    assert "#SBATCH --partition=largemem" in rewritten
    assert "#SBATCH --array=1,4" in rewritten
    assert "#SBATCH --time=48:00:00" in rewritten
    assert "#SBATCH --mem=60G" in rewritten
    assert "#SBATCH --cpus-per-task=1" in rewritten
    assert "#SBATCH --ntasks=1" in rewritten
    assert "mem-per-cpu" not in rewritten
    assert rewritten.endswith("echo run\n")
