"""A stand-in for sbatch, for tests, built from the sbatch manual page.

It reproduces the parts campaign relies on, as documented at
https://slurm.schedmd.com/sbatch.html and job_array.html:

* ``--parsable`` prints the job id, or ``id;cluster`` on a multi-cluster
  installation;
* ``--array=first-last[%throttle]`` is the index range;
* with ``--wait``, the exit code is the highest of any array task.

With ``--wait`` it also runs the array, one task after another, through bash,
with ``SLURM_ARRAY_TASK_ID`` and ``SLURM_ARRAY_JOB_ID`` set and each task's
output written where ``--output`` says. Without it, it only records the call,
as a real sbatch returns before anything has run.

Controlled through the environment:

    FAKE_SBATCH_LOG    a directory; each call writes one JSON file there with
                       its arguments and script. One file per call rather than
                       one shared log, because arrays are submitted from
                       several threads at once and appends from concurrent
                       processes are not atomic on Windows: an entry was lost
    FAKE_SBATCH_IDS    a directory used to hand out unique job ids
    FAKE_SBATCH_MODE   "ok" (default), "cluster" to print "id;cluster", or
                       "reject" to fail the way an invalid partition does
    FAKE_SBATCH_BASH   the bash that runs array tasks
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def _next_job_id(directory: Path) -> int:
    # Arrays are submitted from several threads at once, so ids are claimed by
    # creating a file exclusively: whoever creates it first owns the id.
    directory.mkdir(parents=True, exist_ok=True)
    job_id = 1000
    while True:
        try:
            os.close(os.open(directory / str(job_id), os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            job_id += 1
        else:
            return job_id


def main(argv: list[str]) -> int:
    script = Path(argv[-1])
    options = dict(
        a[2:].split("=", 1) for a in argv[:-1] if a.startswith("--") and "=" in a
    )
    flags = {a for a in argv[:-1] if a.startswith("--") and "=" not in a}

    log = os.environ.get("FAKE_SBATCH_LOG")
    if log:
        entry = {"argv": argv, "script": script.read_text(encoding="utf-8")}
        Path(log).mkdir(parents=True, exist_ok=True)
        name = f"{time.time_ns():020d}-{os.getpid()}.json"
        (Path(log) / name).write_text(json.dumps(entry), encoding="utf-8")

    mode = os.environ.get("FAKE_SBATCH_MODE", "ok")
    if mode == "reject":
        print("sbatch: error: invalid partition specified: nowhere", file=sys.stderr)
        print(
            "sbatch: error: Batch job submission failed: Invalid partition name specified",
            file=sys.stderr,
        )
        return 1

    job_id = _next_job_id(Path(os.environ["FAKE_SBATCH_IDS"]))
    print(f"{job_id};testcluster" if mode == "cluster" else job_id, flush=True)
    if "--wait" not in flags:
        return 0

    match = re.fullmatch(r"(\d+)-(\d+)(?:%\d+)?", options["array"])
    assert match is not None, options["array"]
    bash = os.environ.get("FAKE_SBATCH_BASH", "bash")
    worst = 0
    for index in range(int(match[1]), int(match[2]) + 1):
        env = {
            **os.environ,
            "SLURM_ARRAY_TASK_ID": str(index),
            "SLURM_ARRAY_JOB_ID": str(job_id),
        }
        log_path = Path(
            options["output"].replace("%A", str(job_id)).replace("%a", str(index))
        )
        with log_path.open("w", encoding="utf-8") as out:
            proc = subprocess.run(
                [bash, script.as_posix()],
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                check=False,
            )
        worst = max(worst, proc.returncode)
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
