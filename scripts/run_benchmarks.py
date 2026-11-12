#!/usr/bin/env python3
"""Benchmark driver (macOS port of scripts/run_benchmarks.py of bd-assignment2-matmul @ 1aab86d).
One process per configuration (method, n, threads), warm-up + measured samples, 120 s / 4 GiB budgets, a pause
between processes, a seeded random order of the configurations of each size, and validation of every product
against NumPy (tolerance 2*gamma_n) plus a SHA-256 of C to check bit-for-bit equality with the sequential baseline.
Campaigns: pilot, quick, main, large, pass2, ijk."""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # NumPy is only used for validation; keep it single-threaded
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import selectors  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXE = ROOT / "build" / "matmul3"
U = 2.0 ** -53
T_SAMPLE = 120.0                 # budget: seconds per timed sample
MEM_BYTES = 4 * 1024 ** 3        # budget: resident memory per process (4 GiB)
BS = 96
SEED = 20261112
SEQ_S_PER_N3 = 3.3e-10           # pilot estimate of the sequential time per n^3 (only used to choose the batch size)
BATCH_TARGET_S = 0.2             # batch tiny products so that one sequential sample lasts at least ~0.2 s

THREADS_ALL = [1, 2, 3, 4, 6, 8, 16]      # 16 > 8 logical processors: oversubscription
CAMPAIGNS = {
    # name: (sizes, configs, warm-up, measured repetitions, pause between processes in s, pass)
    "pilot": ([1000, 2000], [("seq", 1), ("omp", 4), ("omp", 8)], 1, 2, 2.0, 1),
    "quick": ([100, 250, 500, 1000], [("seq", 1)] + [("omp", p) for p in (1, 2, 4, 8, 16)], 1, 3, 0.5, 1),
    "main": ([100, 250, 500, 1000, 1500, 2000, 3000], [("seq", 1)] + [("omp", p) for p in THREADS_ALL], 1, 5, 3.0, 1),
    "large": ([4000], [("seq", 1), ("omp", 1), ("omp", 4), ("omp", 8)], 1, 3, 3.0, 1),
    "pass2": ([2000], [("seq", 1)] + [("omp", p) for p in THREADS_ALL], 1, 5, 3.0, 2),
    "ijk": ([500, 1000], [("ijk", 1)], 1, 5, 3.0, 1),
}

RAW_FIELDS = ["campaign", "run_id", "pass", "order", "method", "numeric_type", "n", "block_size", "threads_requested",
              "phase", "repetition", "batch_size", "time_per_mult_s", "checksum"]
RUN_FIELDS = ["campaign", "run_id", "pass", "order", "method", "numeric_type", "n", "block_size", "threads_requested",
              "team_size", "working_threads", "distinct_os_threads", "os_threads_after", "oversubscribed", "status",
              "reason", "warmup_planned", "reps_planned", "warmup_completed", "reps_completed", "batch_size",
              "limit_sample_s", "limit_mem_bytes", "predicted_bytes", "rss_after_load_bytes", "maxrss_bytes",
              "rss_polled_max_bytes", "wall_s", "max_rel_err", "rtol_used", "correctness", "sha256_C",
              "bitwise_equal_seq", "cpu_speed_limit_before", "cpu_speed_limit_after", "omp_env", "start_utc",
              "returncode", "stderr_tail"]


def gamma(k):
    return k * U / (1.0 - k * U)


def batch_for(n):
    t = SEQ_S_PER_N3 * n ** 3
    return 1 if t >= BATCH_TARGET_S else int(math.ceil(BATCH_TARGET_S / t))


def cpu_speed_limit():
    """CPU_Speed_Limit reported by `pmset -g therm` (100 = no thermal throttling requested by the OS)."""
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", subprocess.run(["pmset", "-g", "therm"], capture_output=True,
                                                                  text=True).stdout)
    return int(m.group(1)) if m else ""


def rss_bytes(pid):
    r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return int(r.stdout.strip()) * 1024 if r.stdout.strip().isdigit() else 0


def execute(cmd, env):
    """Runs one process with the per-sample time budget and the memory budget (RSS polled every 0.5 s)."""
    errf = tempfile.TemporaryFile()
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, env=env, start_new_session=True)
    fd = proc.stdout.fileno()
    os.set_blocking(fd, False)
    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    buf, lines, eof = b"", [], False
    status, reason, last, rss_max = "ok", "", time.monotonic(), 0
    while not eof:
        for _ in sel.select(timeout=0.5):
            chunk = os.read(fd, 65536)
            if not chunk:
                eof = True
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                lines.append(line.decode(errors="replace").strip())
                last = time.monotonic()
        if eof:
            break
        rss = rss_bytes(proc.pid)
        rss_max = max(rss_max, rss)
        if time.monotonic() - last > T_SAMPLE:
            status, reason = "timeout", "no sample completed within %g s" % T_SAMPLE
        elif rss > MEM_BYTES:
            status, reason = "memory_limit", "resident memory exceeded %d bytes" % MEM_BYTES
        if status != "ok":
            os.killpg(proc.pid, signal.SIGKILL)
            break
    lines += [ln.strip() for ln in buf.decode(errors="replace").splitlines() if ln.strip()]
    rc = proc.wait()
    wall = time.monotonic() - t0
    errf.seek(0)
    err = errf.read().decode(errors="replace")
    errf.close()
    if status == "ok" and (rc != 0 or "DONE" not in lines):
        status, reason = ("memory_limit", "allocation failed") if rc == 3 else ("error", "exit code %s" % rc)
    return lines, status, reason, wall, rss_max, rc, err


def parse(lines):
    out = {"warmup": [], "rep": [], "info": {}}
    for ln in lines:
        p = ln.split()
        if p and p[0] in ("WARMUP", "REP") and len(p) >= 4:
            out["warmup" if p[0] == "WARMUP" else "rep"].append((int(p[1]), float(p[2]), float(p[3])))
        elif len(p) == 2 and p[0].isupper():
            out["info"][p[0]] = p[1]
    return out


class Reference:
    """NumPy product of the inputs of the size currently measured, and the SHA-256 of the sequential result."""
    def __init__(self):
        self.n, self.C, self.seq_sha = None, None, {}

    def get(self, n):
        if self.n != n:
            A = np.fromfile(DATA / ("A_n%d.bin" % n), dtype="<f8").reshape(n, n)
            B = np.fromfile(DATA / ("B_n%d.bin" % n), dtype="<f8").reshape(n, n)
            self.n, self.C = n, A @ B
        return self.C


def validate(dump, n, ref):
    C = np.fromfile(dump, dtype="<f8").reshape(n, n)
    R = ref.get(n)
    err = float(np.max(np.abs(C - R) / np.abs(R)))      # inputs are in [0,1): no zero entries in R for n >= 1
    with open(dump, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return err, sha


def load_done(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {r["run_id"] for r in csv.DictReader(f) if r["status"] == "ok"}


def append(path, fields, rows):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def known_seq_sha(runs_path):
    """SHA-256 of the sequential product for each n, from earlier runs (any campaign)."""
    out = {}
    if runs_path.exists():
        with open(runs_path) as f:
            for r in csv.DictReader(f):
                if r["method"] == "seq" and r["status"] == "ok" and r["sha256_C"]:
                    out[int(r["n"])] = r["sha256_C"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True, choices=list(CAMPAIGNS))
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--resume", action="store_true", help="skip configurations already completed with status ok")
    args = ap.parse_args()

    sizes, configs, warmup, reps, pause, pass_no = CAMPAIGNS[args.campaign]
    res = ROOT / args.results_dir
    res.mkdir(parents=True, exist_ok=True)
    raw_path, runs_path = res / "raw_measurements.csv", res / "runs.csv"
    done = load_done(runs_path) if args.resume else set()
    seq_sha = known_seq_sha(runs_path)
    ref = Reference()
    logical = os.cpu_count()
    t_campaign = time.monotonic()

    for n in sizes:
        order = list(configs)
        random.Random(SEED + 1000 * pass_no + n).shuffle(order)
        # the sequential baseline goes first so that its SHA-256 is known when the others are checked
        if ("seq", 1) in order and n not in seq_sha:
            order.remove(("seq", 1)); order.insert(0, ("seq", 1))
        for k, (method, p) in enumerate(order):
            run_id = "%s:%s:n%d:p%d:bs%d:pass%d" % (args.campaign, method, n, p, BS, pass_no)
            if run_id in done:
                print("skip", run_id, flush=True)
                continue
            batch = batch_for(n)
            predicted = 3 * 8 * n * n
            row = dict(campaign=args.campaign, run_id=run_id, **{"pass": pass_no}, order=k, method=method,
                       numeric_type="float64", n=n, block_size=BS, threads_requested=p,
                       oversubscribed=p > logical, warmup_planned=warmup, reps_planned=reps, batch_size=batch,
                       limit_sample_s=T_SAMPLE, limit_mem_bytes=MEM_BYTES, predicted_bytes=predicted,
                       start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
            if predicted > MEM_BYTES:
                row.update(status="not_attempted", reason="predicted array storage above the memory budget")
                append(runs_path, RUN_FIELDS, [row])
                continue
            ref.get(n)                      # NumPy reference computed before the run, outside any measurement
            time.sleep(pause)
            dump = tempfile.mktemp(prefix="c3_", suffix=".bin")
            env = dict(os.environ, LC_ALL="C", OMP_NUM_THREADS=str(p), OMP_DYNAMIC="false", OMP_DISPLAY_ENV="TRUE")
            cmd = [str(EXE), method, str(n), str(reps), str(warmup), str(batch), str(DATA / ("A_n%d.bin" % n)),
                   str(DATA / ("B_n%d.bin" % n)), dump, str(BS), str(p)]
            row["cpu_speed_limit_before"] = cpu_speed_limit()
            lines, status, reason, wall, rss_max, rc, err = execute(cmd, env)
            row["cpu_speed_limit_after"] = cpu_speed_limit()
            out = parse(lines)
            info = out["info"]
            omp_env = re.findall(r"\[host\] (OMP_(?:NUM_THREADS|DYNAMIC|PROC_BIND|WAIT_POLICY))='([^']*)'", err)
            row.update(status=status, reason=reason, wall_s="%.3f" % wall, returncode=rc,
                       warmup_completed=len(out["warmup"]), reps_completed=len(out["rep"]),
                       team_size=info.get("TEAM_SIZE", 1 if method != "omp" else ""),
                       working_threads=info.get("WORKING_THREADS", 1 if method != "omp" else ""),
                       distinct_os_threads=info.get("DISTINCT_OS_THREADS", 1 if method != "omp" else ""),
                       os_threads_after=info.get("OS_THREADS_AFTER", ""),
                       rss_after_load_bytes=info.get("RSS_AFTER_LOAD_BYTES", ""),
                       maxrss_bytes=info.get("MAXRSS_BYTES", ""), rss_polled_max_bytes=rss_max,
                       omp_env=";".join("%s=%s" % kv for kv in omp_env),
                       stderr_tail=" | ".join(ln for ln in err.splitlines() if "[host]" not in ln
                                              and "OPENMP DISPLAY" not in ln)[-300:])
            if status == "ok":
                tol = 2 * gamma(n)
                rel, sha = validate(dump, n, ref)
                if method == "seq":
                    seq_sha[n] = sha
                row.update(max_rel_err="%.3e" % rel, rtol_used="%.3e" % tol, sha256_C=sha,
                           correctness="pass" if rel <= tol else "FAIL",
                           bitwise_equal_seq=(sha == seq_sha[n]) if n in seq_sha else "")
            if os.path.exists(dump):
                os.unlink(dump)
            samples = [dict(campaign=args.campaign, run_id=run_id, **{"pass": pass_no}, order=k, method=method,
                            numeric_type="float64", n=n, block_size=BS, threads_requested=p, phase=ph,
                            repetition=i, batch_size=batch, time_per_mult_s="%.9e" % t, checksum="%.17g" % c)
                       for ph, key in (("warmup", "warmup"), ("measured", "rep")) for i, t, c in out[key]]
            append(raw_path, RAW_FIELDS, samples)
            append(runs_path, RUN_FIELDS, [row])
            med = sorted(t for _, t, _ in out["rep"])[len(out["rep"]) // 2] if out["rep"] else float("nan")
            print("%-6s n=%-5d p=%-2d %-8s median=%.4g s  correct=%s bitwise=%s  speed_limit=%s->%s  (%.0f s elapsed)"
                  % (method, n, p, status, med, row.get("correctness", ""), row.get("bitwise_equal_seq", ""),
                     row["cpu_speed_limit_before"], row["cpu_speed_limit_after"], time.monotonic() - t_campaign),
                  flush=True)
    print("campaign %s finished in %.1f min" % (args.campaign, (time.monotonic() - t_campaign) / 60))


if __name__ == "__main__":
    main()
