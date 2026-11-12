#!/usr/bin/env python3
"""Correctness tests: known answers, NumPy reference with tolerance 2*gamma_n, and bit-for-bit equality of the
OpenMP version (every thread count) with the sequential baseline. Adapted from scripts/test_correctness.py of
bd-assignment2-matmul @ 1aab86d. Writes results/correctness_tests.csv; exit status 1 on any failure."""
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.environ.get("MATMUL_EXE", os.path.join(ROOT, "build", "matmul3"))
OUT = os.environ.get("TEST_CSV", os.path.join(ROOT, "results", "correctness_tests.csv"))
U = 2.0 ** -53
BS = 96
THREADS = (1, 2, 3, 4, 7, 8, 16)
rows = []


def gamma(k):
    return k * U / (1.0 - k * U)


def record(test, method, n, threads, ok, err=None, note=""):
    rows.append(dict(test=test, method=method, n=n, threads=threads, block_size=BS, result="pass" if ok else "FAIL",
                     max_rel_err="" if err is None else "%.3e" % err, note=note))
    if not ok:
        print("FAIL:", test, method, n, threads, err, note, flush=True)


def write_dense(path, M):
    np.ascontiguousarray(M, dtype="<f8").tofile(path)


def run(method, n, a, b, threads=1, bs=BS, reps=1, warmup=0, batch=1, dump=True):
    out = tempfile.mktemp(prefix="c3_", suffix=".dump") if dump else "-"
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="false")
    cmd = [EXE, method, str(n), str(reps), str(warmup), str(batch), a, b, out, str(bs), str(threads)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0 or "DONE" not in p.stdout:
        raise RuntimeError("run failed: %s\n%s" % (" ".join(cmd), p.stderr))
    res = {"checksums": [], "info": {}}
    for ln in p.stdout.splitlines():
        t = ln.split()
        if t and t[0] in ("REP", "WARMUP"):
            res["checksums"].append(float(t[3]))
        elif len(t) == 2 and t[0].isupper():
            res["info"][t[0]] = t[1]
    if dump:
        res["C"] = np.fromfile(out, dtype="<f8").reshape(n, n)
        os.unlink(out)
    return res


def rel_err(C, R):
    """max relative error where R != 0; exact zeros must match exactly (returns inf otherwise)."""
    zero = (R == 0)
    if np.any(C[zero] != 0):
        return float("inf")
    nz = ~zero
    return float(np.max(np.abs(C[nz] - R[nz]) / np.abs(R[nz]))) if nz.any() else 0.0


def case(tmp, name, A, B, exact_ref=None, rtol=None, threads=THREADS, with_ijk=True):
    """Runs seq, ijk and omp with every thread count; checks against the reference and bitwise against seq."""
    n = A.shape[0]
    pa, pb = os.path.join(tmp, "A.bin"), os.path.join(tmp, "B.bin")
    write_dense(pa, A); write_dense(pb, B)
    ref = exact_ref if exact_ref is not None else A @ B
    configs = [("seq", 1)] + ([("ijk", 1)] if with_ijk else []) + [("omp", p) for p in threads]
    base = None
    for m, p in configs:
        C = run(m, n, pa, pb, p)["C"]
        if rtol is None:
            ok, err = bool(np.array_equal(C, ref)), None
        else:
            err = rel_err(C, ref)
            ok = err <= rtol
        record(name, m, n, p, ok, err)
        if base is None:
            base = C
        else:
            record(name + " [bitwise vs seq]", m, n, p, bool(np.array_equal(C, base)))


def known_answer_tests(tmp):
    rng = np.random.default_rng(12345)
    case(tmp, "scalar 1x1 (3*4=12)", np.array([[3.0]]), np.array([[4.0]]), exact_ref=np.array([[12.0]]))
    A2 = np.array([[1.0, 2.0], [3.0, 4.0]]); B2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    case(tmp, "2x2 non-commutative", A2, B2, exact_ref=np.array([[2.0, 1.0], [4.0, 3.0]]))
    A3 = np.arange(1.0, 10.0).reshape(3, 3); B3 = A3[::-1, ::-1].copy()
    case(tmp, "3x3 hand-computed", A3, B3,
         exact_ref=np.array([[30.0, 24.0, 18.0], [84.0, 69.0, 54.0], [138.0, 114.0, 90.0]]))
    Z = np.zeros((9, 9)); R9 = rng.random((9, 9))
    case(tmp, "zero x random", Z, R9, exact_ref=Z)
    case(tmp, "random x zero", R9, Z, exact_ref=Z)
    I = np.eye(9)
    case(tmp, "identity x random (left)", I, R9, exact_ref=R9)
    case(tmp, "random x identity (right)", R9, I, exact_ref=R9)
    O = np.ones((7, 7))
    case(tmp, "all-ones 7x7 (every entry 7)", O, O, exact_ref=np.full((7, 7), 7.0))
    Ai = rng.integers(-5, 6, (13, 13)).astype(np.float64); Bi = rng.integers(-5, 6, (13, 13)).astype(np.float64)
    case(tmp, "small-integer matrices (exact)", Ai, Bi,
         exact_ref=(Ai.astype(np.int64) @ Bi.astype(np.int64)).astype(np.float64))
    # sizes below, equal to and above the tile size, not multiples of 96 nor of the thread counts
    for n in (5, 17, 64, 95, 96, 97, 100, 191, 250):
        A, B = rng.random((n, n)), rng.random((n, n))
        case(tmp, "random uniform vs NumPy", A, B, rtol=2 * gamma(n))


def tile_size_tests(tmp):
    """The parallel row split must not depend on the tile size: omp with several bs equals seq with the same bs."""
    rng = np.random.default_rng(7)
    n = 150
    A, B = rng.random((n, n)), rng.random((n, n))
    pa, pb = os.path.join(tmp, "A.bin"), os.path.join(tmp, "B.bin")
    write_dense(pa, A); write_dense(pb, B)
    base = run("ijk", n, pa, pb)["C"]
    for bs in (1, 7, 32, 64, 96, 128, 4096):
        for m, p in (("seq", 1), ("omp", 3), ("omp", 8)):
            C = run(m, n, pa, pb, p, bs=bs)["C"]
            rows.append(dict(test="tile size %d [bitwise vs ijk]" % bs, method=m, n=n, threads=p, block_size=bs,
                             result="pass" if np.array_equal(C, base) else "FAIL", max_rel_err="", note=""))


def thread_report_tests(tmp):
    """The program must run exactly the requested number of threads, each one on a different OS thread."""
    rng = np.random.default_rng(3)
    n = 64
    A = rng.random((n, n))
    pa = os.path.join(tmp, "A.bin"); write_dense(pa, A)
    for p in THREADS:
        info = run("omp", n, pa, pa, p, dump=False)["info"]
        ok = (int(info["TEAM_SIZE"]) == p and int(info["DISTINCT_OS_THREADS"]) == p
              and int(info["WORKING_THREADS"]) == min(p, n))
        record("team size / distinct OS threads / threads with rows", "omp", n, p, ok, None,
               "team=%s distinct=%s working=%s os_threads_after=%s" % (info["TEAM_SIZE"], info["DISTINCT_OS_THREADS"],
                                                                      info["WORKING_THREADS"], info["OS_THREADS_AFTER"]))
    info = run("seq", n, pa, pa, 1, dump=False)["info"]
    record("seq creates no extra OS threads", "seq", n, 1, info["OS_THREADS_AFTER"] == "1", None,
           "os_threads_after=%s" % info["OS_THREADS_AFTER"])
    # more threads than rows: the extra threads get no rows, the result must still be right
    A3 = rng.random((3, 3))
    case(tmp, "n=3 with more threads than rows", A3, A3, rtol=2 * gamma(3), threads=(4, 8, 16), with_ijk=False)


def repeatability_tests(tmp):
    """Each call must define every element of C itself: warm-up + repetitions into the same buffer give
    identical checksums."""
    rng = np.random.default_rng(99)
    n = 40
    A, B = rng.random((n, n)), rng.random((n, n))
    pa, pb = os.path.join(tmp, "A.bin"), os.path.join(tmp, "B.bin")
    write_dense(pa, A); write_dense(pb, B)
    for m, p in [("seq", 1), ("ijk", 1)] + [("omp", p) for p in THREADS]:
        r = run(m, n, pa, pb, p, dump=False, reps=4, warmup=2, batch=3)
        record("repeated runs give identical checksum (reps=4, warmup=2, batch=3)", m, n, p,
               len(r["checksums"]) == 6 and len(set(r["checksums"])) == 1)


def main():
    if not os.path.exists(EXE):
        sys.exit("build first: make build")
    with tempfile.TemporaryDirectory() as tmp:
        known_answer_tests(tmp)
        tile_size_tests(tmp)
        thread_report_tests(tmp)
        repeatability_tests(tmp)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    bad = [r for r in rows if r["result"] != "pass"]
    print("%d checks, %d failed -> %s" % (len(rows), len(bad), os.path.relpath(OUT, ROOT)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
