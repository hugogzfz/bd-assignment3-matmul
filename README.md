# Big Data (ULPGC 2026-2027) – Individual Assignment 3
## Parallel matrix multiplication (OpenMP, shared memory) and optional vectorisation

Author: Hugo González Fernández · Bachelor’s Degree in Data Science and Engineering, ULPGC
Repository: https://github.com/hugogzfz/bd-assignment3-matmul

Research question: how does the number of threads affect the runtime, speedup and efficiency of a blocked
dense `float64` matrix multiplication on a 4-core / 8-thread laptop CPU, and from which matrix size does parallel
execution pay off?

The report (`Assignment03_GonzalezFernandez_Report.pdf`) contains the design, results, discussion and limitations. This README
gives everything needed to rebuild and re-run the study.

## 1. Layout

```
src/c/matmul3.c               seq (A2 blocked kernel, bs = 96), omp (same kernel, rows of C split among threads), ijk (A1 reference)
scripts/gen_inputs.py         generates the input matrices and checks them against data/manifest.json
scripts/test_correctness.py   correctness tests -> results/correctness_tests.csv
scripts/run_benchmarks.py     measurement driver (campaigns main, large, ijk, pass2, bsweep, vec; pilot, quick)
scripts/env_info.py           environment record -> results/environment.json
scripts/analyze.py            summaries (results/summary_*), LaTeX tables (report/tables/) and figures (figures/)
scripts/vec_evidence.py       vectorisation evidence (compiler remarks + disassembly counts)
results/                      raw_measurements.csv (every sample), runs.csv (one row per process), campaign.log, summaries
figures/, report/             figures and LaTeX sources of the report
data/manifest.json            SHA-256 of every input (the matrices themselves are regenerated, not stored)
docs/guia_defensa.md          short oral-defence guide (Spanish)
```

## 2. Reused material

* Blocked kernel, input generator, test structure and driver structure from
  `github.com/hugogzfz/bd-assignment2-matmul` @ `1aab86d7c5b014435b90e9c4baf7031ba9e05aa0`.
* `ijk` reference kernel and input matrices from `github.com/hugogzfz/bd-assignment1-matmul` @
  `101af1cd5b0fe1ffb2809c63798c99398e2e236b` (identical files, checked with the SHA-256 digests).

## 3. Requirements (as used)

macOS 12.7.6 (x86-64, Intel Core i7-6820HQ, 4 cores / 8 threads, 16 GiB), Apple clang 14.0.0, Homebrew `libomp` 23.1.5
(`brew install libomp`; Apple clang ships without OpenMP), Python 3.12.7 with `numpy 2.4.2 pandas 3.0.0 matplotlib 3.10.8` in
a virtual environment (`requirements.txt`), BasicTeX + `latexmk booktabs enumitem caption float` (tlmgr) only for the PDF.
NumPy is used only for input generation and validation, never inside a timed region. About 0.5 GB of disk for the inputs.

## 4. Exact commands

```bash
make venv           # python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make build          # clang -O2 -std=c11 -ffp-contract=off -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include -L.../lib -lomp
make inputs         # regenerates data/A_n*.bin, data/B_n*.bin and verifies them against data/manifest.json
make build-vec      # scalar / SSE2 / AVX2 builds of the optional vectorisation extension (build/vec/)
make test           # correctness suite on the primary and the three vector builds (must report 0 failed)
make bench-quick    # reduced run (< 1 min, n <= 1000) -> results_quick/, to check that everything works
caffeinate -i make bench   # full campaign -> results/ (about 35 min; resumable with --resume)
make vecinfo        # vectorisation evidence
make analyze        # summaries, tables, figures
make report         # report/report.pdf (needs /Library/TeX/texbin in PATH)
```

Single run by hand:
`OMP_NUM_THREADS=4 OMP_DYNAMIC=false build/matmul3 omp 1000 5 1 1 data/A_n1000.bin data/B_n1000.bin - 96 4`
(arguments: `method n reps warmup batch A B dumpC|- bs threads`).

Output lines of `matmul3`: `META`, `RSS_AFTER_LOAD_BYTES`, `OS_THREADS_BEFORE`, `WARMUP|REP idx seconds_per_mult checksum`,
`TEAM_SIZE`, `WORKING_THREADS`, `DISTINCT_OS_THREADS` (omp only), `OS_THREADS_AFTER`, `MAXRSS_BYTES`, `DONE`.

## 5. Fixed settings, timing boundary and protocol

* Flags: `-O2 -std=c11 -ffp-contract=off` (no `-march=native`, no `-ffast-math`) → every method, thread count and vector build
  performs the same additions in the same order and returns identical bits (tested; checked again in every benchmark run).
* Timer: `CLOCK_MONOTONIC` around the multiplication call only (zero-fill of C and OpenMP fork/join inside; file reading, allocation and
  page-touching of C, checksum, dump and validation outside).
* Runtime: `OMP_NUM_THREADS=p`, `OMP_DYNAMIC=false`; `OMP_DISPLAY_ENV` output stored per run. Threads are **not pinned** (macOS has no
  affinity API; `OMP_PROC_BIND` is not honoured). NumPy runs with `OPENBLAS_NUM_THREADS=1`.
* Repetitions: 1 warm-up (kept in the raw CSV, excluded) + 5 measured (3 at n = 4000); batches for n ≤ 500 so a sequential sample lasts ≥ ~0.2 s.
  Median and IQR; S(p) = T_seq/T_p and E(p) = S(p)/p from medians.
* Order: seeded random order within each n (sequential baseline first), 3 s pause between processes, `pmset -g therm`
  `CPU_Speed_Limit` stored before/after each process; the n = 2000 thread sweep is repeated at the end (`pass2`).
* Budgets: 120 s per timed sample, 4 GiB resident memory per process (RSS polled every 0.5 s); violations are stored as
  `timeout` / `memory_limit`, never as a time.
* Memory metric: `getrusage(RUSAGE_SELF).ru_maxrss`, peak resident set of the process, **in bytes on macOS**.

## 6. Inputs

Uniform [0,1) `float64`, `numpy.random.default_rng([20261012, n, w])` (w = 0 for A, 1 for B), little-endian row-major binary files
`data/A_n<n>.bin`, `data/B_n<n>.bin` for n = 100, 250, 500, 1000, 1500, 2000, 3000, 4000. `make inputs` regenerates them and aborts if any
SHA-256 differs from `data/manifest.json` (digests copied from Assignment 2; all identical).

## 7. Columns of the raw CSV files

`results/raw_measurements.csv` (one row per warm-up or measured sample):

| column | meaning |
|---|---|
| campaign | main, large, ijk, pass2, bsweep, vec |
| run_id | `campaign:method:variant:n:p:bs:pass`, key to runs.csv |
| pass | 1, or 2 for the repeated n = 2000 sweep |
| order | position of the process in the randomised order of its n |
| method | seq (baseline), omp (parallel), ijk (A1 reference) |
| variant | build: sse2 (primary, build/matmul3), scalar, sse2-vec, avx2 (vectorisation extension) |
| numeric_type | float64 |
| n | matrix side |
| block_size | tile size b (96 except in bsweep) |
| threads_requested | p (OMP_NUM_THREADS); 1 for seq / ijk |
| phase | warmup or measured |
| repetition | index within the phase |
| batch_size | multiplications timed together |
| time_per_mult_s | seconds per multiplication (sample time / batch_size) |
| checksum | sum of all elements of C after the sample |

`results/runs.csv` (one row per process): the identifiers above plus `team_size`, `working_threads`, `distinct_os_threads` (reported by the
program), `os_threads_after` (OS threads of the process at the end), `oversubscribed` (p > 8), `status` (ok / timeout / memory_limit /
not_attempted / error), `reason`, planned and completed warm-ups and repetitions, `limit_sample_s`, `limit_mem_bytes`, `predicted_bytes`
(3·8n²), `rss_after_load_bytes`, `maxrss_bytes` (peak RSS, used in the report), `rss_polled_max_bytes` (driver poll, cross-check), `wall_s`,
`max_rel_err`, `rtol_used` (2γₙ), `correctness` (pass / FAIL), `sha256_C`, `bitwise_equal_seq`, `cpu_speed_limit_before/after`, `omp_env`,
`start_utc`, `returncode`, `stderr_tail`.

`results/correctness_tests*.csv`: one row per check (test, method, n, threads, block_size, result, max_rel_err, note), for the primary
binary and the three vector builds. `results/summary_*.csv` are produced by `analyze.py`.

## 8. Notes and limitations

* All numbers come from one laptop (`results/environment.json`); absolute times and the shape of the curves depend on the CPU, its turbo
  behaviour and the OS scheduler.
* No thread affinity is possible on macOS; no hardware performance counters were used, so statements about frequency, caches or
  shared execution units are explanations supported indirectly, not measured.
* A 37 s pilot and two quick runs were used only to estimate durations; they are not reported (`results_pilot/`, `results_quick/` are not
  part of the results).
* Use of AI: see the AI-use section at the end of the report.
