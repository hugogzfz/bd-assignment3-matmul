# Big Data (ULPGC 2026-2027) – Individual Assignment 3
## Parallel matrix multiplication (OpenMP, shared memory) and optional vectorisation

Author: Hugo González Fernández · Bachelor’s Degree in Data Science and Engineering, ULPGC
Repository: https://github.com/hugogzfz/bd-assignment3-matmul

Research question: how does the number of threads affect the runtime, speedup and efficiency of a blocked
dense `float64` matrix multiplication on a 4-core / 8-thread laptop CPU, and from which matrix size does parallel
execution pay off?

> Work in progress: results, analysis and report sections are added in later commits.

## 1. Layout

```
src/c/matmul3.c               seq (A2 blocked kernel, bs = 96), omp (same kernel, rows of C split among threads), ijk (A1 reference)
scripts/gen_inputs.py         generates the input matrices and checks them against data/manifest.json
scripts/test_correctness.py   correctness tests -> results/correctness_tests.csv
scripts/run_benchmarks.py     measurement driver (campaigns pilot, quick, main, large, ijk, pass2)
scripts/env_info.py           environment record -> results/environment.json
data/manifest.json            SHA-256 of every input (the matrices themselves are regenerated, not stored)
```

## 2. Reused material

* Blocked kernel, input generator, test structure and driver structure from
  `github.com/hugogzfz/bd-assignment2-matmul` @ `1aab86d7c5b014435b90e9c4baf7031ba9e05aa0`.
* `ijk` reference kernel and input matrices from `github.com/hugogzfz/bd-assignment1-matmul` @
  `101af1cd5b0fe1ffb2809c63798c99398e2e236b` (identical files, checked with the SHA-256 digests).

## 3. Requirements (as used)

macOS 12.7.6 (x86-64, Intel Core i7-6820HQ), Apple clang 14.0.0, Homebrew `libomp` 23.1.5 (`brew install libomp`;
Apple clang ships without OpenMP), Python 3.12.7 with `numpy 2.4.2 pandas 3.0.0 matplotlib 3.10.8` in a virtual
environment. NumPy is used only for input generation and validation, never inside a timed region.

## 4. Exact commands

```bash
make venv           # python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make build          # clang -O2 -std=c11 -ffp-contract=off -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include -L.../lib -lomp
make inputs         # regenerates data/A_n*.bin, data/B_n*.bin and verifies them against data/manifest.json
make test           # correctness suite (must report 0 failed)
make bench-quick    # reduced run (< 1 min, n <= 1000) -> results_quick/
caffeinate -i make bench   # full campaign -> results/ (resumable)
```

Single run by hand:
`OMP_NUM_THREADS=4 OMP_DYNAMIC=false build/matmul3 omp 1000 5 1 1 data/A_n1000.bin data/B_n1000.bin - 96 4`
(arguments: `method n reps warmup batch A B dumpC|- bs threads`).
