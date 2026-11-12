# Big Data (ULPGC 2026-2027) - Individual Assignment 3: parallel (OpenMP) blocked matrix multiplication
# Usage:  make build | inputs | test | env | bench-quick | pilot | bench | analyze | report
# Requires: Apple clang, Homebrew libomp, python venv (make venv).

PY       ?= .venv/bin/python
CC        = clang
OMP      ?= $(shell brew --prefix libomp)
# Same arithmetic as Assignments 1-2: -O2, no -march=native, FMA contraction disabled -> bit-identical results.
CFLAGS   ?= -O2 -std=c11 -ffp-contract=off -Wall -Wextra
OMPFLAGS  = -Xpreprocessor -fopenmp -I$(OMP)/include -L$(OMP)/lib -lomp
SIZES    := 16 32 64 100 250 500 1000 1500 2000 3000 4000

.PHONY: all venv build build-vec inputs test env bench-quick pilot bench bench-small analyze vecinfo report package clean

all: build inputs test bench vecinfo analyze report

venv:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

build:
	mkdir -p build
	$(CC) $(CFLAGS) $(OMPFLAGS) -o build/matmul3 src/c/matmul3.c

# Optional vectorisation extension: same source and flags, only the SIMD code generation changes.
#   scalar: auto-vectorisation disabled; sse2: default x86-64 target (identical to build/matmul3); avx2: 256-bit AVX2.
VECBASE   = -O2 -std=c11 -ffp-contract=off -Wall -Wextra
build-vec:
	mkdir -p build/vec
	$(CC) $(VECBASE) -fno-vectorize -fno-slp-vectorize $(OMPFLAGS) -o build/vec/matmul3_scalar src/c/matmul3.c
	$(CC) $(VECBASE) $(OMPFLAGS) -o build/vec/matmul3_sse2 src/c/matmul3.c
	$(CC) $(VECBASE) -mavx2 $(OMPFLAGS) -o build/vec/matmul3_avx2 src/c/matmul3.c

inputs:
	$(PY) scripts/gen_inputs.py $(SIZES)

test: build build-vec
	$(PY) scripts/test_correctness.py
	for v in scalar sse2 avx2; do MATMUL_EXE=build/vec/matmul3_$$v TEST_CSV=results/correctness_tests_$$v.csv $(PY) scripts/test_correctness.py || exit 1; done

env:
	$(PY) scripts/env_info.py > /dev/null

# Reduced campaign (about 3 minutes, n <= 1000) written to results_quick/ (does not touch results/).
bench-quick: build inputs test
	RESULTS_DIR=results_quick $(PY) scripts/env_info.py > /dev/null
	$(PY) scripts/run_benchmarks.py --campaign quick --results-dir results_quick

# Short pilot used to estimate the duration of the full campaign (results_pilot/).
pilot: build inputs
	$(PY) scripts/run_benchmarks.py --campaign pilot --results-dir results_pilot

# Full campaign of the report (resumable). Run it plugged in, with other apps closed:  caffeinate -i make bench
bench: build inputs test env
	$(PY) scripts/run_benchmarks.py --campaign main  --resume
	$(PY) scripts/run_benchmarks.py --campaign large --resume
	$(PY) scripts/run_benchmarks.py --campaign ijk   --resume
	$(PY) scripts/run_benchmarks.py --campaign pass2 --resume
	$(PY) scripts/run_benchmarks.py --campaign bsweep --resume
	$(PY) scripts/run_benchmarks.py --campaign vec --resume

# Supplementary small-n campaign, added after the main results (about 1-2 minutes).
bench-small: build inputs
	$(PY) scripts/run_benchmarks.py --campaign small --resume

vecinfo: build-vec
	$(PY) scripts/vec_evidence.py

analyze:
	$(PY) scripts/analyze.py

report:
	cd report && PATH=/Library/TeX/texbin:$$PATH latexmk -pdf -interaction=nonstopmode report.tex

# Submission package: tracked files only (no .git, .venv, build/, inputs, .DS_Store) + the report PDF.
package:
	mkdir -p dist
	git archive --format=zip --prefix=Assignment03_GonzalezFernandez_Materials/ -o dist/Assignment03_GonzalezFernandez_Materials.zip HEAD

clean:
	rm -rf build results_quick results_pilot
