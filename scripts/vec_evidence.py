#!/usr/bin/env python3
"""Vectorisation evidence for the three builds of the optional extension: clang optimisation remarks
(-Rpass=loop-vectorize) and instruction counts in the disassembly (otool -tv) of the blocked kernel.
Writes results/vectorization_evidence.csv, results/vectorization_remarks.txt and report/tables/vecev.tex."""
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OMP = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True).stdout.strip()
BASE = ["-O2", "-std=c11", "-ffp-contract=off", "-Xpreprocessor", "-fopenmp", "-I%s/include" % OMP]
BUILDS = [("scalar", ["-fno-vectorize", "-fno-slp-vectorize"]), ("sse2", []), ("avx2", ["-mavx2"])]
KERNEL = "_mul_blocked_rows"


def kernel_asm(binary):
    out = subprocess.run(["otool", "-tv", str(binary)], capture_output=True, text=True).stdout
    lines, inside = [], False
    for ln in out.splitlines():
        if ln.startswith(KERNEL + ":"):
            inside = True
            continue
        if inside and re.match(r"^_\w+:", ln):
            break
        if inside:
            lines.append(ln)
    return lines


def count(lines, pattern):
    return sum(1 for ln in lines if re.search(pattern, ln))


def main():
    rows, remarks = [], []
    for name, flags in BUILDS:
        r = subprocess.run(["clang"] + BASE + flags + ["-Rpass=loop-vectorize", "-Rpass-missed=loop-vectorize",
                                                       "-c", "-o", "/dev/null", str(ROOT / "src/c/matmul3.c")],
                           capture_output=True, text=True)
        rem = [ln for ln in r.stderr.splitlines() if "remark" in ln]
        remarks.append("== %s %s\n%s\n" % (name, " ".join(flags), "\n".join(rem)))
        kernel_rem = [ln for ln in rem if re.search(r"matmul3\.c:1(0[0-9]|1[0-9]):", ln)]   # kernel lines ~100-110
        asm = kernel_asm(ROOT / "build" / "vec" / ("matmul3_%s" % name))
        rows.append(dict(build=name, flags=" ".join(flags) or "(default -O2)",
                         kernel_remark="; ".join(re.sub(r"^.*remark: ", "", x) for x in kernel_rem) or "none",
                         kernel_instructions=len(asm),
                         packed_mul=count(asm, r"\bv?mulpd\b"), packed_add=count(asm, r"\bv?addpd\b"),
                         scalar_mul=count(asm, r"\bv?mulsd\b"), scalar_add=count(asm, r"\bv?addsd\b"),
                         ymm_operands=count(asm, r"%ymm"), xmm_operands=count(asm, r"%xmm"),
                         fma=count(asm, r"vfmadd")))
    (ROOT / "results" / "vectorization_remarks.txt").write_text("\n".join(remarks))
    with open(ROOT / "results" / "vectorization_evidence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    lines = ["\\begin{tabular}{llrrrrrl}", "\\toprule",
             "build & extra flags & mulpd & addpd & mulsd & addsd & ymm & compiler remark (inner loop) \\\\", "\\midrule"]
    for r in rows:
        rem = r["kernel_remark"].replace("_", "\\_")
        rem = re.sub(r"\s*\[-Rpass[^\]]*\]", "", rem)
        lines.append("%s & %s & %d & %d & %d & %d & %d & %s \\\\" % (
            r["build"], "\\texttt{%s}" % r["flags"].replace("_", "\\_") if r["flags"] != "(default -O2)" else "--",
            r["packed_mul"], r["packed_add"], r["scalar_mul"], r["scalar_add"], r["ymm_operands"],
            "\\parbox[t]{5.2cm}{\\raggedright\\footnotesize %s}" % rem))
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "report" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "report" / "tables" / "vecev.tex").write_text("\n".join(lines) + "\n")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
