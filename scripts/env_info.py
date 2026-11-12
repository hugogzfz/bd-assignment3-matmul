#!/usr/bin/env python3
"""Records the hardware / software environment of the experiment (macOS) in results/environment.json.
Structure follows scripts/env_info.py of bd-assignment2-matmul @ 1aab86d; the Linux sources (/proc, lscpu) are
replaced by sysctl, sw_vers and system_profiler."""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return "unavailable: %s" % e


def sysctl(key):
    v = sh("sysctl -n %s 2>/dev/null" % key)
    return int(v) if v.isdigit() else (v or None)


def main():
    import matplotlib
    import numpy
    import pandas
    feats = (str(sysctl("machdep.cpu.features")) + " " + str(sysctl("machdep.cpu.leaf7_features"))).split()
    omp_prefix = sh("brew --prefix libomp")
    info = {
        "cpu_model": sysctl("machdep.cpu.brand_string"),
        "architecture": platform.machine(),
        "physical_cores": sysctl("hw.physicalcpu"),
        "logical_processors": sysctl("hw.logicalcpu"),
        "threads_per_core": sysctl("machdep.cpu.thread_count") // sysctl("machdep.cpu.core_count"),
        "packages": sysctl("hw.packages"),
        "cpu_nominal_frequency_hz": sysctl("hw.cpufrequency"),
        "cache_line_bytes": sysctl("hw.cachelinesize"),
        "cache_L1d_bytes": sysctl("hw.l1dcachesize"),
        "cache_L1i_bytes": sysctl("hw.l1icachesize"),
        "cache_L2_bytes_per_core": sysctl("hw.l2cachesize"),
        "cache_L3_bytes_shared": sysctl("hw.l3cachesize"),
        "cpu_features_of_interest": [f for f in ("SSE2", "SSE4.2", "AVX1.0", "AVX2", "FMA", "AVX512F") if f in feats],
        "ram_bytes": sysctl("hw.memsize"),
        "model_identifier": sysctl("hw.model"),
        "os": "macOS %s (%s)" % (sh("sw_vers -productVersion"), sh("sw_vers -buildVersion")),
        "kernel": platform.release(),
        "power_source": sh("pmset -g batt | head -1"),
        "thermal_state_at_record": sh("pmset -g therm"),
        "cpu_affinity_used": ("none: macOS offers no thread pinning API (THREAD_AFFINITY_POLICY is only a hint); "
                              "OMP_PROC_BIND / OMP_PLACES are not set and would not be honoured"),
        "compiler": sh("clang --version | head -1"),
        "libomp": {"prefix": omp_prefix, "version": sh("brew list --versions libomp")},
        "c_flags_primary": ("-O2 -std=c11 -ffp-contract=off -Xpreprocessor -fopenmp -I%s/include -L%s/lib -lomp "
                            "(no -march=native, no -ffast-math)" % (omp_prefix, omp_prefix)),
        "openmp_runtime_settings": {"OMP_NUM_THREADS": "set to p for every run", "OMP_DYNAMIC": "false",
                                    "OMP_PROC_BIND/OMP_PLACES": "unset", "OMP_WAIT_POLICY/KMP_BLOCKTIME": "libomp defaults"},
        "python": {"version": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pandas.__version__,
                   "matplotlib": matplotlib.__version__},
        "note_threads": ("The C kernels use no numerical library. NumPy (validation only, never timed) runs with "
                         "OPENBLAS_NUM_THREADS=1 in the benchmark driver. The benchmark program reports the OpenMP "
                         "team size and the number of OS threads of the process."),
    }
    out = ROOT / os.environ.get("RESULTS_DIR", "results") / "environment.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
