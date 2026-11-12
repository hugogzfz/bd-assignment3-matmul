#!/usr/bin/env python3
"""Generates the dense input matrices (the same files as Assignments 1 and 2) and checks every file against the
SHA-256 digests in data/manifest.json. Dense part of scripts/gen_inputs.py of bd-assignment2-matmul @ 1aab86d.
Usage: gen_inputs.py n1 n2 ..."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

SEED = 20261012
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sizes", nargs="+", type=int)
    args = ap.parse_args()

    mpath = os.path.join(DATA, "manifest.json")
    with open(mpath) as f:
        manifest = json.load(f)
    new, verified, mismatch = 0, 0, []
    for n in args.sizes:
        for which, name in ((0, "A"), (1, "B")):
            base = "%s_n%d.bin" % (name, n)
            path = os.path.join(DATA, base)
            old = manifest["files"].get(base)
            if not (old and os.path.exists(path) and sha256_file(path) == old["sha256"]):
                rng = np.random.default_rng([SEED, n, which])
                rng.random(n * n, dtype=np.float64).astype("<f8").tofile(path)
            digest = sha256_file(path)
            if old is None:
                manifest["files"][base] = {"n": n, "kind": "dense-uniform", "bytes": os.path.getsize(path),
                                           "sha256": digest}
                new += 1
            elif old["sha256"] == digest:
                verified += 1
            else:
                mismatch.append(base)
        print("dense n=%d done" % n, flush=True)

    if not mismatch:
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    print("inputs: %d verified identical to manifest, %d newly recorded, %d DIFFERENT" % (verified, new, len(mismatch)))
    if mismatch:
        print("WARNING: these files do not match the reference digests:", ", ".join(mismatch))
        sys.exit(1)


if __name__ == "__main__":
    main()
