#!/usr/bin/env python3
"""Summary CSVs, LaTeX tables and figures from results/runs.csv and results/raw_measurements.csv.
Median and IQR of the measured samples (warm-ups excluded); speedup and efficiency use medians.
Structure follows scripts/analyze.py of bd-assignment2-matmul @ 1aab86d."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG, TAB = ROOT / "results", ROOT / "figures", ROOT / "report" / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984", "figure.dpi": 110,
                     "savefig.bbox": "tight", "axes.axisbelow": True, "lines.linewidth": 1.6})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#bdbcb6"
# ordinal blue ramp (steps 250 -> 700 of the reference palette), lightest = smallest value
RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]      # categorical slots 1-3
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
THREADS = [1, 2, 3, 4, 6, 8, 16]
PHYS, LOGICAL = 4, 8
BS_MAIN = 96


def ramp(k, total):
    """Pick `total` evenly spaced steps of the ordinal ramp."""
    idx = np.linspace(0, len(RAMP) - 1, total).round().astype(int)
    return RAMP[idx[k]]


def sig(x, k=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    if x == 0:
        return "0"
    d = max(k - 1 - int(math.floor(math.log10(abs(x)))), 0)
    return ("%." + str(d) + "f") % x


def fmt_t(t):
    """Seconds -> compact string with unit chosen per value (ms below 1 s)."""
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return "--"
    if t < 1e-3:
        return sig(t * 1e6) + " $\\mu$s"
    return sig(t * 1e3) + " ms" if t < 1 else sig(t) + " s"


def save_fig(fig, name):
    fig.savefig(FIG / (name + ".pdf"))
    fig.savefig(FIG / (name + ".png"), dpi=160)
    plt.close(fig)


def write_table(name, header, rows, colspec):
    lines = ["\\begin{tabular}{%s}" % colspec, "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += ["\\midrule" if r == "MID" else " & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / (name + ".tex")).write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- loading
def load():
    runs = pd.read_csv(RES / "runs.csv")
    raw = pd.read_csv(RES / "raw_measurements.csv")
    meas = raw[raw.phase == "measured"]
    g = meas.groupby("run_id").time_per_mult_s
    st = pd.DataFrame({"median": g.median(), "q1": g.quantile(0.25), "q3": g.quantile(0.75),
                       "tmin": g.min(), "tmax": g.max(), "nmeas": g.count()})
    runs = runs.merge(st, left_on="run_id", right_index=True, how="left")
    runs["iqr"] = runs.q3 - runs.q1
    runs["rel_iqr"] = runs.iqr / runs["median"]
    runs["maxrss_mib"] = runs.maxrss_bytes / 2 ** 20
    runs["predicted_mib"] = runs.predicted_bytes / 2 ** 20
    runs["gflops"] = 2.0 * runs.n.astype(float) ** 3 / runs["median"] / 1e9
    runs["label"] = np.where(runs.method == "omp", "omp p=" + runs.threads_requested.astype(str), runs.method)
    return runs, raw


def ok(runs, **kw):
    r = runs[runs.status == "ok"]
    for k, v in kw.items():
        r = r[r[k] == v]
    return r


def add_speedup(df, keys):
    """S = T_seq / T_p and E = S / p against the sequential run with the same keys, a rough IQR-based range
    [Q1(T_seq)/Q3(T_p), Q3(T_seq)/Q1(T_p)], and S1 = T_omp(1) / T_p (robustness check against the one-thread run)."""
    seq = df[df.method == "seq"].set_index(keys)[["median", "q1", "q3"]]
    seq.columns = ["tseq", "tseq_q1", "tseq_q3"]
    out = df.join(seq, on=keys)
    out["S"] = out.tseq / out["median"]
    out["E"] = out.S / out.threads_requested
    out["S_lo"] = out.tseq_q1 / out.q3
    out["S_hi"] = out.tseq_q3 / out.q1
    one = df[(df.method == "omp") & (df.threads_requested == 1)].set_index(keys)["median"].rename("tomp1")
    out = out.join(one, on=keys)
    out["S1"] = out.tomp1 / out["median"]
    return out


# ----------------------------------------------------------------------------- main thread study
def thread_study(runs):
    m = ok(runs, campaign="main")
    lg = ok(runs, campaign="large")
    d = pd.concat([add_speedup(m, ["n"]), add_speedup(lg, ["n"])])
    cols = ["campaign", "n", "method", "threads_requested", "oversubscribed", "team_size", "distinct_os_threads",
            "nmeas", "batch_size", "median", "q1", "q3", "iqr", "rel_iqr", "tmin", "tmax", "S", "S_lo", "S_hi", "E", "S1",
            "gflops", "maxrss_mib", "predicted_mib", "correctness", "bitwise_equal_seq", "max_rel_err",
            "cpu_speed_limit_before", "cpu_speed_limit_after"]
    d = d.sort_values(["n", "method", "threads_requested"], key=lambda s: s.map({"seq": 0, "omp": 1}) if s.name == "method" else s)
    d[cols].to_csv(RES / "summary_threads.csv", index=False)
    sizes = sorted(d.n.unique())

    # table: median (IQR) per n and configuration
    hdr = ["$n$", "seq"] + ["$p{=}%d$" % p for p in THREADS]
    rows = []
    for n in sizes:
        r = d[d.n == n]
        row = [str(n)]
        for meth, p in [("seq", 1)] + [("omp", p) for p in THREADS]:
            x = r[(r.method == meth) & (r.threads_requested == p)]
            row.append("%s\\,(%s)" % (fmt_t(x["median"].iloc[0]), sig(100 * x.rel_iqr.iloc[0], 2) + "\\%")
                       if len(x) else "--")
        rows.append(row)
    write_table("threads_time", hdr, rows, "r" + "l" * 8)

    # table: S and E
    hdr = ["$n$"] + ["$p{=}%d$" % p for p in THREADS]
    rows = []
    for n in sizes:
        r = d[(d.n == n) & (d.method == "omp")]
        row = [str(n)]
        for p in THREADS:
            x = r[r.threads_requested == p]
            row.append("%.2f / %.2f" % (x.S.iloc[0], x.E.iloc[0]) if len(x) else "--")
        rows.append(row)
    write_table("threads_speedup", hdr, rows, "r" + "c" * 7)

    # overhead of the parallel version with one thread (H5)
    ov = d[(d.method == "omp") & (d.threads_requested == 1)][["n", "median", "tseq", "q1", "q3", "tseq_q1", "tseq_q3"]].copy()
    ov["ratio_p1_seq"] = ov["median"] / ov.tseq
    ov["within_5pct"] = (ov.ratio_p1_seq - 1).abs() <= 0.05
    ov.to_csv(RES / "summary_overhead_p1.csv", index=False)
    write_table("overhead_p1", ["$n$", "$T_{seq}$ (IQR)", "$T_{omp,1}$ (IQR)", "$T_{omp,1}/T_{seq}$"],
                [[str(int(r.n)), "%s (%s)" % (fmt_t(r.tseq), fmt_t(r.tseq_q3 - r.tseq_q1)),
                  "%s (%s)" % (fmt_t(r.median), fmt_t(r.q3 - r.q1)), sig(r.ratio_p1_seq)] for r in ov.itertuples()],
                "rlll")

    # break-even and best configuration per n
    be = []
    for n in sizes:
        r = d[(d.n == n) & (d.method == "omp")]
        best = r.loc[r.S.idxmax()]
        be.append(dict(n=n, best_p=int(best.threads_requested), best_S=best.S, best_E=best.E,
                       best_S_lo=best.S_lo, best_S_hi=best.S_hi,
                       S2=r[r.threads_requested == 2].S.max(), S4=r[r.threads_requested == 4].S.max(),
                       S8=r[r.threads_requested == 8].S.max(), S16=r[r.threads_requested == 16].S.max(),
                       S8_over_S4=(r[r.threads_requested == 8].S.max() / r[r.threads_requested == 4].S.max()),
                       S16_over_S8=(r[r.threads_requested == 16].S.max() / r[r.threads_requested == 8].S.max())))
    be = pd.DataFrame(be)
    be.to_csv(RES / "summary_best.csv", index=False)
    write_table("best", ["$n$", "best $p$", "$S$ (IQR range)", "$E$", "$S(8)/S(4)$", "$S(16)/S(8)$"],
                [[str(r.n), str(r.best_p), "%s (%s--%s)" % (sig(r.best_S), sig(r.best_S_lo), sig(r.best_S_hi)),
                  sig(r.best_E, 2), sig(r.S8_over_S4), sig(r.S16_over_S8) if not math.isnan(r.S16_over_S8) else "--"]
                 for r in be.itertuples()], "rrllll")

    # memory
    mem = d[["n", "method", "threads_requested", "maxrss_mib", "predicted_mib"]]
    mem.to_csv(RES / "summary_memory.csv", index=False)
    rows = []
    for n in sizes:
        r = d[d.n == n]
        s = r[r.method == "seq"].iloc[0]
        o = r[r.method == "omp"]
        rows.append([str(n), sig(s.predicted_mib), sig(s.maxrss_mib), sig(o.maxrss_mib.min()) + "--" + sig(o.maxrss_mib.max()),
                     sig(o.maxrss_mib.max() - s.maxrss_mib)])
    write_table("memory", ["$n$", "$3\\cdot 8n^2$ (MiB)", "seq (MiB)", "omp, all $p$ (MiB)", "max(omp)$-$seq (MiB)"],
                rows, "rllll")

    # ------------------------------------------------ figures
    # Fig 1: runtime vs n
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    sel = [1, 2, 4, 8, 16]
    s = d[d.method == "seq"].sort_values("n")
    ax.errorbar(s.n, s["median"], yerr=[s["median"] - s.q1, s.q3 - s["median"]], color=INK, ls="--", marker="o",
                ms=4, lw=1.4, label="seq (baseline)", capsize=2)
    for k, p in enumerate(sel):
        x = d[(d.method == "omp") & (d.threads_requested == p)].sort_values("n")
        ax.errorbar(x.n, x["median"], yerr=[x["median"] - x.q1, x.q3 - x["median"]], color=ramp(k, len(sel)),
                    marker=MARKERS[k], ms=4, label="omp p=%d%s" % (p, " (oversubscribed)" if p > LOGICAL else ""),
                    capsize=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("matrix size n"); ax.set_ylabel("time per multiplication (s), median")
    ax.set_title("Runtime vs n (error bars: IQR)")
    ax.set_xticks(sizes); ax.set_xticklabels([str(x) for x in sizes], fontsize=7.5)
    ax.minorticks_off()
    ax.legend(frameon=False)
    save_fig(fig, "fig1_runtime")

    # Fig 2 & 3: speedup and efficiency vs p, one line per n
    for name, col, ylab, title in (("fig2_speedup", "S", "speedup S(p) = T_seq / T_p", "Speedup vs threads"),
                                   ("fig3_efficiency", "E", "efficiency E(p) = S(p) / p", "Efficiency vs threads")):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.axvspan(LOGICAL, 17, color="#f0efec", zorder=0)
        ax.text(11.6, 0.97, "oversubscription", transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=MUTED, fontsize=7.5)
        for v, t in ((PHYS, "4 physical cores"), (LOGICAL, "8 logical")):
            ax.axvline(v, color=GRID, lw=1, ls=":")
            ax.text(v, 0.03, " " + t, transform=ax.get_xaxis_transform(), color=MUTED, fontsize=7, rotation=90,
                    va="bottom")
        if col == "S":
            ax.plot([1, 16], [1, 16], color=GRID, lw=1, ls="--", label="ideal S = p")
        else:
            ax.axhline(1, color=GRID, lw=1, ls="--", label="ideal E = 1")
        for k, n in enumerate(sizes):
            x = d[(d.method == "omp") & (d.n == n)].sort_values("threads_requested")
            ax.plot(x.threads_requested, x[col], color=ramp(k, len(sizes)), marker=MARKERS[k % 7], ms=4,
                    label="n = %d" % n)
        ax.set_xscale("log", base=2)
        ax.set_xticks(THREADS); ax.set_xticklabels([str(p) for p in THREADS]); ax.minorticks_off()
        ax.set_xlim(0.9, 17)
        if col == "S":
            ax.set_ylim(0, max(5.0, d.S.max() * 1.15))
        else:
            ax.set_ylim(0, 1.25)
        ax.set_xlabel("threads p"); ax.set_ylabel(ylab); ax.set_title(title + " (medians)")
        ax.legend(frameon=False, ncol=2, loc="upper left" if col == "S" else "lower left")
        save_fig(fig, name)

    # Fig 4: speedup vs n (when does parallelism pay?)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.axhline(1, color=INK, lw=1, ls="--")
    ax.text(sizes[2], 1.06, "S = 1 (no gain)", color=MUTED, fontsize=7.5)
    for k, p in enumerate(sel):
        x = d[(d.method == "omp") & (d.threads_requested == p)].sort_values("n")
        ax.errorbar(x.n, x.S, yerr=[x.S - x.S_lo, x.S_hi - x.S], color=ramp(k, len(sel)), marker=MARKERS[k], ms=4,
                    capsize=2, label="p=%d%s" % (p, " (oversubscribed)" if p > LOGICAL else ""))
    ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels([str(x) for x in sizes], fontsize=7.5)
    ax.minorticks_off()
    ax.set_xlabel("matrix size n"); ax.set_ylabel("speedup S(p) (median ratio)")
    ax.set_title("Speedup vs n (bars: range from the IQRs of T_seq and T_p)")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save_fig(fig, "fig4_speedup_vs_n")

    # Fig 5: memory
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(sizes, [3 * 8 * n * n / 2 ** 20 for n in sizes], color=GRID, ls="--", lw=1.2, label="3·8n² (arrays A, B, C)")
    for k, (meth, p, c, mk) in enumerate((("seq", 1, INK, "o"), ("omp", 8, CAT[0], "s"), ("omp", 16, CAT[1], "^"))):
        x = d[(d.method == meth) & (d.threads_requested == p)].sort_values("n")
        ax.plot(x.n, x.maxrss_mib, color=c, marker=mk, ms=4, lw=1.4 if k else 1.2, ls="-",
                label=("seq" if meth == "seq" else "omp p=%d" % p))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sizes); ax.set_xticklabels([str(x) for x in sizes], fontsize=7.5); ax.minorticks_off()
    ax.set_xlabel("matrix size n"); ax.set_ylabel("peak RSS (MiB), ru_maxrss")
    ax.set_title("Peak resident memory per process")
    ax.legend(frameon=False)
    save_fig(fig, "fig5_memory")
    return d


# ----------------------------------------------------------------------------- second pass at n = 2000
def second_pass(runs, raw):
    p1 = add_speedup(ok(runs, campaign="main", n=2000), ["n"])
    p2 = add_speedup(ok(runs, campaign="pass2", n=2000), ["n"])
    if p2.empty:
        return None
    key = ["method", "threads_requested"]
    c = p1.set_index(key)[["median", "rel_iqr", "S", "start_utc"]].join(
        p2.set_index(key)[["median", "rel_iqr", "S", "start_utc"]], lsuffix="_pass1", rsuffix="_pass2")
    c["diff_pct"] = 100 * (c.median_pass2 / c.median_pass1 - 1)
    c = c.reset_index().sort_values(["method", "threads_requested"], ascending=[False, True])
    c.to_csv(RES / "summary_pass2.csv", index=False)
    write_table("pass2", ["config", "pass 1", "pass 2", "$\\Delta$ (\\%)", "$S$ pass 1", "$S$ pass 2"],
                [["seq" if r.method == "seq" else "$p{=}%d$" % r.threads_requested,
                  "%s (%s\\%%)" % (fmt_t(r.median_pass1), sig(100 * r.rel_iqr_pass1, 2)),
                  "%s (%s\\%%)" % (fmt_t(r.median_pass2), sig(100 * r.rel_iqr_pass2, 2)),
                  ("%+.1f" % r.diff_pct), "--" if r.method == "seq" else sig(r.S_pass1),
                  "--" if r.method == "seq" else sig(r.S_pass2)] for r in c.itertuples()], "llllll")

    meas = raw[(raw.phase == "measured") & (raw.n == 2000) & raw.campaign.isin(["main", "pass2"])]
    configs = [("seq", 1)] + [("omp", p) for p in THREADS]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for j, (camp, colr, lab) in enumerate((("main", CAT[0], "pass 1 (main campaign)"),
                                           ("pass2", CAT[1], "pass 2 (end of campaign)"))):
        for i, (meth, p) in enumerate(configs):
            v = meas[(meas.campaign == camp) & (meas.method == meth) & (meas.threads_requested == p)].time_per_mult_s
            xs = i + (j - 0.5) * 0.3 + np.linspace(-0.06, 0.06, len(v))
            ax.scatter(xs, v, s=16, color=colr, edgecolor="white", linewidth=0.6, label=lab if i == 0 else None,
                       zorder=3)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.6, 0.8, 1.0, 1.5, 2.0, 3.0]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(["seq"] + ["p=%d" % p for p in THREADS])
    ax.set_ylabel("time per multiplication (s)")
    ax.set_title("n = 2000: every measured sample, two independent passes")
    ax.legend(frameon=False)
    save_fig(fig, "fig6_pass2")
    return c


def within_process_drift(raw):
    """Time of the last measured sample relative to the warm-up sample of the same process (long runs only)."""
    x = raw[raw.campaign.isin(["main", "large"]) & (raw.n >= 2000)].copy()
    x["k"] = np.where(x.phase == "warmup", -1, x.repetition)
    t = x.pivot_table(index=["n", "method", "threads_requested"], columns="k", values="time_per_mult_s")
    last = t.drop(columns=[-1]).ffill(axis=1).iloc[:, -1]
    out = pd.DataFrame({"warmup_s": t[-1], "first_measured_s": t[0], "last_measured_s": last,
                        "last_over_warmup": last / t[-1]}).reset_index()
    out.to_csv(RES / "summary_drift.csv", index=False)
    rows = []
    for n in sorted(out.n.unique()):
        r = out[out.n == n]
        row = [str(n)]
        for meth, p in [("seq", 1)] + [("omp", p) for p in THREADS]:
            v = r[(r.method == meth) & (r.threads_requested == p)]
            row.append("%+.1f" % (100 * (v.last_over_warmup.iloc[0] - 1)) if len(v) else "--")
        rows.append(row)
    write_table("drift", ["$n$", "seq"] + ["$p{=}%d$" % p for p in THREADS], rows, "r" + "r" * 8)
    return out


def small_sizes(runs):
    """Supplementary campaign (added after the main results): n = 16, 32, 64."""
    sm = ok(runs, campaign="small")
    if sm.empty:
        return None
    sm = add_speedup(sm, ["n"])
    # fork/join cost estimate per parallel region: T_p - T_seq / p (assumes the work itself divides perfectly)
    sm["region_overhead_s"] = np.where(sm.method == "omp", sm["median"] - sm.tseq / sm.threads_requested, np.nan)
    sm[["n", "method", "threads_requested", "batch_size", "nmeas", "median", "q1", "q3", "rel_iqr", "S", "S_lo", "S_hi",
        "E", "S1", "region_overhead_s", "correctness", "bitwise_equal_seq"]].sort_values(
        ["n", "method", "threads_requested"], ascending=[True, False, True]).to_csv(RES / "summary_small.csv", index=False)
    rows = []
    for n in sorted(sm.n.unique()):
        r = sm[sm.n == n]
        s0 = r[r.method == "seq"].iloc[0]
        row = [str(n), "%s (%s\\%%)" % (fmt_t(s0["median"]), sig(100 * s0.rel_iqr, 2))]
        for p in (1, 2, 4, 8):
            x = r[(r.method == "omp") & (r.threads_requested == p)].iloc[0]
            row.append("%s [%.2f]" % (fmt_t(x["median"]), x.S))
        rows.append(row)
    write_table("small", ["$n$", "seq", "$p{=}1$", "$p{=}2$", "$p{=}4$", "$p{=}8$"], rows, "rlllll")

    main = add_speedup(ok(runs, campaign="main"), ["n"])
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.axhline(1, color=INK, lw=1, ls="--")
    ax.axvspan(10, 80, color="#f0efec", zorder=0)
    ax.text(28, 0.97, "supplementary\n(n = 16-64)", transform=ax.get_xaxis_transform(), ha="center", va="top",
            color=MUTED, fontsize=7)
    for k, p in enumerate((2, 4, 8)):
        c = ramp(k, 3)
        x = sm[(sm.method == "omp") & (sm.threads_requested == p)].sort_values("n")
        y = main[(main.method == "omp") & (main.threads_requested == p)].sort_values("n")
        ax.plot(y.n, y.S, color=c, marker=MARKERS[k], ms=4, label="p=%d" % p)
        ax.plot(x.n, x.S, color=c, marker=MARKERS[k], ms=5, mfc="white", ls="--")
    ax.set_xscale("log")
    xt = [16, 32, 64, 100, 250, 500, 1000, 2000, 3000]
    ax.set_xticks(xt); ax.set_xticklabels([str(v) for v in xt], fontsize=7.5); ax.minorticks_off()
    ax.set_xlim(12, 3600); ax.set_ylim(0, 4.5)
    ax.text(3400, 1.05, "S = 1", ha="right", color=MUTED, fontsize=7.5)
    ax.set_xlabel("matrix size n"); ax.set_ylabel("speedup S(p) (median ratio)")
    ax.set_title("Speedup vs n, main campaign (filled) and supplementary small sizes (open)")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, "fig9_small")
    return sm


# ----------------------------------------------------------------------------- extras: ijk, bsweep, vec
def ijk_reference(runs):
    ij = ok(runs, campaign="ijk")
    if ij.empty:
        return None
    m = ok(runs, campaign="main")
    rows, out = [], []
    for r in ij.sort_values("n").itertuples():
        s = m[(m.n == r.n) & (m.method == "seq")].iloc[0]
        o4 = m[(m.n == r.n) & (m.method == "omp") & (m.threads_requested == 4)].iloc[0]
        o8 = m[(m.n == r.n) & (m.method == "omp") & (m.threads_requested == 8)].iloc[0]
        out.append(dict(n=r.n, t_ijk=r.median, t_seq=s["median"], t_omp4=o4["median"], t_omp8=o8["median"],
                        ijk_over_seq=r.median / s["median"], ijk_over_omp8=r.median / o8["median"]))
        rows.append([str(r.n), "%s (%s\\%%)" % (fmt_t(r.median), sig(100 * r.rel_iqr, 2)), fmt_t(s["median"]),
                     fmt_t(o8["median"]), sig(r.median / s["median"]), sig(r.median / o8["median"])])
    pd.DataFrame(out).to_csv(RES / "summary_ijk.csv", index=False)
    write_table("ijk", ["$n$", "ijk (A1)", "seq (blocked)", "omp $p{=}8$", "ijk/seq", "ijk/omp$_8$"], rows, "rlllll")
    return out


def bsweep(runs):
    b = ok(runs, campaign="bsweep")
    if b.empty:
        return None
    b = add_speedup(b, ["n", "block_size"])
    b[["block_size", "method", "threads_requested", "median", "q1", "q3", "rel_iqr", "S", "E", "correctness",
       "bitwise_equal_seq"]].sort_values(["block_size", "method"]).to_csv(RES / "summary_bsweep.csv", index=False)
    rows = []
    for bs in sorted(b.block_size.unique()):
        s = b[(b.block_size == bs) & (b.method == "seq")].iloc[0]
        o = b[(b.block_size == bs) & (b.method == "omp")].iloc[0]
        rows.append([str(bs), "%s (%s\\%%)" % (fmt_t(s["median"]), sig(100 * s.rel_iqr, 2)),
                     "%s (%s\\%%)" % (fmt_t(o["median"]), sig(100 * o.rel_iqr, 2)), sig(o.S)])
    write_table("bsweep", ["tile $b$", "seq", "omp $p{=}4$", "$S(4)$"], rows, "rlll")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for k, (meth, lab) in enumerate((("seq", "seq"), ("omp", "omp p=4"))):
        x = b[b.method == meth].sort_values("block_size")
        ax.errorbar(x.block_size, x["median"], yerr=[x["median"] - x.q1, x.q3 - x["median"]], color=CAT[k],
                    marker=MARKERS[k], ms=4, capsize=2, label=lab)
    ax.axvline(BS_MAIN, color=GRID, ls=":", lw=1)
    ax.set_xscale("log", base=2); ax.set_xticks(sorted(b.block_size.unique()))
    ax.set_xticklabels([str(v) for v in sorted(b.block_size.unique())]); ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("tile size b"); ax.set_ylabel("time per multiplication (s)")
    ax.set_title("Tile-size check, n = 2000 (dotted: b = 96 used)")
    ax.legend(frameon=False)
    save_fig(fig, "fig8_bsweep")
    return b


VEC = [("scalar", "scalar (-fno-vectorize)"), ("sse2-vec", "SSE2, 128-bit (default -O2)"), ("avx2", "AVX2, 256-bit (-mavx2)")]


def vectorisation(runs):
    v = ok(runs, campaign="vec")
    if v.empty:
        return None
    v = add_speedup(v, ["n", "variant"])                # thread speedup inside each build
    sc = v[v.variant == "scalar"].set_index(["n", "method", "threads_requested"])["median"].rename("t_scalar")
    v = v.join(sc, on=["n", "method", "threads_requested"])
    v["simd_gain"] = v.t_scalar / v["median"]           # same n, same p: only the SIMD code changes
    sc_seq = v[(v.variant == "scalar") & (v.method == "seq")].set_index("n")["median"].rename("t_scalar_seq")
    v = v.join(sc_seq, on="n")
    v["S_vs_scalar_seq"] = v.t_scalar_seq / v["median"]
    v[["n", "variant", "method", "threads_requested", "median", "q1", "q3", "rel_iqr", "S", "E", "simd_gain",
       "S_vs_scalar_seq", "gflops", "correctness", "bitwise_equal_seq"]].sort_values(
        ["n", "variant", "method", "threads_requested"]).to_csv(RES / "summary_vec.csv", index=False)
    rows = []
    for n in sorted(v.n.unique()):
        for var, lab in VEC:
            r = v[(v.n == n) & (v.variant == var)]
            row = [str(n), lab.split(" (")[0]]
            for meth, p in [("seq", 1), ("omp", 1), ("omp", 2), ("omp", 4), ("omp", 8)]:
                x = r[(r.method == meth) & (r.threads_requested == p)].iloc[0]
                row.append("%s [%s]" % (fmt_t(x["median"]), sig(x.simd_gain)))
            rows.append(row)
        rows.append("MID")
    write_table("vec", ["$n$", "build", "seq", "$p{=}1$", "$p{=}2$", "$p{=}4$", "$p{=}8$"], rows[:-1], "rllllll")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=False)
    for ax, n in zip(axes, sorted(v.n.unique())):
        for k, (var, lab) in enumerate(VEC):
            x = v[(v.n == n) & (v.variant == var) & (v.method == "omp")].sort_values("threads_requested")
            ax.errorbar(x.threads_requested, x.gflops, color=CAT[k], marker=MARKERS[k], ms=4, label=lab)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
        ax.minorticks_off(); ax.set_ylim(bottom=0)
        ax.set_xlabel("threads p"); ax.set_title("n = %d" % n)
    axes[0].set_ylabel("GFLOP/s (2n³ / median time)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Vectorisation x threads (OpenMP version; same source, only SIMD code generation changes)",
                 fontsize=9, y=1.04)
    save_fig(fig, "fig7_vec")
    return v


# ----------------------------------------------------------------------------- environment, checks, limits
def checks(runs):
    r = runs[runs.campaign.isin(["main", "large", "ijk", "pass2", "bsweep", "vec", "small"])]
    okr = r[r.status == "ok"]
    omp = okr[okr.method == "omp"]
    info = {
        "processes": int(len(r)),
        "status_counts": r.status.value_counts().to_dict(),
        "correctness_pass": int((okr.correctness == "pass").sum()),
        "bitwise_equal_seq_true": int((okr.bitwise_equal_seq.astype(str) == "True").sum()),
        "bitwise_checked": int(okr.bitwise_equal_seq.notna().sum()),
        "omp_team_equals_requested": int((omp.team_size == omp.threads_requested).sum()),
        "omp_distinct_os_threads_equal_requested": int((omp.distinct_os_threads == omp.threads_requested).sum()),
        "omp_processes": int(len(omp)),
        "max_rel_err_overall": float(okr.max_rel_err.max()),
        "cpu_speed_limit_min": int(min(okr.cpu_speed_limit_before.min(), okr.cpu_speed_limit_after.min())),
        "runs_with_speed_limit_below_100": int(((okr.cpu_speed_limit_before < 100) | (okr.cpu_speed_limit_after < 100)).sum()),
        "largest_n_ok": int(okr.n.max()),
        "measured_samples": int(okr.nmeas.sum()),
        "campaign_first_start_utc": str(r.start_utc.min()), "campaign_last_start_utc": str(r.start_utc.max()),
    }
    tests = pd.concat([pd.read_csv(f).assign(binary=f.stem.replace("correctness_tests", "") or "_primary")
                       for f in sorted(RES.glob("correctness_tests*.csv"))])
    info["tests_total"] = int(len(tests))
    info["tests_failed"] = int((tests.result != "pass").sum())
    info["tests_per_binary"] = tests.groupby("binary").size().to_dict()
    (RES / "summary_checks.json").write_text(json.dumps(info, indent=2))
    env = json.loads((RES / "environment.json").read_text())
    rows = [["CPU", env["cpu_model"].replace("(R)", "").replace("(TM)", "")],
            ["Cores / threads", "%d physical, %d logical (Hyper-Threading), 1 package" %
             (env["physical_cores"], env["logical_processors"])],
            ["Caches", "L1d %d KiB and L2 %d KiB per core, L3 %d MiB shared, line %d B" %
             (env["cache_L1d_bytes"] // 1024, env["cache_L2_bytes_per_core"] // 1024,
              env["cache_L3_bytes_shared"] // 2 ** 20, env["cache_line_bytes"])],
            ["RAM", "%d GiB" % (env["ram_bytes"] // 2 ** 30)],
            ["Machine / OS", "%s, %s" % (env["model_identifier"], env["os"])],
            ["Compiler", env["compiler"] + ", " + env["libomp"]["version"]],
            ["Python (validation)", "%s, NumPy %s" % (env["python"]["version"], env["python"]["numpy"])]]
    write_table("env", ["Item", "Value"], [[a, b.replace("_", "\\_")] for a, b in rows], "lp{11cm}")
    return info


def main():
    runs, raw = load()
    d = thread_study(runs)
    second_pass(runs, raw)
    within_process_drift(raw)
    small_sizes(runs)
    ijk_reference(runs)
    bsweep(runs)
    vectorisation(runs)
    info = checks(runs)
    print(json.dumps(info, indent=2))
    print("summaries in results/, tables in report/tables/, figures in figures/")
    return d


if __name__ == "__main__":
    main()
