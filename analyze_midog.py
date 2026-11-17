#!/usr/bin/env python3
"""Reproduces every MIDOG number in the manuscript.
usage: python analyze_midog.py results/midog2021_seed42.jsonl,results/midog2021_seed100.jsonl,results/midog2021_seed2024.jsonl results/midog2021_stain_aug_seed42.jsonl [out.json]
Bootstrap: images are resampled with replacement within each scanner (each MIDOG image is a separate case);
counts are averaged over seeds before pooling. 4000 resamples, seed 0."""
import json, sys
import numpy as np

flip = [json.loads(l) for f in sys.argv[1].split(",") for l in open(f)]
hed = [json.loads(l) for l in open(sys.argv[2])]
S = ["XR", "S360", "CS2"]
f1 = lambda c: 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1e-9)
rng = np.random.default_rng(0)
out = {}


def cell_counts(rows, train, T, seed=None, mode="iou"):
    c = np.zeros(3)
    for r in rows:
        if r["train"] == train and (seed is None or r["seed"] == seed):
            m = r["test"][T][mode]
            c += [m["tp"], m["fp"], m["fn"]]
    return c


def frames(rows, train, T, mode="iou"):
    d = {}
    for r in rows:
        if r["train"] == train:
            for k, v in r["test"][T][mode]["frames"].items():
                d.setdefault(k, []).append(v)
    return {k: np.mean(v, 0) for k, v in d.items()}


seeds = sorted({r["seed"] for r in flip})
print("seeds", seeds, "runs", len(flip), "stain-aug runs", len(hed))
print("\n== Table: pooled over folds; mean +- SD over seeds ==")
tab = {}
for tr in S + ["Mixed"]:
    for T in S:
        per = [cell_counts(flip, tr, T, s) for s in seeds]
        F = [f1(c) for c in per]
        P = [c[0] / (c[0] + c[1]) for c in per]
        R = [c[0] / (c[0] + c[2]) for c in per]
        ap = np.mean([np.mean([r["test"][T]["iou"]["ap"] for r in flip if r["train"] == tr and r["seed"] == s]) for s in seeds])
        tab[f"{tr}->{T}"] = dict(f1=float(np.mean(F)), f1_sd=float(np.std(F)), p=float(np.mean(P)), r=float(np.mean(R)), ap=float(ap))
        print(f"{tr:5}->{T:5} F1 {np.mean(F):.3f} +- {np.std(F):.3f}  P {np.mean(P):.2f}  R {np.mean(R):.2f}  AP50 {ap:.3f}")
out["table"] = tab

names = {T: sorted(frames(flip, "XR", T)) for T in S}
A = {(a, T): frames(flip, a, T) for a in S + ["Mixed"] for T in S}
Ah = {(a, T): frames(hed, a, T) for a in S for T in S}


def agg(D, a, T, idx):
    return np.array([D[(a, T)][names[T][i]] for i in idx]).sum(0)


def F(D, pairs, s):
    return f1(sum(agg(D, a, T, s[T]) for a, T in pairs))


ind = [(T, T) for T in S]
cr = [(a, T) for T in S for a in S if a != T]
mx = [("Mixed", T) for T in S]


def boot(fn, B=4000):
    n = {T: len(names[T]) for T in S}
    e = fn({T: np.arange(n[T]) for T in S})
    v = [fn({T: rng.integers(n[T], size=n[T]) for T in S}) for _ in range(B)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(e), float(lo), float(hi)


print("\n== Contrasts (pooled F1 difference, image bootstrap) ==")
con = {}
con["in_minus_cross"] = boot(lambda s: F(A, ind, s) - F(A, cr, s))
con["mixed_minus_in"] = boot(lambda s: F(A, mx, s) - F(A, ind, s))
con["mixed_minus_cross"] = boot(lambda s: F(A, mx, s) - F(A, cr, s))
full = {T: np.arange(len(names[T])) for T in S}
con["F1_in_domain"], con["F1_cross"], con["F1_mixed"] = F(A, ind, full), F(A, cr, full), F(A, mx, full)
for T in S:
    con[f"gap_{T}"] = boot(lambda s, T=T: F(A, [(T, T)], s) - F(A, [(a, T) for a in S if a != T], s))
    con[f"mixed_minus_in_{T}"] = boot(lambda s, T=T: F(A, [("Mixed", T)], s) - F(A, [(T, T)], s))
for k, v in con.items():
    print(k, v if isinstance(v, float) else tuple(round(x, 3) for x in v))
per_seed = {}
for s in seeds:
    c = lambda pairs: f1(sum(cell_counts(flip, a, T, s) for a, T in pairs))
    per_seed[s] = dict(gap=c(ind) - c(cr), mixed_in=c(mx) - c(ind))
    print("seed", s, {k: round(v, 3) for k, v in per_seed[s].items()})
out["contrasts"], out["per_seed"] = con, {str(k): v for k, v in per_seed.items()}

print("\n== Cross-scanner precision/recall asymmetry (mean over seeds) ==")
print("in-domain mean AP50 %.3f ; cross-scanner mean AP50 %.3f" % (
    np.mean([tab[f"{T}->{T}"]["ap"] for T in S]), np.mean([tab[f"{a}->{T}"]["ap"] for a in S for T in S if a != T])))
out["ap_in"] = float(np.mean([tab[f"{T}->{T}"]["ap"] for T in S]))
out["ap_cross"] = float(np.mean([tab[f"{a}->{T}"]["ap"] for a in S for T in S if a != T]))
print("thresholds chosen on validation (mean):", {tr: round(float(np.mean([r["thr"]["iou"] for r in flip if r["train"] == tr])), 3) for tr in S + ["Mixed"]})

print("\n== Centroid criterion (25 px) vs IoU: max |dF1| over 12 cells ==")
d = []
for tr in S + ["Mixed"]:
    for T in S:
        d.append(abs(np.mean([f1(cell_counts(flip, tr, T, s, "dist")) for s in seeds]) - tab[f"{tr}->{T}"]["f1"]))
print("max", round(max(d), 3)); out["max_dist_diff"] = float(max(d))

print("\n== Stain augmentation (seed 42 only; compared with flip-only seed 42) ==")
fl42 = [r for r in flip if r["seed"] == 42]
B42 = {(a, T): frames(fl42, a, T) for a in S for T in S}
def Fx(D, pairs, s):
    return f1(sum(agg(D, a, T, s[T]) for a, T in pairs))
hc = {}
hc["in_domain"] = boot(lambda s: Fx(Ah, ind, s) - Fx(B42, ind, s))
hc["cross"] = boot(lambda s: Fx(Ah, cr, s) - Fx(B42, cr, s))
hc["gap_change"] = boot(lambda s: (Fx(Ah, ind, s) - Fx(Ah, cr, s)) - (Fx(B42, ind, s) - Fx(B42, cr, s)))
hc["F1"] = dict(in_flip=Fx(B42, ind, full), in_hed=Fx(Ah, ind, full), cr_flip=Fx(B42, cr, full), cr_hed=Fx(Ah, cr, full))
for k, v in hc.items():
    print(k, v if isinstance(v, dict) else tuple(round(x, 3) for x in v))
better = worse = 0
for a in S:
    for T in S:
        for f in range(5):
            rb = [r for r in fl42 if r["train"] == a and r["fold"] == f][0]["test"][T]["iou"]["f1"]
            rh = [r for r in hed if r["train"] == a and r["fold"] == f][0]["test"][T]["iou"]["f1"]
            better += rh > rb; worse += rh < rb
print("fold-level cells: stain better", better, "worse", worse)
hc["better"], hc["worse"] = int(better), int(worse)
out["stain"] = hc
out["secs_mean_flip"] = float(np.mean([r["secs"] for r in flip])); out["secs_mean_hed"] = float(np.mean([r["secs"] for r in hed]))
print("mean seconds per run: flip %.0f  stain-aug %.0f" % (out["secs_mean_flip"], out["secs_mean_hed"]))
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"), indent=1)
