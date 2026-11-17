#!/usr/bin/env python3
"""
Cross-scanner mitosis detection on MIDOG 2021 (3 labelled scanners x 50 images = 150 cases) with CASE-LEVEL folds.

Every MIDOG image is a different case, so folds are made by image (balanced by mitosis count) within each scanner.
Protocol (per fold k = 0..4, each fold holds 10 images per scanner):
  test  = fold k, ALL scanners          (no image is ever in train, val and test at once)
  val   = fold k+1 (mod 5), scanner(s) of the training set  -> picks the score threshold
  train = remaining 3 folds, scanner = XR | S360 | CS2 | Mixed (all three)
One trained model is evaluated on the test fold of every scanner, full images, tiled inference.
Only 'mitotic figure' boxes (50x50 px) are targets; 'not mitotic figure' (hard negatives) are not used.
Unlabelled scanners (Leica GT450 in the 2021 training release) are excluded automatically.

Usage (Kaggle, GPU T4):
  python midog_experiment.py run --root /kaggle/input/midog2021-jpeg/midog2021 --out /kaggle/working/runs \
         --train XR,S360,CS2,Mixed --attn none --folds 0,1,2,3,4 --seeds 42 --iters 2000
  python midog_experiment.py aggregate --out /kaggle/working/runs
"""
import argparse
import json
import math
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou, nms

Image.MAX_IMAGE_PIXELS = None
CODE = {"hamamatsu_xr": "XR", "hamamatsu_s360": "S360", "aperio_cs2": "CS2", "leica_gt450": "GT450"}
NFOLDS = 5
SCANNERS = []  # filled from the data (scanners that have mitosis labels)
MODES = ("iou", "dist")


# ----------------------------------------------------------------------------- data
def load_frames(root):
    """Lazy frames: pixels are read with get_img() only when needed (images are ~39 Mpx)."""
    import pandas as pd
    root = Path(root)
    imgs = pd.read_csv(root / "images.csv")
    ann = pd.read_csv(root / "annotations.csv")
    ann = ann[ann.label == "mitotic"]
    frames = []
    for sc_name, g in imgs.groupby("scanner"):
        if g.n_mitotic.sum() == 0:
            print(f"excluding {sc_name}: no mitosis labels")
            continue
        code = CODE[sc_name]
        order = g.sort_values(["n_mitotic", "image"], ascending=[False, True]).reset_index(drop=True)
        for i, r in order.iterrows():  # deal images round-robin by mitosis count -> balanced folds
            b = ann[ann.image == r.image][["x1", "y1", "x2", "y2"]].to_numpy(np.float32).reshape(-1, 4)
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, r.width)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, r.height)
            frames.append(dict(name=f"{code}_{Path(r.image).stem}", scanner=code, slide=i % NFOLDS,
                               path=str(root / "images" / r.image), boxes=b, w=int(r.width), h=int(r.height)))
    if not frames:
        raise SystemExit(f"no labelled images under {root}")
    SCANNERS[:] = sorted({f["scanner"] for f in frames})
    return frames


def get_img(f):
    img = f.get("img")
    if img is None:
        img = np.asarray(Image.open(f["path"]).convert("RGB"))
        assert img.shape[:2] == (f["h"], f["w"]), (f["name"], img.shape, f["w"], f["h"])
    return img


RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float32)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)


def hed_jitter(img, rng, sigma=0.05):
    od = -np.log((img.astype(np.float32) + 1.0) / 256.0)
    hed = od @ HED_FROM_RGB
    hed = hed * rng.uniform(1 - sigma, 1 + sigma, 3) + rng.uniform(-sigma, sigma, 3)
    out = 256.0 * np.exp(-(hed @ RGB_FROM_HED)) - 1.0
    return np.clip(out, 0, 255).astype(np.uint8)


def dihedral(img, boxes, k):
    """k in 0..7: bit0 h-flip, bit1 v-flip, bit2 transpose (boxes x2/y2 exclusive)."""
    c = img.shape[0]
    b = boxes.copy()
    if k & 1:
        img = img[:, ::-1]
        b[:, [0, 2]] = c - b[:, [2, 0]]
    if k & 2:
        img = img[::-1]
        b[:, [1, 3]] = c - b[:, [3, 1]]
    if k & 4:
        img = img.transpose(1, 0, 2)
        b = b[:, [1, 0, 3, 2]]
    return np.ascontiguousarray(img), b


class CropSampler:
    def __init__(self, frames, crop, aug, rng, pos_prob=0.6):
        self.frames, self.crop, self.aug, self.rng, self.pos_prob = frames, crop, aug, rng, pos_prob

    def _cut(self, boxes, x0, y0):
        c = self.crop
        kept, partial = [], False
        for x1, y1, x2, y2 in boxes:
            ix1, iy1, ix2, iy2 = max(x1, x0), max(y1, y0), min(x2, x0 + c), min(y2, y0 + c)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            vis = inter / ((x2 - x1) * (y2 - y1))
            if vis >= 0.8:
                kept.append((ix1 - x0, iy1 - y0, ix2 - x0, iy2 - y0))
            elif vis > 0:
                partial = True
        return np.array(kept, dtype=np.float32).reshape(-1, 4), partial

    def sample(self):
        r, c = self.rng, self.crop
        f = self.frames[r.integers(len(self.frames))]
        img, boxes = f["img"], f["boxes"]
        H, W = img.shape[:2]
        for _ in range(10):
            if len(boxes) and r.random() < self.pos_prob:
                b = boxes[r.integers(len(boxes))]
                x0 = int((b[0] + b[2]) / 2 - r.uniform(0.15, 0.85) * c)
                y0 = int((b[1] + b[3]) / 2 - r.uniform(0.15, 0.85) * c)
            else:
                x0, y0 = int(r.integers(0, W - c + 1)), int(r.integers(0, H - c + 1))
            x0, y0 = min(max(x0, 0), W - c), min(max(y0, 0), H - c)
            kept, partial = self._cut(boxes, x0, y0)
            if not partial:
                break
        crop = img[y0:y0 + c, x0:x0 + c]
        crop, kept = dihedral(crop, kept, int(r.integers(8)))
        if self.aug == "hed":
            crop = hed_jitter(crop, r)
        return crop, kept


# ----------------------------------------------------------------------------- model
class ChannelAtt(nn.Module):
    def __init__(self, ch, use_max, r=16):
        super().__init__()
        self.use_max = use_max
        self.mlp = nn.Sequential(nn.Linear(ch, ch // r), nn.ReLU(), nn.Linear(ch // r, ch))

    def forward(self, x):
        w = self.mlp(x.mean((2, 3)))
        if self.use_max:
            w = w + self.mlp(x.amax((2, 3)))
        return x * torch.sigmoid(w)[:, :, None, None]


class SpatialAtt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(s))


class Gate(nn.Module):
    """x + gamma * att(x), gamma starts at 0 so the pretrained features are untouched at the start.
    se   = channel attention (avg)            dual = channel(avg) + spatial  (manuscript module)
    cbam = channel(avg+max) + spatial (published CBAM)"""

    def __init__(self, kind, ch):
        super().__init__()
        self.ch = ChannelAtt(ch, use_max=(kind == "cbam"))
        self.sp = SpatialAtt() if kind in ("dual", "cbam") else None
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = self.ch(x)
        if self.sp is not None:
            y = self.sp(y)
        return x + self.gamma * y


class AttnBackbone(nn.Module):
    def __init__(self, backbone, kind, levels=("0", "1")):  # P2, P3 = where ~34 px objects are detected
        super().__init__()
        self.backbone, self.out_channels = backbone, backbone.out_channels
        self.gates = nn.ModuleDict({l: Gate(kind, backbone.out_channels) for l in levels})

    def forward(self, x):
        f = self.backbone(x)
        for l, g in self.gates.items():
            f[l] = g(f[l])
        return f


def build_model(attn, crop, pretrained):
    kw = dict(min_size=crop, max_size=crop, box_score_thresh=0.01, box_detections_per_img=100)
    if pretrained:
        m = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
                                    trainable_backbone_layers=3, **kw)
    else:
        m = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, **kw)
    # anchors matched to ~34 px mitoses (median box 34 px, max 100); one size/level keeps head shapes
    m.rpn.anchor_generator = AnchorGenerator(sizes=((16,), (32,), (48,), (64,), (96,)),
                                             aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    if attn != "none":
        m.backbone = AttnBackbone(m.backbone, attn)
    return m


# ----------------------------------------------------------------------------- train / infer
def to_tensor(img, device):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255).to(device)


def train(model, frames, args, device, rng):
    use_cuda = device.type == "cuda"
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    warm = min(100, args.iters // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, args.iters - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    sampler = CropSampler(frames, args.crop, args.aug, rng)
    run_loss = 0.0
    for it in range(args.iters):
        imgs, tgts = [], []
        for _ in range(args.batch):
            im, b = sampler.sample()
            imgs.append(to_tensor(im, device))
            tgts.append({"boxes": torch.from_numpy(b).to(device).reshape(-1, 4),
                         "labels": torch.ones(len(b), dtype=torch.int64, device=device)})
        with torch.autocast("cuda", enabled=use_cuda):
            loss = sum(model(imgs, tgts).values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iter {it}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run_loss += loss.item()
        if (it + 1) % 200 == 0:
            print(f"    iter {it + 1}/{args.iters} loss {run_loss / 200:.3f}", flush=True)
            run_loss = 0.0


def axis(length, tile, stride):
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


@torch.no_grad()
def detect(model, img, device, tile, stride, bs=8):
    model.eval()
    H, W = img.shape[:2]
    pos = [(x, y) for y in axis(H, tile, stride) for x in axis(W, tile, stride)]
    boxes, scores = [], []
    for i in range(0, len(pos), bs):
        chunk = pos[i:i + bs]
        batch = [to_tensor(img[y:y + tile, x:x + tile], device) for x, y in chunk]
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            outs = model(batch)
        for (x, y), o in zip(chunk, outs):
            b = o["boxes"].float().cpu()
            b[:, [0, 2]] += x
            b[:, [1, 3]] += y
            boxes.append(b)
            scores.append(o["scores"].float().cpu())
    b, s = torch.cat(boxes), torch.cat(scores)
    keep = nms(b, s, 0.5)[:300]
    return b[keep].numpy(), s[keep].numpy()


# ----------------------------------------------------------------------------- metrics
def match(pb, ps, gb, mode, dist):
    """Greedy matching in descending score order -> consistent with any score threshold."""
    order = np.argsort(-ps)
    pb, ps = pb[order], ps[order]
    flags, gscore = np.zeros(len(pb), bool), np.full(len(gb), -1.0)
    if len(pb) and len(gb):
        if mode == "iou":
            sim = box_iou(torch.from_numpy(pb), torch.from_numpy(gb)).numpy()
            ok = sim >= 0.5
        else:
            pc, gc = (pb[:, :2] + pb[:, 2:]) / 2, (gb[:, :2] + gb[:, 2:]) / 2
            d = np.linalg.norm(pc[:, None] - gc[None], axis=2)
            sim, ok = -d, d <= dist
        taken = np.zeros(len(gb), bool)
        for i in range(len(pb)):
            cand = np.where(ok[i] & ~taken)[0]
            if len(cand):
                j = cand[np.argmax(sim[i][cand])]
                taken[j], flags[i], gscore[j] = True, True, ps[i]
    return ps, flags, gscore


def make_rec(det, gt, mode, dist):
    ps, flags, gscore = match(det[0], det[1], gt, mode, dist)
    garea = np.sqrt((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])) if len(gt) else np.zeros(0)
    return dict(scores=ps, flags=flags, gscore=gscore, ngt=len(gt), garea=garea)


def counts(rec, t):
    sel = rec["scores"] >= t
    k, tp = int(sel.sum()), int(rec["flags"][sel].sum())
    return tp, k - tp, rec["ngt"] - tp


def pooled_sorted(recs):
    s = np.concatenate([r["scores"] for r in recs]) if recs else np.zeros(0)
    f = np.concatenate([r["flags"] for r in recs]) if recs else np.zeros(0, bool)
    o = np.argsort(-s)
    return s[o], f[o], sum(r["ngt"] for r in recs)


def best_threshold(recs):
    s, f, n = pooled_sorted(recs)
    if len(s) == 0 or n == 0:
        return 0.5
    tp = np.cumsum(f)
    k = np.arange(1, len(s) + 1)
    f1 = 2 * tp / (k + n)  # 2TP / (2TP + FP + FN) with FP = k-TP, FN = n-TP
    return float(s[int(np.argmax(f1))])


def average_precision(recs):
    s, f, n = pooled_sorted(recs)
    if len(s) == 0 or n == 0:
        return 0.0
    tp = np.cumsum(f)
    prec, rec = tp / np.arange(1, len(s) + 1), tp / n
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[0], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def summarize(recs, names, thr, edges):
    tot = np.array([counts(r, thr) for r in recs]).sum(0) if recs else np.zeros(3, int)
    tp, fp, fn = (int(v) for v in tot)
    n, hit = np.zeros(4, int), np.zeros(4, int)
    for r in recs:
        b = np.digitize(r["garea"], edges)
        for k in range(4):
            m = b == k
            n[k] += int(m.sum())
            hit[k] += int((r["gscore"][m] >= thr).sum())
    return dict(ap=average_precision(recs), tp=tp, fp=fp, fn=fn,
                p=tp / max(tp + fp, 1), r=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                bins=dict(n=n.tolist(), hit=hit.tolist()),
                frames={nm: list(counts(r, thr)) for nm, r in zip(names, recs)})


def evaluate(model, val, test, args, device, edges):
    det = lambda f: detect(model, get_img(f), device, args.crop, args.stride)
    vdet = {f["name"]: det(f) for f in val}
    thr = {m: best_threshold([make_rec(vdet[f["name"]], f["boxes"], m, args.dist) for f in val]) for m in MODES}
    out, tdets = {}, {}
    for sc in SCANNERS:
        fs = [f for f in test if f["scanner"] == sc]
        dets = {f["name"]: det(f) for f in fs}
        tdets.update(dets)
        out[sc] = {m: summarize([make_rec(dets[f["name"]], f["boxes"], m, args.dist) for f in fs],
                                [f["name"] for f in fs], thr[m], edges) for m in MODES}
    return thr, out, tdets


# ----------------------------------------------------------------------------- run
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    frames = load_frames(args.root)
    slides = list(range(NFOLDS))
    edges = np.full(3, 1e9)  # all boxes are 50x50 px: size bins are meaningless, everything lands in bin 0
    print(f"{len(frames)} images, scanners {SCANNERS}, mitoses per scanner "
          f"{ {sc: int(sum(len(f['boxes']) for f in frames if f['scanner'] == sc)) for sc in SCANNERS} }")
    res_path = out / "results.jsonl"
    done = {json.loads(l)["key"] for l in res_path.read_text().splitlines()} if res_path.exists() else set()
    folds = [-1] if args.leaky else ([int(x) for x in args.folds.split(",")] if args.folds else slides)
    for seed in [int(x) for x in args.seeds.split(",")]:
        for fold in folds:
            val_slide = -1 if args.leaky else slides[(slides.index(fold) + 1) % len(slides)]
            for train_set in args.train.split(","):
                for attn in args.attn.split(","):
                    key = f"{attn}|{train_set}|f{fold}|s{seed}" + ("|hed" if args.aug == "hed" else "") + ("|leaky" if args.leaky else "")
                    if key in done:
                        print("skip", key)
                        continue
                    scs = tuple(SCANNERS) if train_set == "Mixed" else (train_set,)
                    if train_set != "Mixed" and train_set not in SCANNERS:
                        raise SystemExit(f"unknown --train {train_set}; choose from {SCANNERS} or Mixed")
                    tr = [f for f in frames if f["scanner"] in scs and f["slide"] not in (fold, val_slide)]
                    va = [f for f in frames if f["scanner"] in scs and f["slide"] == val_slide]
                    te = [f for f in frames if f["slide"] == fold]
                    if args.leaky:  # CONTROL ONLY: naive protocol, test frames are training frames
                        va, te = list(tr), list(frames)
                    print(f"== {key}: train {len(tr)} frames, val {len(va)}, test {len(te)}", flush=True)
                    t0 = time.time()
                    for f in tr:
                        f["img"] = get_img(f)
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    model = build_model(attn, args.crop, not args.no_pretrained)
                    train(model, tr, args, device, np.random.default_rng(seed))
                    thr, res, tdets = evaluate(model, va, te, args, device, edges)
                    for f in tr:
                        f.pop("img", None)
                    rec = dict(key=key, attn=attn, train=train_set, fold=fold, seed=seed, val_slide=val_slide,
                               iters=args.iters, aug=args.aug, protocol="leaky" if args.leaky else "slide", secs=round(time.time() - t0), thr=thr, test=res)
                    with open(res_path, "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    with open(out / "preds" / (key.replace("|", "_") + ".pkl"), "wb") as fh:
                        pickle.dump(dict(thr=thr, dets=tdets), fh)
                    for sc in SCANNERS:
                        m = res[sc]["iou"]
                        print(f"   test {sc}: AP50 {m['ap']:.3f}  F1 {m['f1']:.3f} (P {m['p']:.3f} R {m['r']:.3f})"
                              f"  [{rec['secs']}s]", flush=True)
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- aggregate
def f1_of(c):
    tp, fp, fn = c
    return 2 * tp / max(2 * tp + fp + fn, 1e-9)


def lab(r):
    return r["attn"] + ("+hed" if r.get("aug", "flip") == "hed" else "") + ("+leaky" if r.get("protocol") == "leaky" else "")


def aggregate(args):
    rows = [json.loads(l) for l in (Path(args.out) / "results.jsonl").read_text().splitlines()]
    SCANNERS[:] = sorted(rows[0]["test"])
    rng = np.random.default_rng(0)
    for mode in MODES:
        print(f"\n===== matching: {mode} ({'IoU>=0.5' if mode == 'iou' else f'centroid within {args.dist}px'}) =====")
        groups = {}
        for r in rows:
            for sc in SCANNERS:
                groups.setdefault((lab(r), r["train"], sc), []).append((r["seed"], r["test"][sc][mode]))
        frame_counts = {}
        print(f"{'attn':6} {'train->test':10} {'runs':>4} {'pooled F1 (seed mean+-sd)':>27} {'AP50':>6}  recall")
        for (attn, tr, sc), items in sorted(groups.items()):
            by_seed = {}
            for seed, m in items:
                d = by_seed.setdefault(seed, dict(c=np.zeros(3), ap=[], n=np.zeros(4), hit=np.zeros(4)))
                d["c"] += [m["tp"], m["fp"], m["fn"]]
                d["ap"].append(m["ap"])
                d["n"] += m["bins"]["n"]
                d["hit"] += m["bins"]["hit"]
            f1s = [f1_of(d["c"]) for d in by_seed.values()]
            ap = np.mean([np.mean(d["ap"]) for d in by_seed.values()])
            n, hit = sum(d["n"] for d in by_seed.values()), sum(d["hit"] for d in by_seed.values())
            tpf = sum(d["c"][0] for d in by_seed.values()); fnf = sum(d["c"][2] for d in by_seed.values())
            rb = f"{tpf / max(tpf + fnf, 1):.2f}"
            print(f"{attn:9} {tr + '->' + sc:10} {len(items):>4} {np.mean(f1s):>13.3f} +- {np.std(f1s):.3f} {ap:>10.3f}  {rb}")
            fr = {}
            for _, m in items:
                for nm, c in m["frames"].items():
                    fr.setdefault(nm, []).append(c)
            frame_counts[(attn, tr, sc)] = {nm: np.mean(v, 0) for nm, v in fr.items()}

        def boot(a, b=None, B=2000):
            names = sorted(a)
            A = np.array([a[n] for n in names])
            Bm = np.array([b[n] for n in names]) if b is not None else None
            vals = []
            for _ in range(B):
                i = rng.integers(len(names), size=len(names))
                v = f1_of(A[i].sum(0))
                if Bm is not None:
                    v -= f1_of(Bm[i].sum(0))
                vals.append(v)
            return np.percentile(vals, [2.5, 97.5])

        print("\nimage-bootstrap 95% CI of pooled F1 (seed-averaged counts; each image is a separate case)")
        for k, a in sorted(frame_counts.items()):
            lo, hi = boot(a)
            print(f"  {k[0]:9} {k[1]}->{k[2]}: {f1_of(sum(a.values())):.3f}  [{lo:.3f}, {hi:.3f}]")
        print("\npaired difference vs attn=none (same frames), 95% CI")
        for (attn, tr, sc), a in sorted(frame_counts.items()):
            base = frame_counts.get(("none", tr, sc))
            if attn == "none" or base is None or set(base) != set(a):
                continue
            lo, hi = boot(a, base)
            d = f1_of(sum(a.values())) - f1_of(sum(base.values()))
            print(f"  {attn:9} {tr}->{sc}: dF1 {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--train", default="XR,S360,CS2,Mixed")
    r.add_argument("--attn", default="none,dual", help="none,se,dual,cbam")
    r.add_argument("--folds", default="", help="fold numbers 0-4, e.g. 0,1,2 (default all)")
    r.add_argument("--seeds", default="42")
    r.add_argument("--iters", type=int, default=2000)
    r.add_argument("--batch", type=int, default=4)
    r.add_argument("--lr", type=float, default=0.01)
    r.add_argument("--crop", type=int, default=512)
    r.add_argument("--stride", type=int, default=448, help="tile stride at inference (512 tile, 64 px overlap > 50 px box)")
    r.add_argument("--aug", default="flip", choices=["flip", "hed"])
    r.add_argument("--dist", type=float, default=25.0, help="centroid radius (px) for the relaxed criterion (half box)")
    r.add_argument("--leaky", action="store_true", help="CONTROL: train on all slides, test on the same frames (naive protocol). Never report as a result.")
    r.add_argument("--no-pretrained", action="store_true", help="smoke tests only")
    a = sub.add_parser("aggregate")
    a.add_argument("--out", required=True)
    a.add_argument("--dist", type=float, default=25.0)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else aggregate(args)


if __name__ == "__main__":
    main()
