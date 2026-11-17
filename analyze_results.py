import json, sys, numpy as np, collections
rows=[json.loads(l) for l in open(sys.argv[1])]
lab=lambda r:r["attn"]+("+hed" if r.get("aug","flip")=="hed" else "")+("+leaky" if r.get("protocol")=="leaky" else "")
MODE="iou"
def f1(c): tp,fp,fn=c; return 2*tp/max(2*tp+fp+fn,1e-9)
# per (attn,train,test,seed): frame -> counts
D=collections.defaultdict(dict)
for r in rows:
    for sc in "AH":
        for nm,c in r["test"][sc][MODE]["frames"].items():
            D[(lab(r),r["train"],sc,r["seed"])][nm]=np.array(c,float)
seeds=sorted({r["seed"] for r in rows})
LABELS=sorted({lab(r) for r in rows})
seeds_of=lambda a:sorted({r["seed"] for r in rows if lab(r)==a})
def framecounts(attn,train,sc):           # seed-averaged
    out={}
    for nm in D[(attn,train,sc,seeds_of(attn)[0])]:
        out[nm]=np.mean([D[(attn,train,sc,s)][nm] for s in seeds_of(attn)],0)
    return out
slide=lambda nm:int(nm[1:3])
slides=sorted({slide(n) for n in framecounts("none","A","A")})
FC={(a,t,s):framecounts(a,t,s) for a in LABELS for t in("A","H","Mixed") for s in "AH"}
def F(attn,train,sc,sl):                  # pooled F1 over chosen slides (list w/ repeats)
    c=np.zeros(3)
    for k in sl:
        for nm,v in FC[(attn,train,sc)].items():
            if slide(nm)==k: c+=v
    return f1(c)
rng=np.random.default_rng(0)
def boot(fn,B=4000):
    pt=fn(slides); vs=[fn(list(rng.choice(slides,len(slides)))) for _ in range(B)]
    return pt,*np.percentile(vs,[2.5,97.5])
def show(name,fn): p,l,h=boot(fn); print(f"  {name:58s} {p:+.3f}  [{l:+.3f}, {h:+.3f}]")

print("OLD (leaky) vs NEW (slide-level), dual attention, F1")
old={("A","A"):.970,("A","H"):.797,("H","H"):.940,("H","A"):.749,("Mixed","A"):.964,("Mixed","H"):.966}
for (t,s),o in old.items():
    n=F("dual",t,s,slides); print(f"  {t}->{s}: old {o:.3f}  new {n:.3f}  inflation {o-n:+.3f}")

for att in ("none","dual"):
    print(f"\nSCANNER-SHIFT, attn={att}: (in-domain F1) minus (cross-scanner F1), same test scanner; >0 means a real shift penalty")
    show("test A: A->A minus H->A", lambda sl:F(att,"A","A",sl)-F(att,"H","A",sl))
    show("test H: H->H minus A->H", lambda sl:F(att,"H","H",sl)-F(att,"A","H",sl))
    show("average of the two", lambda sl:.5*(F(att,"A","A",sl)-F(att,"H","A",sl)+F(att,"H","H",sl)-F(att,"A","H",sl)))
    print(f"MIXED vs in-domain, attn={att}")
    show("Mixed->A minus A->A", lambda sl:F(att,"Mixed","A",sl)-F(att,"A","A",sl))
    show("Mixed->H minus H->H", lambda sl:F(att,"Mixed","H",sl)-F(att,"H","H",sl))

print("\nATTENTION effect (variant minus none), slide-bootstrap, averaged over all 6 train->test cells")
cells=[(t,s) for t in("A","H","Mixed") for s in "AH"]
for att in ("dual","cbam"):
    show(f"{att} - none (mean of 6 cells)", lambda sl:np.mean([F(att,t,s,sl)-F("none",t,s,sl) for t,s in cells]))
    show(f"{att} - none (Mixed cells only)", lambda sl:np.mean([F(att,"Mixed",s,sl)-F("none","Mixed",s,sl) for s in "AH"]))
print("\nper-seed sign consistency of (variant - none), pooled F1 per seed, 6 cells x 3 seeds = 18 comparisons")
def seedF1(att,t,s,sd):
    return f1(sum(D[(att,t,s,sd)].values()))
for att in ("dual","cbam"):
    d=[seedF1(att,t,s,sd)-seedF1("none",t,s,sd) for t,s in cells for sd in seeds]
    print(f"  {att}: negative in {sum(x<0 for x in d)}/18, mean {np.mean(d):+.3f}, min {min(d):+.3f}, max {max(d):+.3f}")
print("\nPER-FOLD (slide) F1, none, A-trained->A: ", [round(F('none','A','A',[k]),2) for k in slides], " mitoses/slide:",[21,56,25,82,42])
print("PER-FOLD F1, none, Mixed->H:", [round(F('none','Mixed','H',[k]),2) for k in slides])

if "none+hed" in LABELS:
    print("\nSTAIN AUGMENTATION (none+hed minus none), slide-bootstrap")
    for t,sc in cells:
        show(f"{t}->{sc}", lambda sl,t=t,sc=sc:F("none+hed",t,sc,sl)-F("none",t,sc,sl))
    show("mean of 6 cells", lambda sl:np.mean([F("none+hed",t,s,sl)-F("none",t,s,sl) for t,s in cells]))
    show("cross-scanner cells only (A->H, H->A)", lambda sl:np.mean([F("none+hed","A","H",sl)-F("none","A","H",sl),F("none+hed","H","A",sl)-F("none","H","A",sl)]))
    show("scanner-shift gap with hed (avg in-domain minus cross)", lambda sl:.5*(F("none+hed","A","A",sl)-F("none+hed","H","A",sl)+F("none+hed","H","H",sl)-F("none+hed","A","H",sl)))
    for sd in seeds_of("none+hed"):
        d=[seedF1("none+hed",t,s,sd)-seedF1("none",t,s,sd) for t,s in cells]
        print(f"  seed {sd}: mean dF1 {np.mean(d):+.3f}, negative in {sum(x<0 for x in d)}/6")

if "none+leaky" in LABELS:
    print("\nLEAKAGE CONTROL (same code, same detector): naive protocol minus slide-level protocol, attn=none")
    for t,sc in cells:
        print(f"  {t}->{sc}: naive {F('none+leaky',t,sc,slides):.3f}  slide-level {F('none',t,sc,slides):.3f}")
    for t,sc in cells:
        show(f"{t}->{sc}  (naive - slide-level)", lambda sl,t=t,sc=sc:F("none+leaky",t,sc,sl)-F("none",t,sc,sl))
    show("mean of 6 cells", lambda sl:np.mean([F("none+leaky",t,s,sl)-F("none",t,s,sl) for t,s in cells]))
    show("apparent scanner-shift gap under NAIVE protocol (in-domain minus cross)", lambda sl:.5*(F("none+leaky","A","A",sl)-F("none+leaky","H","A",sl)+F("none+leaky","H","H",sl)-F("none+leaky","A","H",sl)))
