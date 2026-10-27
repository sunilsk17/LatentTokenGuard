"""
experiments/make_label_audit.py
Build a blind review page for checking CHAIR labels by eye.

Samples first mentions from the published runs (equal numbers labelled real and hallucinated
per model), copies the images (resized), and writes a self-contained index.html. The reviewer
sees the image and the caption sentence with the object highlighted, but not the CHAIR label,
and answers whether the object is in the image. Answers are kept in the browser and downloaded
as label_audit_answers.json; experiments/score_label_audit.py compares them with the labels.

Usage:
    python experiments/make_label_audit.py --out <folder> --per_class 34
"""

import argparse
import html
import json
import os
import random
import re

from PIL import Image

RUNS = {"LLaVA-1.5-7B": "results/caption_runs/llava15_7b_4bit_n1000_s42",
        "LLaVA-1.5-13B": "results/caption_runs/llava15_13b_n1000_s42",
        "Qwen2.5-VL-7B": "results/caption_runs/qwen25vl_7b_n1000_s42"}


def sentence_around(caption: str, start: int, end: int) -> str:
    """The sentence containing [start, end) with the object wrapped in <mark>."""
    left = max(caption.rfind(c, 0, start) for c in ".!?\n") + 1
    rights = [i for i in (caption.find(c, end) for c in ".!?\n") if i != -1]
    right = min(rights) + 1 if rights else len(caption)
    s = caption[left:start], caption[start:end], caption[end:right]
    return (html.escape(s[0].lstrip()) + "<mark>" + html.escape(s[1]) + "</mark>" + html.escape(s[2])).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--per_class", type=int, default=34, help="items per model and label")
    p.add_argument("--images_dir", default="data/coco/val2017")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(os.path.join(args.out, "img"), exist_ok=True)

    items = []
    for model, run in RUNS.items():
        pool = {0: [], 1: []}
        for line in open(os.path.join(run, "captions.jsonl")):
            r = json.loads(line)
            for m in r["mentions"]:
                if m["first"]:
                    pool[m["label"]].append((r, m))
        for label in (0, 1):
            for r, m in rng.sample(pool[label], args.per_class):
                items.append({"model": model, "image_id": r["image_id"], "object": m["canonical"],
                              "surface": m["surface"], "chair_label": label,
                              "context": sentence_around(r["caption"], m["start"], m["end"])})
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["id"] = i
        name = f"{it['image_id']:012d}.jpg"
        dst = os.path.join(args.out, "img", name)
        if not os.path.exists(dst):
            im = Image.open(os.path.join(args.images_dir, name)).convert("RGB")
            im.thumbnail((900, 900))
            im.save(dst, quality=85)
        it["img"] = f"img/{name}"

    with open(os.path.join(args.out, "key.json"), "w") as f:   # labels, kept out of the page
        json.dump(items, f, indent=1)
    public = [{k: it[k] for k in ("id", "img", "object", "surface", "context")} for it in items]
    page = PAGE.replace("__ITEMS__", json.dumps(public))
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write(page)
    print(f"{len(items)} items -> {args.out}/index.html")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label audit</title>
<style>
:root{--bg:#fbfbfa;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--accent:#2a78d6;--yes:#1a7f37;--no:#c2410c;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--ink:#efeee9;--muted:#a3a29a;--line:#2f2e2b;--card:#1f1f1d;--accent:#5598e7}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
header{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-bottom:12px}
h1{font-size:18px;margin:0}.muted{color:var(--muted);font-size:14px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;flex:1;min-width:160px}.bar>div{height:100%;background:var(--accent)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
img{max-width:100%;max-height:60vh;display:block;margin:0 auto 12px;border-radius:6px}
.q{font-size:18px;margin:4px 0 6px}.ctx{color:var(--muted);margin-bottom:14px}
mark{background:#fde68a;color:#1d1d1b;padding:0 2px;border-radius:3px}
.btns{display:flex;gap:8px;flex-wrap:wrap}
button{font:inherit;padding:10px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.sel{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}
.nav{display:flex;justify-content:space-between;margin-top:14px;gap:8px;flex-wrap:wrap}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px}
</style></head><body><main>
<header><h1>Is the highlighted object in the image?</h1><span class="muted" id="count"></span><div class="bar"><div id="prog"></div></div></header>
<p class="muted">Answer from the image only. "Not an object" is for words used in another sense (e.g. "orange" the colour, a "tie" in a game score). Keys: <kbd>1</kbd> yes · <kbd>2</kbd> no · <kbd>3</kbd> unsure · <kbd>4</kbd> not an object · <kbd>←</kbd> <kbd>→</kbd> move.</p>
<div class="card"><img id="img" alt=""><div class="q" id="q"></div><div class="ctx" id="ctx"></div>
<div class="btns" id="btns"></div>
<div class="nav"><button id="prev">← Previous</button><button id="next_open">Next unanswered</button><button id="dl">Download answers</button><button id="next">Next →</button></div></div>
</main><script>
const ITEMS=__ITEMS__;const KEY="label-audit-v1";const OPTS=[["yes","Yes, it is there"],["no","No, it is not there"],["unsure","Unsure"],["not_object","Not an object"]];
let ans={};try{ans=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0;const $=id=>document.getElementById(id);
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}}
function show(){const it=ITEMS[i];$("img").src=it.img;$("img").alt="COCO image "+it.img;
$("q").innerHTML="Is there a <b>"+it.object+"</b> in the image?"+(it.surface.toLowerCase()!==it.object?" <span class='muted'>(written as “"+it.surface+"”)</span>":"");
$("ctx").innerHTML="“"+it.context+"”";$("btns").innerHTML="";
OPTS.forEach(([v,l],k)=>{const b=document.createElement("button");b.textContent=(k+1)+". "+l;if(ans[it.id]===v)b.className="sel";b.onclick=()=>pick(v);$("btns").appendChild(b)});
const n=Object.keys(ans).length;$("count").textContent=(i+1)+" / "+ITEMS.length+" · "+n+" answered";$("prog").style.width=(100*n/ITEMS.length)+"%"}
function pick(v){ans[ITEMS[i].id]=v;save();if(i<ITEMS.length-1)i++;show()}
$("prev").onclick=()=>{if(i>0)i--;show()};$("next").onclick=()=>{if(i<ITEMS.length-1)i++;show()};
$("next_open").onclick=()=>{const j=ITEMS.findIndex(it=>!(it.id in ans));if(j>=0){i=j;show()}};
$("dl").onclick=()=>{const blob=new Blob([JSON.stringify(ans,null,1)],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="label_audit_answers.json";a.click()};
document.addEventListener("keydown",e=>{if(e.key>="1"&&e.key<="4")pick(OPTS[+e.key-1][0]);if(e.key==="ArrowLeft")$("prev").click();if(e.key==="ArrowRight")$("next").click()});
show();
</script></body></html>"""

if __name__ == "__main__":
    main()
