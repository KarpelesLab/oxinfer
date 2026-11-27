"""oxinfer's whole pipeline (its own OCR) against LiteParse's on each test image: the page blocks
(kind, level, text) and the tables cell by cell."""
import json, subprocess, sys, os
S = sys.argv[1]
ox = sys.argv[2]
extra = sys.argv[3:]
ref = os.path.join(S, "lt/ref")
tot_b = same_b = tot_c = same_c = 0
for name in sorted(os.listdir(ref)):
    d = os.path.join(ref, name)
    img = [f for f in os.listdir(os.path.join(S, "lt/img")) if f.rsplit(".", 1)[0] == name][0]
    out = subprocess.run([ox, "ocr", os.path.join(S, "lt/img", img), "--model", os.path.join(S, "hf"), "--tables", "--json", "-q"] + extra,
                         capture_output=True, text=True, check=True).stdout
    page = json.loads(out)
    lay = page["layout"]
    mine = [(b["kind"], b.get("level"), b["text"]) for b in lay["blocks"] if b["kind"] not in ("table_text", "figure_text")]
    mine = [(k, l, "" if k == "table" else t) for k, l, t in mine]
    lp = json.load(open(os.path.join(d, "blocks.json")))["pages"][0]["blocks"]
    want = []
    for b in lp:
        k = "table" if b["kind"] == "merged_table" else b["kind"]
        want.append((k, b.get("level"), "" if k == "table" else b.get("text", "")))
    import difflib
    sm = difflib.SequenceMatcher(a=want, b=mine, autojunk=False)
    nb = sum(bl.size for bl in sm.get_matching_blocks())
    tot_b += len(want); same_b += nb
    m = json.load(open(os.path.join(d, "model.json")))
    cells_w = [(t_i, c["r0"], c["c0"], c["r1"], c["c1"], c["text"]) for t_i, t in enumerate(m["tables"]) for c in t["cells"]]
    cells_m = [(t_i, c["row"], c["col"], c["row"] + c["row_span"] - 1, c["col"] + c["col_span"] - 1, c["text"]) for t_i, t in enumerate(lay["tables"]) for c in t["cells"]]
    nc = len(set(cells_w) & set(cells_m))
    tot_c += len(cells_w); same_c += nc
    print(f"{name}: blocks {nb}/{len(want)} (ours {len(mine)}), tables {len(lay['tables'])}/{len(m['tables'])}, cells {nc}/{len(cells_w)} (ours {len(cells_m)})")
    if nb != len(want) or nb != len(mine):
        for op in sm.get_opcodes():
            if op[0] != "equal":
                print("   ", op[0], want[op[1]:op[2]], "->", mine[op[3]:op[4]])
    if nc != len(cells_w) or nc != len(cells_m):
        print("    missing", sorted(set(cells_w) - set(cells_m))[:6], "extra", sorted(set(cells_m) - set(cells_w))[:6])
print(f"total: blocks {same_b}/{tot_b}, cells {same_c}/{tot_c}")
