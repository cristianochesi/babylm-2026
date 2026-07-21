#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_gold_coverage.py
======================
Diagnose *why* ooa_trajectory_eval.py finds no morphemes, by comparing the GOLD
exponent inventory against the pieces MorPiece actually produced.

It answers three questions for a real vocab snapshot:
  1. Which GOLD exponents are present / absent (exact-key match)?
  2. What distinctive ++ suffix pieces did MorPiece actually emit (so GOLD can
     be remapped to reality)?
  3. Does any piece (++ or root) END in an inflection signature (-iamo, -ato,
     -ava, ...), i.e. is the morpheme present but cut at a different boundary?

Usage:
  python3 audit_gold_coverage.py --lang ita \
      --vocab .../ooa_incremental_vocabs/vocab_9800000.json
  # or point at the directory; it audits the highest-exposure snapshot:
  python3 audit_gold_coverage.py --lang ita --ooa_dir .../ooa_incremental_vocabs
"""
import argparse
import glob as globmod
import json
import os
import re
import sys

# Reuse the exact GOLD definitions the evaluator uses, so the audit stays in sync.
from ooa_trajectory_eval import GOLD_ITA, GOLD_ENG

GOLDS = {"ita": GOLD_ITA, "eng": GOLD_ENG}


def _num(path):
    m = re.search(r"(\d+)(?=\.json$)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def load_vocab(path):
    obj = json.load(open(path, encoding="utf-8"))
    v = obj.get("vocab", obj) if isinstance(obj, dict) else {}
    out = {}
    for k, c in v.items():
        try:
            out[k] = float(c)
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ita", "eng"], required=True)
    ap.add_argument("--vocab", help="a single vocab_<N>.json")
    ap.add_argument("--ooa_dir", help="dir of vocab_*.json (audits highest-exposure)")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    path = args.vocab
    if not path and args.ooa_dir:
        cand = globmod.glob(os.path.join(args.ooa_dir, "vocab_*.json"))
        path = max(cand, key=_num) if cand else None
    if not path:
        sys.exit("provide --vocab <file> or --ooa_dir <dir>")

    vocab = load_vocab(path)
    print(f"vocab: {os.path.basename(path)}   ({len(vocab)} pieces)\n")

    gold = GOLDS[args.lang]

    # 1. GOLD coverage ------------------------------------------------------
    print("=== GOLD coverage (exact-key match) ===")
    n_present = 0
    for g in gold:
        present = [(e, int(vocab[e])) for e in g["exponents"] if e in vocab]
        absent = [e for e in g["exponents"] if e not in vocab]
        tag = "ambig" if g["ambiguous"] else ("free " if not g["bound"] else "BOUND")
        if present:
            n_present += 1
            print(f"  [{tag}] {g['label']:30s} tier={g['gold']}  PRESENT {present}")
        else:
            print(f"  [{tag}] {g['label']:30s} tier={g['gold']}  --absent-- {absent}")
    print(f"\n  {n_present}/{len(gold)} gold morphemes have >=1 present exponent.")
    strict_present = sum(
        1 for g in gold
        if not g["ambiguous"] and g["bound"]
        and any(e in vocab for e in g["exponents"])
    )
    print(f"  {strict_present} of those are STRICT (bound, unambiguous) — the "
          f"ones the trajectory uses.\n")

    # 2. Real distinctive ++ inventory -------------------------------------
    suff = [(k, int(c)) for k, c in vocab.items()
            if k.startswith("++") and len(k) - 2 >= 3]      # >=3 chars after ++
    suff.sort(key=lambda x: (-(len(x[0]) - 2), -x[1]))
    print(f"=== distinctive ++ pieces MorPiece actually produced "
          f"(>=3 chars, top {args.topk}) ===")
    if not suff:
        print("  (none) — at this config MorPiece emits only short ++ pieces.")
    for k, c in suff[:args.topk]:
        print(f"  {k:14s} {c}")
    print()

    # 3. Inflection-signature endswith search ------------------------------
    sigs = (["iamo", "ate", "ete", "ano", "ono", "ato", "ata", "ito", "uto",
             "ava", "avano", "eva", "are", "ere", "ire", "ero", "era", "rebbe"]
            if args.lang == "ita"
            else ["ing", "ed", "ness", "tion", "ly", "est", " er"])
    print("=== any piece ending in an inflection signature (boundary may differ) ===")
    for sig in sigs:
        hits = sorted(
            [(k, int(c)) for k, c in vocab.items()
             if k.replace("++", "").endswith(sig)],
            key=lambda x: -x[1])[:6]
        if hits:
            print(f"  -{sig:6s}: " + ", ".join(f"{k}({c})" for k, c in hits))
    print("\nIf a morpheme is absent as an exact ++ piece but shows up here ending "
          "a longer piece,\nMorPiece is cutting it at a different boundary — remap "
          "GOLD_ITA exponents to the\nforms that actually appear above, then re-run "
          "ooa_trajectory_eval.py.")


if __name__ == "__main__":
    main()
