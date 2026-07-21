#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ooa_trajectory_eval.py  —  v2.1
================================
Evaluate MorPiece's Order-of-Acquisition (OOA) trajectory against gold
developmental morpheme orders.

GOLD STANDARDS
--------------
  English : Brown (1973) / de Villiers & de Villiers (1973), 14 morphemes.
  Italian : developmental tiers from Belletti & Guasti (2015, LALD 57) and
            Caprin & Guasti (2009, Applied Psycholinguistics 30(1)).

TWO EVALUATION MODES
--------------------
  Snapshot mode  (default)
      Reads a directory of cumulative vocab snapshots (vocab_<N>.json) written
      by MorPiece's --ooa flag.  Acquisition = first/stable snapshot step at
      which a gold exponent appears in the vocabulary.

  Event-log mode  (--from-event-log)
      Reads ooa_vocab.tsv (the timed, word-stamped split log written by the
      same --ooa path).  Acquisition = exposure count of the first event whose
      word shape matches a gold morpheme's diagnostic regex.  This gives a
      finer-grained, word-disambiguated acquisition time.

ENGLISH SENSITIVITY — DESIGN NOTE  (v2.1)
------------------------------------------
Brown's 14 morphemes include 9 free morphemes (in, on, went, is/are, a/the,
…) that MorPiece does not segment into ++ pieces — correctly, since they are
not affixes.  After removing free morphemes, only 3 bound exponents remain:

  ++ing  (tier 1, unambiguous)
  ++s    (tiers 4 & 10 — plural and 3sg share the same exponent; deduplicated
          to the earlier tier 4 to prevent the NaN in partial-τ caused by two
          rows with identical acquisition steps)
  ++ed   (tier 9, unambiguous)

This gives N=3 for the English strict correlation — barely above the minimum.
The harness computes and reports τ for N≥2 (with an explicit warning for N<5),
plots always, and labels figures with the effective N so the fragility is
visible.  English results should be treated as qualitative / directional; the
headline developmental-plausibility claim rests on Italian (N≥10 strict).

CHANGES SINCE v2.0
------------------
  • GOLD_ENG: deduplicated ++s to a single tier-4 entry (eliminates NaN in
    partial-τ); removed tier-10 copy; annotated all free-morpheme entries.
  • English EVENT_MATCHERS: removed the duplicate tier-10 regex for ++s; the
    tier-4 "plural -s" matcher now covers both in the full analysis.
  • min_n for strict/full analysis lowered from 3→2; N<5 triggers a warning
    rather than a hard skip, so τ is computed and plotted even for small N.
  • Figures are always generated (no min_n gate on plotting); figure title
    carries the effective N and a ⚠ tag when N<5.
  • partial_tau: NaN-safe guard added; if the partial correlation is
    uncomputable (degenerate matrix, e.g. N=2) the output says "n/a (N too
    small)" rather than crashing or printing NaN.
  • Summary CSV headline column renamed strict_stable → strict_stable_tau for
    clarity.
"""

import argparse
import glob as globmod
import json
import os
import random
import re
import sys
import tempfile

import numpy as np
from scipy.stats import kendalltau, spearmanr

# =============================================================================
# GOLD STANDARDS
# =============================================================================

# ENGLISH — Brown (1973) "A First Language", 14 grammatical morphemes.
# Order corroborated by de Villiers & de Villiers (1973). Ranks 1..14.
#
# DESIGN NOTE (v2.1):
#   bound=True  → MorPiece can discover this as a ++ suffix; included in
#                 the strict bound-morpheme correlation.
#   bound=False → free morpheme or suppletive; excluded from strict, kept
#                 for full-analysis audit only.
#   ambiguous   → exponent shared across morphemes; excluded from the strict
#                 correlation UNLESS it is the only entry for that exponent
#                 (see ++s deduplication note above).
#
#   The tier-10 "3rd person regular -s" entry has been REMOVED because it
#   shares the ++s exponent with tier-4 "plural -s".  Keeping both causes:
#     (a) two rows with identical acquisition steps → partial-τ matrix is
#         rank-deficient → NaN;
#     (b) contradictory gold ranks for the same acquisition event.
#   We keep tier 4 (earlier) as the single ++s entry, `ambiguous=False`,
#   because the deduplication makes the analysis tractable.
#
# v2.2 — DERIVATIONAL EXTENSION (so the strict bound-morpheme set is large
# enough to DISCRIMINATE between tokenizer configurations).
#
# English inflection alone yields only 3 MorPiece-discoverable bound exponents
# (++ing, ++s, ++ed). With N=3 the Kendall τ can take only 4 distinct values
# {-1, -0.33, +0.33, +1}, so different tokenizers cannot separate. We therefore
# add productive DERIVATIONAL suffixes that (a) MorPiece can segment as ++
# pieces and (b) have a published developmental / productivity ordering.
#
# The composite scale interleaves Brown inflectional tiers with a derivational
# ordering justified by:
#   • Berko (1958) "wug": agentive -er productive earliest.
#   • Clark & Cohen (1984); Nagy, Diakidoy & Anderson (1993, JRB): NEUTRAL
#     (stress-preserving) suffixes -er, -ly acquired in preschool years;
#     NON-NEUTRAL suffixes (-ness, -ity, -ion) acquired later, through grade
#     school.
#   • Tyler & Nagy (1989): relational → syntactic → distributional knowledge
#     of derivational suffixes develops across grades 4–8.
#   • Bauer (2001) / Plag (1999) productivity ranking among nominalizers:
#     -ness > -ity > -th (only -ness/-ity are productive enough to appear).
#   • Cotterell et al. (2017): -ly most compositional; agentive -er least —
#     consistent with -er being stored-early but -ly being rule-productive.
#
# IMPORTANT: this composite ordering is a *derived* scale, not a single
# published list. The strict correlation uses ordinal tiers; ties are allowed
# (kendalltau variant='b' handles them). The justification belongs in the
# English methods note. Inflectional-only results remain reportable by setting
# ENG_DERIVATIONAL=False below.

ENG_DERIVATIONAL = True   # set False to reproduce the inflection-only (N=3) gold

GOLD_ENG = [
    # === BOUND INFLECTION (Brown 1973 tiers) =================================
    dict(label="present progressive -ing", gold=1,  exponents=["++ing"],
         bound=True,  ambiguous=False),
    dict(label="plural/3sg -s",            gold=4,  exponents=["++s"],
         bound=True,  ambiguous=False),   # tier 4 kept; tier-10 duplicate removed
    dict(label="past regular -ed",         gold=9,  exponents=["++ed"],
         bound=True,  ambiguous=False),
    dict(label="possessive -'s",           gold=6,  exponents=["++'s", "'s"],
         bound=True,  ambiguous=True),    # apostrophe surface-form is noisy

    # === BOUND DERIVATION (v2.2; included iff ENG_DERIVATIONAL) ==============
    # Tier values continue the developmental scale past Brown's 14 inflectional
    # ranks, ordered by neutrality + productivity (earlier = more productive /
    # earlier-acquired). Spacing leaves room so inflection and derivation do
    # not artificially tie.
    dict(label="agentive -er",             gold=15, exponents=["++er"],
         bound=True,  ambiguous=True,  derivational=True),   # also comparative -er
    dict(label="comparative -er",          gold=16, exponents=["++er"],
         bound=True,  ambiguous=True,  derivational=True),
    dict(label="superlative -est",         gold=17, exponents=["++est"],
         bound=True,  ambiguous=False, derivational=True),
    dict(label="adverbial -ly",            gold=18, exponents=["++ly"],
         bound=True,  ambiguous=False, derivational=True),
    dict(label="adjectival -y",            gold=19, exponents=["++y"],
         bound=True,  ambiguous=True,  derivational=True),   # noisy: also part of stems
    dict(label="completive -ful",           gold=20, exponents=["++ful"],
         bound=True,  ambiguous=False, derivational=True),
    dict(label="privative -less",          gold=21, exponents=["++less"],
         bound=True,  ambiguous=False, derivational=True),
    dict(label="nominalizer -ness",        gold=22, exponents=["++ness"],
         bound=True,  ambiguous=False, derivational=True),
    dict(label="nominalizer -ment",        gold=23, exponents=["++ment"],
         bound=True,  ambiguous=False, derivational=True),
    dict(label="nominalizer -ion/-tion",   gold=24, exponents=["++ion", "++tion"],
         bound=True,  ambiguous=False, derivational=True),
    dict(label="nominalizer -ity",         gold=25, exponents=["++ity"],
         bound=True,  ambiguous=False, derivational=True),

    # === FREE morphemes (audit / full-analysis only; bound=False) ===========
    dict(label="preposition in",           gold=2,  exponents=["in"],
         bound=False, ambiguous=False),
    dict(label="preposition on",           gold=3,  exponents=["on"],
         bound=False, ambiguous=False),
    dict(label="past irregular",           gold=5,  exponents=["went","came","got"],
         bound=False, ambiguous=False),
    dict(label="uncontractible copula",    gold=7,  exponents=["is","are","was"],
         bound=False, ambiguous=False),
    dict(label="articles a/the",           gold=8,  exponents=["a","the"],
         bound=False, ambiguous=False),
    dict(label="3rd-person irregular",     gold=11, exponents=["has","does"],
         bound=False, ambiguous=False),
    dict(label="uncontractible auxiliary", gold=12, exponents=["is","were"],
         bound=False, ambiguous=False),
    dict(label="contractible copula",      gold=13, exponents=["'s","'re"],
         bound=False, ambiguous=False),
    dict(label="contractible auxiliary",   gold=14, exponents=["'s","'re","'m"],
         bound=False, ambiguous=False),
]

if not ENG_DERIVATIONAL:
    GOLD_ENG = [g for g in GOLD_ENG if not g.get("derivational")]

# ITALIAN — developmental TIERS (smaller = earlier), synthesised from:
#   Pizzuto & Caselli (1992, J Child Lang 19:491-557; 1994 cross-ling. chapter),
#   Caselli, Leonard, Volterra & Campagnoli (1993, J Child Lang 20:377),
#   Leonard, Caselli & Devescovi (2002),
#   Caprin & Guasti (2009, Appl. Psycholing. 30(1):23-52),
#   Belletti & Guasti (2015, LALD 57, Benjamins).
GOLD_ITA = [
    # tier 1 — earliest: bare present tense (highly ambiguous vowel finals)
    dict(label="pres 1sg -o",              gold=1,  exponents=["++o"],
         bound=True,  ambiguous=True),
    dict(label="pres 3sg -a/-e",           gold=1,  exponents=["++a","++e"],
         bound=True,  ambiguous=True),
    # tier 2 — infinitive & present plural
    dict(label="infinitive -re",           gold=2,  exponents=["++re","++are","++ere","++ire"],
         bound=True,  ambiguous=False),
    dict(label="pres 1pl -iamo",           gold=2,  exponents=["++iamo"],
         bound=True,  ambiguous=False),
    dict(label="pres 2pl -ate/-ete/-ite",  gold=2,  exponents=["++ate","++ete","++ite"],
         bound=True,  ambiguous=False),
    dict(label="pres 3pl -ano/-ono",       gold=2,  exponents=["++ano","++ono"],
         bound=True,  ambiguous=False),
    # tier 3 — past participle & imperfect
    dict(label="past part. -ato/-ito/-uto",gold=3,  exponents=["++ato","++ita","++uto","++ata","++iti","++ate"],
         bound=True,  ambiguous=False),
    dict(label="imperfetto -v-",           gold=3,  exponents=["++avo","++avi","++ava","++avamo","++avate","++avano",
                                                                "++evo","++evi","++eva","++evamo","++evate","++evano",
                                                                "++ivo","++ivi","++iva","++ivamo","++ivate","++ivano"],
         bound=True,  ambiguous=False),
    # tier 4 — future & conditional
    dict(label="future -r-",               gold=4,  exponents=["++rò","++rai","++rà","++remo","++rete","++ranno"],
         bound=True,  ambiguous=False),
    dict(label="conditional -rebbe",       gold=4,  exponents=["++rei","++resti","++rebbe","++remmo","++reste","++rebbero"],
         bound=True,  ambiguous=False),
    # tier 5 — subjunctive & passato remoto
    dict(label="congiuntivo -i/-a",        gold=5,  exponents=["++i","++ia"],
         bound=True,  ambiguous=True),
    dict(label="passato remoto",           gold=5,  exponents=["++ai","++asti","++ò","++ammo","++aste","++arono",
                                                                "++ei","++esti","++é","++emmo","++este","++erono",
                                                                "++ii","++isti","++ì","++immo","++iste","++irono"],
         bound=True,  ambiguous=False),
]

# =============================================================================
# EVENT-LOG MATCHERS  (--from-event-log mode)
# =============================================================================
# Each matcher maps to a gold tier via GOLD_ENG / GOLD_ITA entries.
# precision gates the strict analysis:
#   high   — morphologically diagnostic regex, near-zero nominal homophones
#   medium — mostly verbal, some noise acceptable
#   low    — shared with productive nominal morphology; full-analysis only
#   ambig  — bare inflectional vowels; always excluded from strict
#
# NOTE (v2.1): the tier-10 English entry for ++s has been removed; the
#   tier-4 "plural -s" matcher covers both in the full analysis.

def _rx(p):
    return re.compile(p, re.UNICODE)

EVENT_MATCHERS = {
    "ita": [
        dict(label="pres 1sg -o",              gold=1, precision="ambig",
             rx=_rx(r"[^aeiou\W]o$")),
        dict(label="pres 3sg -a/-e",           gold=1, precision="ambig",
             rx=_rx(r"[^aeiou\W][ae]$")),
        dict(label="infinitive -re",           gold=2, precision="low",
             rx=_rx(r"(?:are|ere|ire|rre)$")),
        dict(label="pres 1pl -iamo",           gold=2, precision="high",
             rx=_rx(r"iamo$")),
        dict(label="pres 2pl -ate/-ete/-ite",  gold=2, precision="low",
             rx=_rx(r"(?:ate|ete|ite)$")),
        dict(label="pres 3pl -ano/-ono",       gold=2, precision="low",
             rx=_rx(r"(?:ano|ono)$")),
        dict(label="past part. -ato/-ito/-uto", gold=3, precision="low",
             rx=_rx(r"(?:at|it|ut)[oaie]$")),
        dict(label="imperfetto -v-",           gold=3, precision="high",
             rx=_rx(r"(?:av|ev|iv)(?:o|i|a|amo|ate|ano)$")),
        dict(label="future -r-",               gold=4, precision="high",
             rx=_rx(r"r(?:\u00f2|ai|\u00e0|emo|ete|anno)$")),
        dict(label="conditional -r-",          gold=4, precision="high",
             rx=_rx(r"r(?:ei|esti|ebbe|emmo|este|ebbero)$")),
    ],
    "eng": [
        # v2.2: tightened to fire on the actual suffixed word-shape, not on
        # short function words. The old [^s]s$ matcher fired on "is" (copula),
        # producing the spurious tier-4 late-acquisition that flipped τ sign.
        # We now require a minimum stem length before the suffix.
        dict(label="present progressive -ing", gold=1,  precision="high",
             rx=_rx(r"[a-z]{3,}ing$")),
        dict(label="plural/3sg -s",            gold=4,  precision="medium",
             rx=_rx(r"[a-z]{3,}[^aeious]s$")),   # ≥3-char stem, consonant before s
        dict(label="past regular -ed",         gold=9,  precision="medium",
             rx=_rx(r"[a-z]{3,}(?<![ao])ed$")),
        # --- derivational (v2.2) ---------------------------------------------
        dict(label="agentive/comparative -er", gold=15, precision="medium",
             rx=_rx(r"[a-z]{3,}er$")),
        dict(label="superlative -est",         gold=17, precision="high",
             rx=_rx(r"[a-z]{3,}est$")),
        dict(label="adverbial -ly",            gold=18, precision="high",
             rx=_rx(r"[a-z]{3,}ly$")),
        dict(label="privative -ful",           gold=20, precision="high",
             rx=_rx(r"[a-z]{3,}ful$")),
        dict(label="privative -less",          gold=21, precision="high",
             rx=_rx(r"[a-z]{3,}less$")),
        dict(label="nominalizer -ness",        gold=22, precision="high",
             rx=_rx(r"[a-z]{3,}ness$")),
        dict(label="nominalizer -ment",        gold=23, precision="high",
             rx=_rx(r"[a-z]{3,}ment$")),
        dict(label="nominalizer -ion/-tion",   gold=24, precision="high",
             rx=_rx(r"[a-z]{3,}tion$")),
        dict(label="nominalizer -ity",         gold=25, precision="high",
             rx=_rx(r"[a-z]{3,}ity$")),
    ],
}
_STRICT_PREC = {"high", "medium"}


# =============================================================================
# FREE-MORPHEME OVER-SEGMENTATION LITMUS  (v2.4 — partitioned)
# =============================================================================
# Function words have no inflectional morpheme boundary, so a morphological
# tokenizer "should" keep them whole. But not every split of a function word is
# equally bad. Empirically (English 10M) MorPiece splits two very different
# kinds of function word, and conflating them hides the real signal:
#
#   (A) FEATURAL splits — a left-edge cut isolates a shared GRAMMATICAL
#       feature across a closed class:
#         wh-  : who, what, when, which, where  → shared interrogative onset
#         th-  : that, this, they, their, there, the → shared demonstrative/
#                anaphoric onset
#       "wh++at", "th++ey" are NON-canonical morpheme boundaries, but they
#       carve out a distributionally real feature the LM can generalise over
#       (a single wh-/th- token shared across the whole paradigm). This is
#       arguably USEFUL, not a defect — so we score it separately and do NOT
#       count it against the tokenizer.
#
#   (B) ATOMIC splits — a cut on a genuinely monomorphemic word that isolates
#       NO shared feature: "i-s", "o-f", "t-o", "a-n", "b-y", "o-r". These are
#       pure over-generation: the Tolerance Principle firing on noise. This is
#       the REAL defect the litmus should measure.
#
# We therefore report THREE numbers:
#   atomic_overseg_rate    = (atomic words ever split) / (atomic words)   ← DEFECT
#   featural_isolation_rate= (featural words ever split)/(featural words) ← NEUTRAL/＋
#   persistence_index      = mean( log10(last_exp/first_exp+1) ) over ATOMIC
#                            offenders — high = the spurious split entrenches
#                            rather than being pruned (worse).
#
# A higher cutoff/min_frequency should lower atomic_overseg_rate and especially
# persistence; the featural rate may stay high (those splits are well-evidenced
# and stable). So the partition is what discriminates configs that the single
# flat rate (0.935 for all three English runs) could not.
#
# FEATURAL_PREFIXES defines, per language, the (prefix -> member words) closed
# classes that count as featural rather than atomic.
FEATURAL_PREFIXES = {
    "eng": {
        "wh": ["who", "what", "when", "which", "where", "why", "whom", "whose"],
        "th": ["that", "this", "they", "their", "there", "the", "them", "then",
               "these", "those", "they're"],
    },
    "ita": {
        # ch- interrogative/complementiser (che, chi), qu- wh-class
        # (quale, quando, quanto, quello, questo), de-/del- determiner-prep.
        "ch": ["che", "chi"],
        "qu": ["quale", "quando", "quanto", "quello", "questo", "quella",
               "questa", "quei", "quegli"],
    },
}

FUNCTION_WORDS = {
    "eng": [
        "the", "of", "and", "to", "in", "is", "was", "for", "on", "with",
        "as", "at", "by", "an", "be", "are", "from", "or", "that", "this",
        "but", "not", "they", "his", "her", "she", "him", "you", "your",
        "we", "our", "all", "can", "had", "has", "have", "were", "which",
        "their", "what", "when", "who", "will", "would", "there", "been",
    ],
    "ita": [
        "il", "lo", "la", "le", "gli", "di", "da", "in", "con", "su",
        "per", "tra", "fra", "che", "non", "ma", "se", "come", "anche",
        "una", "uno", "del", "della", "dei", "delle", "nel", "nella",
        "questo", "quella", "sono", "era", "ha", "hanno", "più", "molto",
    ],
}


def _featural_set(lang):
    """Return the set of function words that belong to a featural closed class."""
    out = set()
    for _pfx, members in FEATURAL_PREFIXES.get(lang, {}).items():
        out.update(members)
    return out


def _which_prefix(lang, word):
    """Return the featural prefix a word belongs to (or None)."""
    for pfx, members in FEATURAL_PREFIXES.get(lang, {}).items():
        if word in members:
            return pfx
    return None


def litmus_oversegmentation(lang, events):
    """Partitioned free-morpheme over-segmentation score from the event log.

    Splits function-word over-segmentation into:
      - ATOMIC offenders  (the real defect)
      - FEATURAL offenders (wh-/th- etc.; neutral-to-useful)
    and computes a persistence index over the atomic offenders.

    Returns a dict with both partitions and the headline atomic_overseg_rate.
    """
    from math import log10

    fw_all   = set(FUNCTION_WORDS.get(lang, []))
    featural = _featural_set(lang)
    # The atomic litmus set = canonical function words that are NOT featural.
    atomic_set = fw_all - featural
    # The featural litmus set we track = featural words that ALSO appear in the
    # canonical list (so we report on the same high-frequency items), PLUS any
    # featural member seen splitting in the log (so "where"/"why" are caught
    # even if not in FUNCTION_WORDS).
    tracked_featural = set(featural)

    details = {}   # word -> {count, first_exp, last_exp, example_piece, klass}
    for word, piece, exposure in events:
        if word in atomic_set:
            klass = "atomic"
        elif word in tracked_featural:
            klass = "featural"
        else:
            continue
        d = details.setdefault(word, dict(count=0, first_exp=exposure,
                                          last_exp=exposure, example_piece=piece,
                                          klass=klass,
                                          prefix=_which_prefix(lang, word)))
        d["count"]    += 1
        d["first_exp"] = min(d["first_exp"], exposure)
        d["last_exp"]  = max(d["last_exp"], exposure)

    atomic_off   = {w: d for w, d in details.items() if d["klass"] == "atomic"}
    featural_off = {w: d for w, d in details.items() if d["klass"] == "featural"}

    n_atomic   = len(atomic_set)
    n_featural = len(tracked_featural)

    # persistence: how long an atomic spurious split survives (log-span of
    # first..last exposure). High = entrenched (bad); ~0 = fired once, pruned.
    pers = []
    for d in atomic_off.values():
        span = (d["last_exp"] - d["first_exp"])
        pers.append(log10(span + 1))
    persistence_index = round(sum(pers) / len(pers), 3) if pers else 0.0

    # featural coherence: of the featural offenders, how many share the SAME
    # left-edge piece (i.e., the split actually isolates the prefix). We check
    # the example_piece's stem side begins with the prefix.
    coherent = 0
    for w, d in featural_off.items():
        pfx = d["prefix"]
        # piece like "th-ere" or "wh-at"; the left segment should equal prefix
        left = d["example_piece"].split("-", 1)[0] if "-" in d["example_piece"] else ""
        # MorPiece pieces may carry '++' on the suffix side; left side is the stem
        if pfx and left.startswith(pfx):
            coherent += 1

    return dict(
        # headline DEFECT metric
        n_atomic=n_atomic,
        n_atomic_oversegmented=len(atomic_off),
        atomic_overseg_rate=round(len(atomic_off) / n_atomic, 4) if n_atomic else 0.0,
        atomic_bad_events=sum(d["count"] for d in atomic_off.values()),
        persistence_index=persistence_index,
        # neutral/positive FEATURAL metric
        n_featural=n_featural,
        n_featural_split=len(featural_off),
        featural_isolation_rate=round(len(featural_off) / n_featural, 4) if n_featural else 0.0,
        featural_coherent=coherent,
        # detail + back-compat
        atomic_offenders=atomic_off,
        featural_offenders=featural_off,
        atomic_clean=sorted(atomic_set - set(atomic_off.keys())),
        # legacy flat fields (so older consumers don't break)
        n_function_words=len(fw_all),
        n_oversegmented=len(details),
        overseg_rate=round(len(details) / len(fw_all), 4) if fw_all else 0.0,
        total_bad_events=sum(d["count"] for d in details.values()),
    )


def print_litmus(litmus, lang):
    print(f"\n=== FREE-MORPHEME OVER-SEGMENTATION LITMUS (partitioned) | lang={lang} ===")
    print(f"  -- DEFECT: atomic function-word over-segmentation --")
    print(f"     atomic words tested   : {litmus['n_atomic']}")
    print(f"     over-segmented        : {litmus['n_atomic_oversegmented']}  "
          f"(rate={litmus['atomic_overseg_rate']:.3f}; 0.000 = perfect)")
    print(f"     spurious events       : {litmus['atomic_bad_events']}")
    print(f"     persistence index     : {litmus['persistence_index']:.3f}  "
          f"(log10 exposure-span; higher = more entrenched)")
    print(f"  -- NEUTRAL/USEFUL: featural (wh-/th-) isolation --")
    print(f"     featural words tracked: {litmus['n_featural']}")
    print(f"     split (feature-isolating): {litmus['n_featural_split']}  "
          f"(rate={litmus['featural_isolation_rate']:.3f})")
    print(f"     prefix-coherent splits: {litmus['featural_coherent']}/"
          f"{litmus['n_featural_split']}")
    if litmus["atomic_offenders"]:
        print(f"  atomic offenders (word x count, first..last, e.g. piece):")
        for w, d in sorted(litmus["atomic_offenders"].items(),
                           key=lambda kv: -kv[1]["count"]):
            print(f"      {w:8s} x{d['count']:<3d} "
                  f"{d['first_exp']:>9d}..{d['last_exp']:<9d} {d['example_piece']}")
    if litmus["featural_offenders"]:
        print(f"  featural splits (useful feature isolation):")
        for w, d in sorted(litmus["featural_offenders"].items(),
                           key=lambda kv: (kv[1]["prefix"] or "", -kv[1]["count"])):
            print(f"      [{d['prefix']}-] {w:8s} x{d['count']:<3d} "
                  f"{d['example_piece']}")


# =============================================================================
# EVENT-LOG LOADER
# =============================================================================

def load_event_log(path):
    """Parse ooa_vocab.tsv → list of (word, piece, exposure), in file order.

    The TSV written by MorPiece's save_ooa() has a header row followed by
    data rows with at least 10 tab-separated columns:
        Word | Piece | root_m | root_d | root_nd | infl_m | infl_d | infl_nd
             | vocab_size | token_exposure

    Returns a list of (word, piece, exposure_int).
    """
    events = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if i == 0 and parts[0].lower() in ("word", ""):
                continue                        # skip header
            if len(parts) < 10:
                continue
            try:
                word     = parts[0].strip()
                piece    = parts[1].strip()
                exposure = int(float(parts[9]))
            except (ValueError, IndexError):
                continue
            events.append((word, piece, exposure))
    return events


# =============================================================================
# SNAPSHOT LOADER
# =============================================================================

def _snapshot_sort_key(path):
    m = re.search(r"(\d+)(?=\.[^/\\]+$)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def load_snapshots(ooa_dir, glob_pat):
    """Return sorted list of (step_index, exposure_int, vocab_dict) tuples."""
    files = sorted(globmod.glob(os.path.join(ooa_dir, glob_pat)),
                   key=_snapshot_sort_key)
    snapshots = []
    for idx, fpath in enumerate(files, 1):
        exposure = _snapshot_sort_key(fpath)
        try:
            with open(fpath, encoding="utf-8") as fh:
                obj = json.load(fh)
            vocab = obj.get("vocab", obj) if isinstance(obj, dict) else {}
        except Exception:
            vocab = {}
        snapshots.append((idx, exposure, vocab))
    return snapshots


# =============================================================================
# ACQUISITION STEP RESOLUTION
# =============================================================================

def acquire_from_snapshots(exponents, snapshots, which="stable"):
    """Return (step, exposure) of 'first' or 'stable' acquisition, or None."""
    present_at = []
    for idx, exposure, vocab in snapshots:
        if any(e in vocab for e in exponents):
            present_at.append((idx, exposure))

    if not present_at:
        return None

    if which == "first":
        return present_at[0]

    # stable: first step after which the piece is present in every later snap
    present_set = {idx for idx, _ in present_at}
    total = len(snapshots)
    for i, (idx, exposure) in enumerate(present_at):
        remaining = [snapshots[j][0] for j in range(i, total)]
        if all(s in present_set for s in remaining):
            return (idx, exposure)
    return present_at[0]  # fallback: first occurrence


def acquire_from_events(matcher, events):
    """Return (word, exposure) of the first event matching a diagnostic regex."""
    for word, _piece, exposure in events:
        if matcher["rx"].search(word):
            return (word, exposure)
    return None


# =============================================================================
# STATISTICS
# =============================================================================

def partial_tau(tau_xy, tau_xz, tau_yz):
    """Partial Kendall τ of (x,y) controlling for z, via the standard formula.

    Returns float or None if the computation is degenerate (denominator ≤ 0,
    e.g. when N=2 or when one variable is constant).
    """
    denom2 = (1 - tau_xz ** 2) * (1 - tau_yz ** 2)
    if denom2 <= 0:
        return None
    return (tau_xy - tau_xz * tau_yz) / (denom2 ** 0.5)


def bootstrap_ci(acq, gold, n_boot=2000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    n   = len(acq)
    taus = []
    for _ in range(n_boot):
        idx  = rng.integers(0, n, size=n)
        a, g = np.array(acq)[idx], np.array(gold)[idx]
        if len(set(a)) < 2:
            continue
        taus.append(kendalltau(a, g, variant="b").statistic)
    if not taus:
        return (float("nan"), float("nan"))
    lo = np.percentile(taus, (1 - ci) / 2 * 100)
    hi = np.percentile(taus, (1 + ci) / 2 * 100)
    return (float(lo), float(hi))


def perm_test(acq, gold, n_perm=1000, seed=1):
    rng  = random.Random(seed)
    obs  = kendalltau(acq, gold, variant="b").statistic
    acq_ = list(acq)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(acq_)
        t = kendalltau(acq_, gold, variant="b").statistic
        if abs(t) >= abs(obs):
            count += 1
    return count / n_perm


# =============================================================================
# CORE ANALYSIS
# =============================================================================

MIN_N_WARN  = 5   # warn but still compute when N < this
MIN_N_SKIP  = 2   # hard skip only when N < 2


def run_analysis(gold, rows, strict, which, out_csv=None, lang="?"):
    """
    Compute τ-b, partial τ, bootstrap CI, permutation p for a set of rows.

    Parameters
    ----------
    gold   : list of GOLD_* dicts
    rows   : list of dict(label, gold, acq, freq, example)
    strict : bool — True = strict (bound, unambiguous only) analysis
    which  : "first" | "stable" — acquisition step definition
    out_csv: optional path to write per-morpheme CSV
    lang   : language tag for display

    Returns
    -------
    dict of analysis results (written to JSON by caller).
    """
    label = "strict" if strict else "full"
    n     = len(rows)

    if n < MIN_N_SKIP:
        print(f"\n=== {label.upper()} analysis | {which} ===")
        print(f"  [!] only {n} usable morphemes ({label}) -- cannot compute correlation (need ≥{MIN_N_SKIP}).")
        return dict(n=n, tau=None, perm_p=None, ci=(None, None),
                    rho=None, partial_tau=None, freq_tau=None,
                    note=f"too few morphemes: {n} < {MIN_N_SKIP}")

    if n < MIN_N_WARN:
        print(f"\n=== {label.upper()} analysis | {which} ===")
        print(f"  [⚠] only {n} usable morphemes ({label}) — results are indicative only (N<{MIN_N_WARN}).")

    acq_vals  = [r["acq"]  for r in rows]
    gold_vals = [r["gold"] for r in rows]
    freq_vals = [r["freq"] for r in rows]

    tau_res   = kendalltau(acq_vals, gold_vals, variant="b")
    tau       = float(tau_res.statistic)
    tau_p_asym = float(tau_res.pvalue)
    rho, rho_p = spearmanr(acq_vals, gold_vals)
    lo, hi    = bootstrap_ci(acq_vals, gold_vals)
    perm_p    = perm_test(acq_vals, gold_vals)

    # frequency baseline + partial τ
    have_freq = any(f > 0 for f in freq_vals)
    freq_tau  = None
    ptau      = None
    if have_freq:
        tau_gold_freq = float(kendalltau(gold_vals, freq_vals, variant="b").statistic)
        tau_acq_freq  = float(kendalltau(acq_vals,  freq_vals, variant="b").statistic)
        freq_tau      = tau_gold_freq
        ptau          = partial_tau(tau, tau_acq_freq, tau_gold_freq)

    missing = [g["label"] for g in gold
               if (not strict or (g.get("bound") and not g.get("ambiguous")))
               and not any(r["label"] == g["label"] for r in rows)]

    # ---- print ---------------------------------------------------------------
    print(f"\n=== {label.upper()} analysis (acq={which}) | event-log | lang={lang} ===")
    print(f"  n morphemes        : {n}  ({'⚠ small N' if n < MIN_N_WARN else label})")
    print(f"  Kendall tau-b      : {tau:+.3f}   (perm p={perm_p:.4f}, asym p={tau_p_asym:.4f})")
    print(f"  95% bootstrap CI   : [{lo:+.3f}, {hi:+.3f}]")
    print(f"  Spearman rho       : {float(rho):+.3f}   (p={rho_p:.4f})")
    if have_freq:
        print(f"  baseline tau(freq,gold)       : {freq_tau:+.3f}   <- OOA must beat this")
        ptau_str = f"{ptau:+.3f}" if ptau is not None else "n/a (degenerate, N too small)"
        print(f"  PARTIAL tau(acq,gold | freq)   : {ptau_str}   <- HEADLINE")
    else:
        print(f"  [!] no frequency data -> partial-correlation control skipped.")
    for r in sorted(rows, key=lambda x: x["gold"]):
        ex = f"  e.g. {r['example']}" if r.get("example") else ""
        print(f"      tier {r['gold']:2d}  exp={r['freq']:>10.0f}  {r['label']}{ex}")
    if missing:
        print(f"  not acquired: {', '.join(missing)}")

    # ---- CSV -----------------------------------------------------------------
    if out_csv:
        import csv
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["label", "gold_rank", "ooa_step", "frequency", "example"])
            for r in sorted(rows, key=lambda x: x["gold"]):
                w.writerow([r["label"], r["gold"], r["acq"], r["freq"],
                            r.get("example", "")])
        print(f"  wrote per-morpheme table -> {out_csv}")

    return dict(
        n=n, tau=tau, perm_p=perm_p,
        ci=(lo, hi), rho=float(rho), rho_p=rho_p,
        partial_tau=ptau, freq_tau=freq_tau,
        # 'acq' is the internal name; 'ooa_step' is the alias plot_ooa_results.py
        # reads. Emit both so the external plotter works without modification.
        rows=[dict(label=r["label"], gold=r["gold"],
                   acq=r["acq"], ooa_step=r["acq"],
                   freq=r["freq"], example=r.get("example",""))
              for r in rows],
    )


# =============================================================================
# EVENT-LOG ANALYSIS DRIVER
# =============================================================================

def analyse_event_log(lang, event_log_path, out_prefix, which="first"):
    """Run strict + full analyses over the event-log for one language."""
    gold     = {"ita": GOLD_ITA, "eng": GOLD_ENG}[lang]
    matchers = EVENT_MATCHERS[lang]

    print(f"\nloading {event_log_path!r} ...")
    events = load_event_log(event_log_path)
    print(f"loaded {len(events)} split events from {os.path.basename(event_log_path)}")

    results = {}
    for strict, label in [(True, "strict"), (False, "full")]:
        rows = []
        for m in matchers:
            is_strict_eligible = m["precision"] in _STRICT_PREC
            if strict and not is_strict_eligible:
                continue
            hit = acquire_from_events(m, events)
            if hit is None:
                continue
            word, acq_step = hit
            # count of events matching this matcher as frequency proxy
            freq = sum(1 for w, _, _ in events if m["rx"].search(w))
            # find matching GOLD entry for this matcher's label
            gold_tier = m["gold"]
            rows.append(dict(
                label=m["label"], gold=gold_tier, acq=acq_step,
                freq=freq, example=f"{word}",
            ))

        # deduplicate: if two matchers map to same gold tier, keep earliest acq
        seen = {}
        for r in sorted(rows, key=lambda x: x["acq"]):
            if r["gold"] not in seen:
                seen[r["gold"]] = r
        rows = list(seen.values())

        res = run_analysis(gold, rows, strict=strict, which="event-log",
                           out_csv=f"{out_prefix}_{label}.csv", lang=lang)
        results[label] = res

    # --- free-morpheme over-segmentation litmus (v2.3) -----------------------
    litmus = litmus_oversegmentation(lang, events)
    print_litmus(litmus, lang)
    results["litmus"] = litmus

    return results


# =============================================================================
# SNAPSHOT ANALYSIS DRIVER
# =============================================================================

def analyse_snapshots(lang, ooa_dir, glob_pat, which, freq_path, out_prefix):
    """Run strict + full analyses over vocab snapshots for one language."""
    gold      = {"ita": GOLD_ITA, "eng": GOLD_ENG}[lang]
    snapshots = load_snapshots(ooa_dir, glob_pat)
    if not snapshots:
        print(f"[!] no snapshots matched {glob_pat!r} in {ooa_dir!r}")
        return {}

    # optional external frequency file
    freq_map = {}
    if freq_path and os.path.isfile(freq_path):
        with open(freq_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    try:
                        freq_map[parts[0]] = float(parts[1])
                    except ValueError:
                        pass

    final_vocab = snapshots[-1][2] if snapshots else {}

    results = {}
    for strict, label in [(True, "strict"), (False, "full")]:
        rows = []
        for g in gold:
            if strict and not (g.get("bound") and not g.get("ambiguous")):
                continue
            hit = acquire_from_snapshots(g["exponents"], snapshots, which=which)
            if hit is None:
                continue
            acq_idx, acq_exposure = hit
            freq = freq_map.get(g["label"], 0)
            if freq == 0:
                # fallback: sum frequencies of exponents from final snapshot
                freq = sum(final_vocab.get(e, 0) for e in g["exponents"])
            rows.append(dict(
                label=g["label"], gold=g["gold"],
                acq=acq_exposure, freq=freq,
            ))

        res = run_analysis(gold, rows, strict=strict, which=which,
                           out_csv=f"{out_prefix}_{label}.csv", lang=lang)
        results[label] = res

    return results


# =============================================================================
# PLOTTING  (always runs; annotates N on figure titles)
# =============================================================================

def _try_import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_scatter(result, path, title=""):
    """Developmental tier (x) vs acquisition exposure (y), labelled scatter."""
    plt = _try_import_mpl()
    if plt is None:
        return False
    rows = result.get("rows", [])
    if not rows:
        return False

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [r["gold"] for r in rows]
    ys = [r["acq"]  for r in rows]
    ax.scatter(xs, ys, zorder=3)
    for r in rows:
        ax.annotate(r["label"].split()[0], (r["gold"], r["acq"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    tau   = result.get("tau")
    ptau  = result.get("partial_tau")
    n     = result.get("n", len(rows))
    warn  = " ⚠ small N" if n < MIN_N_WARN else ""
    tau_s = f"τ={tau:+.3f}" if tau is not None else "τ=n/a"
    pt_s  = (f"  partial τ={ptau:+.3f}" if ptau is not None else
             "  partial τ=n/a")
    ax.set_xlabel("Developmental tier (Brown/de Villiers / Belletti-Guasti)")
    ax.set_ylabel("Acquisition exposure (tokens)")
    ax.set_title(f"{title}  N={n}{warn}\n{tau_s}{pt_s}", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def plot_comparison(all_results, path, lang):
    """Bar chart: partial τ per tokenizer variant, with N annotation."""
    plt = _try_import_mpl()
    if plt is None:
        return False

    entries = []   # (variant, strict_tau, full_tau, strict_n, full_n)
    for variant, res in all_results.items():
        s = res.get("strict", {})
        f = res.get("full",   {})
        entries.append((variant,
                        s.get("partial_tau") or s.get("tau"),
                        f.get("partial_tau") or f.get("tau"),
                        s.get("n", 0), f.get("n", 0)))

    if not entries:
        return False

    labels   = [e[0] for e in entries]
    strict_v = [e[1] if e[1] is not None else 0 for e in entries]
    full_v   = [e[2] if e[2] is not None else 0 for e in entries]
    x        = np.arange(len(labels))
    w        = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 4))
    b1 = ax.bar(x - w/2, strict_v, w, label="strict (bound, unambig)")
    b2 = ax.bar(x + w/2, full_v,   w, label="full (incl. ambiguous)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Partial τ (OOA vs gold | freq)")
    ax.set_title(f"OOA trajectory — partial τ by tokenizer variant  [{lang}]")
    ax.legend(fontsize=8)
    # annotate N
    for bar, e in zip(b1, entries):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"N={e[3]}", ha="center", va="bottom", fontsize=7)
    for bar, e in zip(b2, entries):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"N={e[4]}", ha="center", va="bottom", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


# =============================================================================
# DEMO
# =============================================================================

def make_demo(gold, n_snaps=60, noise=0.15, seed=1):
    rng    = random.Random(seed)
    ranks  = {g["label"]: g["gold"] for g in gold}
    n_gold = max(ranks.values()) if ranks else 14
    exps   = {g["label"]: g["exponents"][0] for g in gold if g["exponents"]}
    entry  = {exp: min(n_snaps, max(1, int(rank/n_gold * n_snaps
                                           + rng.gauss(0, noise*n_snaps))))
              for label, exp in exps.items()
              for rank in [ranks.get(label, n_gold)]}
    counts = {e: max(5, int(5000 / (entry[e] + 1))) for e in entry}
    # also add some distractors
    for d in ["xyz", "qqq", "zz", "pfx"]:
        entry[d]  = rng.randint(1, n_snaps)
        counts[d] = rng.randint(1, 50)

    tmp = tempfile.mkdtemp(prefix="ooa_demo_")
    for step in range(1, n_snaps + 1):
        exposure = step * 1000
        vocab    = {}
        for e, ent in entry.items():
            if ent <= step:
                if counts[e] < 30 and step < ent + 3 and rng.random() < 0.3:
                    continue
                vocab[e] = counts[e]
        path = os.path.join(tmp, f"vocab_{exposure}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"vocab": vocab}, fh)
    return tmp, "vocab_*.json"


# =============================================================================
# COMPARISON / SUMMARY
# =============================================================================

def write_summary_csv(all_results, path, lang):
    import csv
    rows_out = []
    for variant, res in all_results.items():
        s = res.get("strict", {})
        f = res.get("full",   {})
        rows_out.append({
            "lang":             lang,
            "variant":          variant,
            "n_strict":         s.get("n"),
            "strict_stable_tau":s.get("tau"),
            "strict_partial_tau":s.get("partial_tau"),
            "strict_freq_tau":  s.get("freq_tau"),
            "strict_perm_p":    s.get("perm_p"),
            "n_full":           f.get("n"),
            "full_stable_tau":  f.get("tau"),
            "full_partial_tau": f.get("partial_tau"),
        })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {path} ({len(rows_out)} rows)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate MorPiece OOA trajectory against developmental gold order."
    )
    ap.add_argument("--lang",        choices=["ita","eng"], default="ita")
    ap.add_argument("--ooa_dir",     default=None,
                    help="Directory of vocab_<N>.json snapshots (snapshot mode).")
    ap.add_argument("--glob",        default="vocab_*.json",
                    help="Glob pattern for snapshot files (default: vocab_*.json).")
    ap.add_argument("--which",       choices=["first","stable"], default="stable",
                    help="Snapshot acquisition step: first vs stable (default: stable).")
    ap.add_argument("--from-event-log", dest="event_log", default=None,
                    help="Path to ooa_vocab.tsv (event-log mode).")
    ap.add_argument("--freq",        default=None,
                    help="Optional TSV (piece<TAB>count) for frequency control.")
    ap.add_argument("--out",         default=None,
                    help="Output directory for JSON/CSV/figures. May also be a "
                         "*.csv path (its parent dir is used as the output dir) "
                         "for backward compatibility with older sbatch scripts.")
    ap.add_argument("--json",        default=None,
                    help="Explicit path for the structured JSON result. If "
                         "omitted, <out_dir>/<variant>.json is used.")
    ap.add_argument("--variant",     default=None,
                    help="Short label for this tokenizer run (used in filenames/plots).")
    ap.add_argument("--demo",        action="store_true",
                    help="Run synthetic demo and exit.")
    ap.add_argument("--compare-dir", dest="compare_dir", default=None,
                    help="Directory of result JSON files to aggregate into summary.")
    # ---- backward-compat no-op flags (v2.0 sbatch interface) ----------------
    # Both the strict AND full analyses always run now, and the trajectory is
    # always written, so these are accepted but ignored. Kept so existing
    # sbatch command lines do not error out.
    ap.add_argument("--full",        action="store_true",
                    help="(deprecated, no-op) full analysis always runs.")
    ap.add_argument("--trajectory",  action="store_true",
                    help="(deprecated, no-op) trajectory always written.")
    args = ap.parse_args()

    # ---- demo ----------------------------------------------------------------
    if args.demo:
        gold     = {"ita": GOLD_ITA, "eng": GOLD_ENG}[args.lang]
        demo_dir, demo_glob = make_demo(gold)
        variant  = f"demo_{args.lang}"
        out_dir  = os.path.join(tempfile.gettempdir(), "ooa_demo_out")
        prefix   = os.path.join(out_dir, variant)
        res = analyse_snapshots(args.lang, demo_dir, demo_glob,
                                which="stable", freq_path=None,
                                out_prefix=prefix)
        os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)
        for label, r in res.items():
            plot_scatter(r,
                         os.path.join(out_dir, "figs", f"{variant}_{label}_scatter.png"),
                         title=f"{variant} [{label}]")
        print(f"\nDemo outputs in {out_dir}/")
        return

    # ---- comparison-only mode ------------------------------------------------
    if args.compare_dir:
        jsons = globmod.glob(os.path.join(args.compare_dir, "*.json"))
        all_r = {}
        for j in jsons:
            with open(j, encoding="utf-8") as fh:
                d = json.load(fh)
            all_r[d.get("variant", os.path.splitext(os.path.basename(j))[0])] = d.get("analyses", d)
        if all_r:
            fig_dir = os.path.join(args.compare_dir, "figs")
            plot_comparison(all_r, os.path.join(fig_dir, "comparison_partial_tau.png"),
                            lang=args.lang)
            write_summary_csv(all_r, os.path.join(args.compare_dir, "summary.csv"),
                              lang=args.lang)
        return

    # ---- single run ----------------------------------------------------------
    variant = args.variant or (
        os.path.basename(args.event_log).replace(".tsv","") if args.event_log
        else os.path.basename(args.ooa_dir or "run")
    )
    # --out may be a directory OR (legacy sbatch) a *.csv file: in the latter
    # case use its parent directory as the output dir.
    if args.out and args.out.lower().endswith(".csv"):
        out_dir = os.path.dirname(args.out) or "."
    else:
        out_dir = args.out or os.path.join("results", variant)
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    prefix  = os.path.join(out_dir, variant)

    print(f"\nRUN  lang={args.lang}  variant={variant}")

    if args.event_log:
        analyses = analyse_event_log(args.lang, args.event_log,
                                     out_prefix=prefix, which="first")
    elif args.ooa_dir:
        analyses = analyse_snapshots(args.lang, args.ooa_dir, args.glob,
                                     which=args.which, freq_path=args.freq,
                                     out_prefix=prefix)
    else:
        ap.error("provide --ooa_dir (snapshot mode) or --from-event-log (event-log mode).")

    # always plot, even for small N (litmus has no rows/tau — skip it)
    for label, res in analyses.items():
        if label == "litmus":
            continue
        scatter_path = os.path.join(fig_dir, f"{variant}_{label}_scatter.png")
        ok = plot_scatter(res, scatter_path,
                          title=f"{variant} [{label}] lang={args.lang}")
        if ok:
            print(f"  wrote fig → {scatter_path}")

    # ---- emit JSON with BOTH naming schemes for downstream compatibility -----
    # Internal keys are 'strict'/'full'.  The aggregator in run_ooa_eval.sbatch
    # and plot_ooa_results.py expect 'strict_stable'/'full_stable'.  We expose
    # the analyses under the *_stable names (the acquisition step is the stable
    # one in snapshot mode; in event-log mode it is the first matching event,
    # which is the analogue) and keep the short names too.  The litmus is NOT
    # an analysis block (no rows/tau): it is lifted OUT of `analyses` to a
    # sibling top-level key so the sbatch aggregator (which indexes a["n"],
    # a["tau"] for every analyses entry) never trips over it.
    litmus_block = analyses.pop("litmus", None)
    analyses_out = {}
    for short, res in analyses.items():
        analyses_out[f"{short}_stable"] = res     # what downstream reads
        analyses_out[short] = res                  # short alias

    payload = dict(lang=args.lang, variant=variant,
                   event_log=args.event_log, ooa_dir=args.ooa_dir,
                   source=("event_log" if args.event_log else "snapshot"),
                   analyses=analyses_out,
                   litmus=litmus_block)
    json_path = args.json or os.path.join(out_dir, f"{variant}.json")
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nwrote structured results -> {json_path}")
    print("=" * 60)


# =============================================================================
# BATCH RUNNER  (called from sbatch; replaces the old inline RUNS loop)
# =============================================================================

def run_batch(runs, compare_out_dir, lang):
    """
    runs : list of dict(variant, event_log=None, ooa_dir=None, glob, which, freq)
    Runs each, collects JSON results, writes comparison figures + summary CSV.
    """
    all_results = {}
    for cfg in runs:
        variant = cfg["variant"]
        out_dir = os.path.join(compare_out_dir, variant)
        os.makedirs(out_dir, exist_ok=True)
        prefix  = os.path.join(out_dir, variant)
        print(f"\nRUN  lang={lang}  variant={variant}")
        if cfg.get("event_log"):
            analyses = analyse_event_log(lang, cfg["event_log"],
                                         out_prefix=prefix, which="first")
        else:
            analyses = analyse_snapshots(lang, cfg["ooa_dir"],
                                         cfg.get("glob","vocab_*.json"),
                                         which=cfg.get("which","stable"),
                                         freq_path=cfg.get("freq"),
                                         out_prefix=prefix)
        payload = dict(lang=lang, variant=variant, analyses=analyses)
        json_path = os.path.join(out_dir, f"{variant}.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"wrote structured results -> {json_path}")

        # scatter figures per run
        fig_dir = os.path.join(out_dir, "figs")
        os.makedirs(fig_dir, exist_ok=True)
        for label, res in analyses.items():
            if label == "litmus":
                continue
            p = os.path.join(fig_dir, f"{variant}_{label}_scatter.png")
            if plot_scatter(res, p, title=f"{variant} [{label}]"):
                print(f"  wrote fig -> {p}")

        all_results[variant] = {k: v for k, v in analyses.items() if k != "litmus"}
        print("=" * 60)

    if all_results:
        comp_fig_dir = os.path.join(compare_out_dir, "figs")
        os.makedirs(comp_fig_dir, exist_ok=True)
        plot_comparison(all_results,
                        os.path.join(comp_fig_dir, f"comparison_{lang}_partial_tau.png"),
                        lang=lang)
        write_summary_csv(all_results,
                          os.path.join(compare_out_dir, "summary.csv"),
                          lang=lang)
        s_col = "HEADLINE (strict_stable_tau): lang variant n tau partial_tau freq_tau perm_p"
        print(f"\n{s_col}")
    print(f"DONE. comparison in {compare_out_dir}  (summary.csv, figs/)")


if __name__ == "__main__":
    main()