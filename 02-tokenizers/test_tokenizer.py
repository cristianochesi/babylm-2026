import argparse
import csv
from difflib import SequenceMatcher
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm

from transformers import AutoTokenizer

# ======================================================================
# SIGMORPHON 2022 Shared Task on Morpheme Segmentation - official metrics
# ======================================================================
# Separator inserted between morphemes before the *character-level*
# Levenshtein distance is computed. The official task data marks morpheme
# boundaries with " @@"; the choice of marker only rescales the absolute
# Levenshtein distance (it does NOT affect Prec/Rec/F1, which work on the
# morpheme lists). Keep it fixed to whatever produced the reference table
# below so the numbers stay comparable.
MORPH_SEP = " @@"

# Reference scores reproduced from Table 3 of
# "Linguistic Units as Tokens: Intrinsic and Extrinsic Evaluation with
# BabyLM" (Fusco et al., BabyLM 2025) -- https://aclanthology.org/2025.babylm-main.35
# Kept here so every run prints a side-by-side comparison.
#            name        avg_lev  precision  recall   f1
REFERENCE_RESULTS = [
    ("BPE",        2.08,    21.03,    26.62,  23.50),
    ("MoP",        1.96,    24.54,    29.52,  26.80),
    ("SillyTok",   2.77,    12.45,    18.81,  14.98),
    ("ParFind",    1.24,    38.99,    30.12,  33.99),
]

words = []
categories = []
splits = []
segmentations_original = []
segmentations_new = []
predictions = []
accuracy_normalized = []
accuracy = []


# ----------------------------------------------------------------------
# SIGMORPHON 2022 metric primitives
# ----------------------------------------------------------------------
def levenshtein(s1, s2):
    """Character-level Levenshtein (edit) distance between two strings.

    This is the distance used by the official SIGMORPHON 2022 evaluation
    script, computed here with an O(min(m, n)) two-row DP.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current = [i + 1]
        for j, c2 in enumerate(s2):
            insert = previous[j + 1] + 1
            delete = current[j] + 1
            substitute = previous[j] + (c1 != c2)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def morpheme_overlap(gold_morphs, pred_morphs):
    """True positives = size of the multiset intersection of two morpheme lists.

    Mirrors the overlap counting in the official SIGMORPHON 2022 script:
    each predicted morpheme can be matched against at most one (identical)
    gold morpheme.
    """
    remaining = list(gold_morphs)
    tp = 0
    for m in pred_morphs:
        if m in remaining:
            remaining.remove(m)
            tp += 1
    return tp


def sigmorphon_word_scores(gold_seg, pred_seg):
    """Per-word SIGMORPHON-2022 statistics.

    `gold_seg` / `pred_seg` are whitespace-separated morpheme strings.
    Returns: (levenshtein_distance, true_positives, n_gold, n_pred).
    Aggregating these across the corpus yields Avg. Lev. Dist. / Prec / Rec / F1.
    """
    gold_morphs = gold_seg.split()
    pred_morphs = pred_seg.split()
    # Re-join with a single canonical separator so the Levenshtein distance
    # is independent of how separators were stored upstream.
    gold_str = MORPH_SEP.join(gold_morphs)
    pred_str = MORPH_SEP.join(pred_morphs)
    lev = levenshtein(gold_str, pred_str)
    tp = morpheme_overlap(gold_morphs, pred_morphs)
    return lev, tp, len(gold_morphs), len(pred_morphs)


def aggregate_sigmorphon(levs, tps, gold_counts, pred_counts):
    """Corpus-level SIGMORPHON 2022 metrics.

    Precision/Recall are micro-averaged (totals over the whole dataset),
    exactly as in the official evaluation script.
    Returns: (avg_lev, precision, recall, f1)  -- P/R/F1 in percent.
    """
    n = len(levs)
    total_tp = sum(tps)
    total_gold = sum(gold_counts)
    total_pred = sum(pred_counts)

    avg_lev = sum(levs) / n if n else 0.0
    precision = 100.0 * total_tp / total_pred if total_pred else 0.0
    recall = 100.0 * total_tp / total_gold if total_gold else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return avg_lev, precision, recall, f1


# ----------------------------------------------------------------------
# Parallel evaluation
# ----------------------------------------------------------------------
def process_chunk(chunk_data):
    tokenizer, chunk_with_indices, gold_segmentations = chunk_data
    local_results = []

    for idx, word in chunk_with_indices:
        tokens = tokenizer.tokenize(word.lower().replace('-', ''))
        prediction = " ".join(d.replace("Ġ", "").replace("▁", "").replace("++", "") for d in tokens)
        gold = gold_segmentations[idx]

        acc_norm = token_edit_distance(gold, prediction)
        acc = morpheme_edit_distance(gold, prediction)
        lev, tp, n_gold, n_pred = sigmorphon_word_scores(gold, prediction)

        local_results.append((idx, prediction, acc, acc_norm, lev, tp, n_gold, n_pred))

    return local_results


def test_simoprhon_parallel(tokenizer, words, gold_segmentations, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()

    # Split into chunks with original indices
    indexed_words = list(enumerate(words))
    chunk_size = len(words) // num_processes + 1
    chunks = [indexed_words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

    # Prepare args
    chunk_data = [(tokenizer, chunk, gold_segmentations) for chunk in chunks]

    results = []
    with Pool(processes=num_processes) as pool:
        for chunk_result in tqdm(pool.map(process_chunk, chunk_data), total=len(chunk_data)):
            results.extend(chunk_result)

    # Sort results back by original index
    results.sort(key=lambda x: x[0])
    predictions = [r[1] for r in results]
    accuracy = [r[2] for r in results]
    accuracy_normalized = [r[3] for r in results]
    levenshteins = [r[4] for r in results]
    true_positives = [r[5] for r in results]
    gold_counts = [r[6] for r in results]
    pred_counts = [r[7] for r in results]

    return (predictions, accuracy, accuracy_normalized,
            levenshteins, true_positives, gold_counts, pred_counts)


def token_edit_distance(tokens1, tokens2):
    matcher = SequenceMatcher(None, tokens1, tokens2)
    matches = sum(triple.size for triple in matcher.get_matching_blocks())
    if len(tokens1) == 0:
        return matches/1
    else:
        return matches/len(tokens1)


def morpheme_edit_distance(tokens1, tokens2):
    t1 = tokens1.split()
    t2 = tokens2.split()
    ratio = 0
    for i in range(len(t1)):
        if len(t2) > i:
            matcher = SequenceMatcher(None, t1[i], t2[i])
            ratio += matcher.ratio()
            # print(t1[i], t2[i], ratio)
    if len(t1) == 0:
        return ratio / 1
    else:
        return ratio / len(t1)


def tokenization_accuracy(gold_tokens, proposed_tokens):
    return 1 - morpheme_edit_distance(gold_tokens, proposed_tokens)


# ----------------------------------------------------------------------
# Results table (reference rows + this run)
# ----------------------------------------------------------------------
def format_results_table(computed_row):
    """Return the SIGMORPHON comparison table as a plain-text string."""
    header = ["Tokenizer", "Avg. Lev. Dist.", "Prec", "Rec", "F1"]
    rows = [list(r) for r in REFERENCE_RESULTS] + [list(computed_row)]

    str_rows = []
    for name, lev, prec, rec, f1 in rows:
        str_rows.append([str(name), f"{lev:.2f}", f"{prec:.2f}",
                         f"{rec:.2f}", f"{f1:.2f}"])

    widths = [max(len(header[i]), max(len(r[i]) for r in str_rows))
              for i in range(len(header))]

    def fmt(cols):
        cells = []
        for i, c in enumerate(cols):
            cells.append(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]))
        return "  ".join(cells)

    line = "-" * (sum(widths) + 2 * (len(widths) - 1))
    out = [fmt(header), line]
    out += [fmt(r) for r in str_rows]
    return "\n".join(out)


def save_results_table(tokenizer_folder, computed_row, path="metrics_comparison.tsv"):
    """Write the comparison table to a TSV file."""
    rows = [list(r) for r in REFERENCE_RESULTS] + [list(computed_row)]
    with open(tokenizer_folder+path, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Tokenizer', 'Avg. Lev. Dist.', 'Prec', 'Rec', 'F1'])
        for name, lev, prec, rec, f1 in rows:
            writer.writerow([name, f"{lev:.2f}", f"{prec:.2f}",
                             f"{rec:.2f}", f"{f1:.2f}"])
    print(f"Saved comparison table -> {tokenizer_folder+path}")


def plot_results_table(tokenizer_folder, computed_row, path="metrics_comparison.png"):
    """Render the comparison table + bar charts as a publication-ready figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib not available - skipping plot.")
        return

    rows = [list(r) for r in REFERENCE_RESULTS] + [list(computed_row)]
    names = [r[0] for r in rows]
    levs = [r[1] for r in rows]
    precs = [r[2] for r in rows]
    recs = [r[3] for r in rows]
    f1s = [r[4] for r in rows]
    this_idx = len(rows) - 1  # the computed (this-run) row

    fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(2, 2, height_ratios=[1.0, 1.3], hspace=0.35, wspace=0.3)

    # --- top: the table itself ---
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis("off")
    ax_t.set_title("SIGMORPHON 2022 morpheme-segmentation metrics",
                   fontweight="bold", pad=12)
    cell_text = [[f"{lev:.2f}", f"{p:.2f}", f"{r:.2f}", f"{f:.2f}"]
                 for _, lev, p, r, f in rows]
    table = ax_t.table(cellText=cell_text,
                       rowLabels=names,
                       colLabels=["Avg. Lev. Dist.", "Prec", "Rec", "F1"],
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    # highlight this run's row (data cells + row-label cell at column -1)
    for col in list(range(4)) + [-1]:
        cell = table.get_celld().get((this_idx + 1, col))
        if cell is not None:
            cell.set_facecolor("#fff3cd")

    x = range(len(names))
    bar_colors = ["#9aa0a6"] * len(names)
    bar_colors[this_idx] = "#d97706"

    # --- bottom-left: Precision / Recall / F1 ---
    ax_b = fig.add_subplot(gs[1, 0])
    width = 0.26
    ax_b.bar([i - width for i in x], precs, width, label="Prec", color="#4c72b0")
    ax_b.bar(list(x), recs, width, label="Rec", color="#55a868")
    ax_b.bar([i + width for i in x], f1s, width, label="F1", color="#c44e52")
    ax_b.set_xticks(list(x))
    ax_b.set_xticklabels(names, rotation=20, ha="right")
    ax_b.set_ylabel("Score (%)")
    ax_b.set_title("Precision / Recall / F1  (higher is better)")
    ax_b.legend(fontsize=8)
    ax_b.grid(axis="y", alpha=0.3)

    # --- bottom-right: Avg. Levenshtein distance ---
    ax_l = fig.add_subplot(gs[1, 1])
    ax_l.bar(list(x), levs, 0.6, color=bar_colors)
    ax_l.set_xticks(list(x))
    ax_l.set_xticklabels(names, rotation=20, ha="right")
    ax_l.set_ylabel("Avg. Levenshtein distance")
    ax_l.set_title("Avg. Lev. Distance  (lower is better)")
    ax_l.grid(axis="y", alpha=0.3)

    fig.savefig(tokenizer_folder+path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure -> {tokenizer_folder+path}")


# ----------------------------------------------------------------------
# Sanity check
# ----------------------------------------------------------------------
def test_tokenization(tokenizer, test_texts=None):
    """Test tokenizer functionality"""
    print("\n" + "="*50)
    print("TESTING TOKENIZATION")
    print("="*50)
    
    if test_texts is None:
        test_texts = [
             "<pad> This is a test!",
             "routed into very very deepest intuistion lexicalization is not an issue.",
             "*CHI: what which whatever which when where whose who do u do?",
             "%err: sanity check. gorking morhing mother father sister",
             "4x2s snowball superhero immortal inconsisten immaterial immigration"
         ]
    
    try:
        for text in test_texts:
            print(f"\nOriginal: {text}")
            
            # Encode
            ids = tokenizer.encode(text)
            print(f"Encoded: {ids}")
            
            # Decode
            decoded = tokenizer.decode(ids)
            print(f"Decoded: {decoded}")
        
        print("\n✅ Tokenization sanity check completed")
        return True
        
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    
    parser = argparse.ArgumentParser(description='Test Tokenizer with a specific benchmark')
    parser.add_argument('--tokenizer_folder', type=str, required=False, default='./tokenizer', help='Path to the corpus file to be used for training')
    parser.add_argument('--benchmark_file', type=str, required=False, default='./benchmarks/sigmorphon2022_eng_adapted.csv', help='Path to the folder for saving the trained tokenizer')
    parser.add_argument('--output_folder', type=str, required=False, default='./benchmarks_results', help='Path to the folder for saving the benchamrk results')
    parser.add_argument('--no_plot', action='store_true', help='Skip writing the comparison figure (PNG)')
    args = parser.parse_args()

    benchmark_basename = os.path.basename(args.benchmark_file)          # sigmorphon2022_eng_adapted.csv
    tokenizer_name = os.path.basename(args.tokenizer_folder)
    lang = benchmark_basename.split("_")[1]
    
    # ── output dir: <tokenizer_dir>/<lang_code>/  ──
    # tokenizer_folder may point to the .json file itself; strip to get the dir
    tok_dir = args.tokenizer_folder
    if os.path.isfile(tok_dir) or tok_dir.endswith('.json'):
        tok_dir = os.path.dirname(tok_dir)                              # BPE_16K_multilingual/
    output_folder = os.path.join(tok_dir, "evaluation") + os.sep        # BPE_16K_multilingual/evaluation/
    os.makedirs(output_folder, exist_ok=True)
    benchmark_file = args.benchmark_file
    
    #for multilingual evaluation only
    output_folder = os.path.join(tok_dir, lang) + os.sep 
    os.makedirs(output_folder, exist_ok=True)

    # tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

    test_tokenization(tokenizer)
    
    with open(benchmark_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)  # Uses headers from the first line
        for row in reader:
            words.append(row['word'])
            categories.append(row['category'])
            splits.append(row['split'])
            segmentations_original.append(row['segmentation_original'].replace("@",""))
            segmentations_new.append(row['segmentation_new'].replace("@",""))

    # Gold standard for the intrinsic evaluation: the shallow / surface
    # segmentation, in line with the BabyLM 2025 paper setup.
    gold_segmentations = segmentations_new

    (predictions, accuracy, accuracy_normalized,
     levenshteins, true_positives,
     gold_counts, pred_counts) = test_simoprhon_parallel(
        tokenizer, words, gold_segmentations)

    # ---- per-word results ----
    rows = list(zip(words, categories, segmentations_original, segmentations_new,
                    predictions, accuracy, accuracy_normalized, levenshteins))
    with open(output_folder+tokenizer_name+'_'+'results.tsv', 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Words', 'Categories', 'Original segmentation',
                         'New segmentation', 'Proposed segmentation',
                         'Morphemes identification accuracy', 'Edit distance',
                         'Levenshtein distance'])
        writer.writerows(rows)

    # ---- per-category morpheme-identification accuracy (unchanged) ----
    category_groups = defaultdict(list)
    for word, category, seg_orig, seg_new, pred, acc, acc_norm, lev in rows:
        category_groups[str(category)].append(acc)

    average_accuracy_by_category = {}
    for category, accuracies in category_groups.items():
        average_accuracy_by_category[str(category)] = sum(accuracies) / len(accuracies)

    print("\nAverage Accuracy by Category:")
    for category, avg_acc in average_accuracy_by_category.items():
        print(f"{category}: {avg_acc:.4f}")

    # ---- per-category SIGMORPHON 2022 metrics ----
    cat_stats = defaultdict(lambda: ([], [], [], []))
    for i, category in enumerate(categories):
        levs_c, tps_c, gold_c, pred_c = cat_stats[str(category)]
        levs_c.append(levenshteins[i])
        tps_c.append(true_positives[i])
        gold_c.append(gold_counts[i])
        pred_c.append(pred_counts[i])

    with open(output_folder+tokenizer_name+'_'+'metrics_by_category.tsv', 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Category', 'N', 'Avg. Lev. Dist.', 'Prec', 'Rec', 'F1'])
        print("\nSIGMORPHON 2022 metrics by category:")
        print("""\nhttps://aclanthology.org/2022.sigmorphon-1.11.pdf\n000 : Root words (free morphemes) progress ==> progress\n100 : Inflection only prepared ==> prepare @@ed\n010 : Derivation only intensive ==> intense @@ive\n001 : Compound only hotpot ==> hot @@pot\n101 : Inflection and Compound wheelbands ==> wheel @@band @@s\n011 : Derivation and Compound tankbuster ==> tank @@bust @@er\n110 : Inflection and Derivation urbanizes ==> urban @@ize @@s\n111 : Inflection, Derivation, Compound trackworkers ==> track @@work @@er @@s """)
        for category in sorted(cat_stats):
            levs_c, tps_c, gold_c, pred_c = cat_stats[category]
            avg_lev_c, prec_c, rec_c, f1_c = aggregate_sigmorphon(
                levs_c, tps_c, gold_c, pred_c)
            writer.writerow([category, len(levs_c), f"{avg_lev_c:.2f}",
                             f"{prec_c:.2f}", f"{rec_c:.2f}", f"{f1_c:.2f}"])
            print(f"  {category}: Lev={avg_lev_c:.2f}  P={prec_c:.2f}  "
                  f"R={rec_c:.2f}  F1={f1_c:.2f}  (n={len(levs_c)})")

    # ---- overall SIGMORPHON 2022 metrics for this tokenizer ----
    avg_lev, precision, recall, f1 = aggregate_sigmorphon(
        levenshteins, true_positives, gold_counts, pred_counts)
    computed_row = (tokenizer_name, avg_lev, precision, recall, f1)

    # ---- comparison table (always printed / saved / plotted) ----
    print("\n" + "=" * 60)
    print("SIGMORPHON 2022 morpheme-segmentation metrics")
    print("=" * 60)
    print(format_results_table(computed_row))
    print()

    save_results_table(output_folder+tokenizer_name, computed_row)
    if not args.no_plot:
        plot_results_table(output_folder+tokenizer_name, computed_row)

    print("\n✅ Tokenization test completed")
        
if __name__ == "__main__":
    main()