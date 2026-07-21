#!/usr/bin/env python3
"""
SHARED multilingual UniGram tokenizer (eng / nld / zho) for the single-model
BabyLM-2026 submission.

  * trains over MULTIPLE corpora in one stream (--input_paths, comma-separated),
    one per language, with an optional per-source cap for balanced merges;
  * vocab_size defaults to 32768 (2**15): top of BabyBabelLM's 8k-32k GoldFish
    range, ~30% of a hidden-512 model's params as a tied embedding, enough budget
    for Chinese once English+Dutch share Latin subwords;
  * reports PER-LANGUAGE fertility (tokens / character and tokens / whitespace-
    word) plus byte-fallback/<unk> usage, so the 32k-vs-48k decision is grounded:
    if Chinese tokens/char is well above ~1.0, retrain at --vocab_size 48000.

Back-compatible: --input_path (single) still works for monolingual runs.
"""

import argparse
import json
import pyarrow.parquet as pq
from collections import Counter
from tqdm import tqdm
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers import Tokenizer, pre_tokenizers, decoders, normalizers, Regex, processors
from transformers import PreTrainedTokenizerFast
from pathlib import Path


def initialize_tokenizer(args):
    start_of_text_symbol = "<s>"
    end_of_text_symbol = "</s>"
    unk_symbol = "<unk>"
    mask_symbol = "<mask>"
    pad_symbol = "<pad>"

    special_tokens = [unk_symbol, start_of_text_symbol, end_of_text_symbol, pad_symbol, mask_symbol]
    special_tokens += [f"<special_{i}>" for i in range(11)]

    tokenizer = Tokenizer(Unigram(byte_fallback=True))

    # ---- normalizer ----
    normalizer_list = []
    if args.lowercase:
        normalizer_list.append(normalizers.Lowercase())
    normalizer_list.append(normalizers.Prepend(" "))
    if args.strip_accents:
        normalizer_list.append(normalizers.StripAccents())
    normalizer_list.append(normalizers.NFKC())                       # unifies CJK width etc.
    # baseline newline handling: KEEP "\n" as a token (normalise the spacing around
    # it) rather than collapsing newlines into spaces.
    normalizer_list.append(normalizers.Replace(Regex(r"\n"), "\n "))
    normalizer_list.append(normalizers.Replace(Regex(r" *\n"), "\n"))
    tokenizer.normalizer = normalizers.Sequence(normalizer_list)

    # ---- pre-tokenizer ----
    # Optimized Script-Splitting Regex Rule for ENG, NLD, and ZHO:
    # 1. Separates Chinese characters (\u4e00-\u9fff) individually.
    # 2. Employs the baseline tokenization split for Latin text.
    script_aware_regex = Regex(
        r"[\u4e00-\u9fff]|"
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
        r" ?\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )

    pre_tokenizer_list = [
        pre_tokenizers.Split(
            script_aware_regex,
            behavior="isolated",
            invert=False
        ),
    ]
    # Note: ByteLevel configurations are omitted because Unigram models natively 
    # extract the initial atomic alphabet directly from corpus character frequency.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre_tokenizer_list)

    # ---- decoder ----
    tokenizer.decoder = decoders.Sequence([decoders.Strip(' ', 1, 0)])

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{start_of_text_symbol} $A {end_of_text_symbol}",
        pair=f"{start_of_text_symbol} $A {end_of_text_symbol} {start_of_text_symbol} $B {end_of_text_symbol}",
        special_tokens=[(start_of_text_symbol, 1), (end_of_text_symbol, 2)],
    )

    # UnigramTrainer optimizes from an oversized candidate pool down to target vocab_size
    trainer = UnigramTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        unk_token=unk_symbol
    )
    return tokenizer, trainer, special_tokens


# --------------------------------------------------------------------------- #
#  multi-source reading
# --------------------------------------------------------------------------- #
def _read_path(path: Path, text_column: str = "text", max_lines: int = 0):
    """Yield up to max_lines (0 = all) non-empty texts from a parquet/text file
    or a directory of them."""
    if path.is_file():
        parquet_files = [path] if path.suffix == ".parquet" else []
        text_files = [] if path.suffix == ".parquet" else [path]
    else:
        parquet_files = sorted(path.glob("*.parquet"))
        text_files = [f for f in sorted(path.glob("*"))
                      if f.is_file() and f.suffix != ".parquet"]
    n = 0
    for fp in parquet_files:
        table = pq.read_table(fp, columns=[text_column])
        for value in tqdm(table[text_column], desc=f"Reading {fp.name}", total=len(table[text_column])):
            text = value.as_py()
            if text and text.strip():
                yield text.strip()
                n += 1
                if max_lines and n >= max_lines:
                    return
    for fp in text_files:
        with fp.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Reading {fp.name}"):
                text = line.strip()
                if text:
                    yield text
                    n += 1
                    if max_lines and n >= max_lines:
                        return


def multi_iterator(paths, max_lines_per_source=0, text_column="text"):
    for p in paths:
        yield from _read_path(Path(p), text_column=text_column, max_lines=max_lines_per_source)


def resolve_input_paths(args):
    if args.input_paths:
        return [s for s in args.input_paths.split(",") if s]
    return [str(args.input_path)]


# --------------------------------------------------------------------------- #
#  saving (unchanged except it records the source list)
# --------------------------------------------------------------------------- #
def save_transformers_tokenizer(tokenizer, output_dir, special_tokens, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    tokenizer_config = {
        "add_prefix_space": False, "added_tokens_decoder": {}, "bos_token": "<s>",
        "clean_up_tokenization_spaces": True, "eos_token": "</s>", "mask_token": "<mask>",
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": "<pad>", "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>",
    }
    for idx, token in enumerate(special_tokens):
        tokenizer_config["added_tokens_decoder"][str(idx)] = {
            "content": token, "lstrip": False, "normalized": False,
            "rstrip": False, "single_word": False, "special": True}
    with (output_dir / "tokenizer_config.json").open("w") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    special_tokens_map = {
        "bos_token": "<s>", "eos_token": "</s>", "mask_token": "<mask>",
        "pad_token": "<pad>", "unk_token": "<unk>",
        "additional_special_tokens": ["A:", "B:", "C:", "D:", "<??>", "<?>", "<F>", "XYZ",
                                      "*TARGET_CHILD:", "*INVESTIGATOR:"],
    }
    with (output_dir / "special_tokens_map.json").open("w") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)

    training_config = {
        "vocab_size": args.vocab_size, 
        "min_frequency": getattr(args, 'min_frequency', None), # Unigram does not use frequency thresholds
        "strip_accents": args.strip_accents,
        "lowercase": args.lowercase, 
        "input_paths": resolve_input_paths(args),
        "languages": args.lang_names,
    }
    with (output_dir / "training_config.json").open("w") as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)
    print(f"Tokenizer saved to {output_dir} (Unigram-based, lowercase={args.lowercase})")


# --------------------------------------------------------------------------- #
#  PER-LANGUAGE fertility / coverage stats (grounds the vocab-size decision)
# --------------------------------------------------------------------------- #
def per_source_stats(tokenizer, paths, lang_names, max_lines_per_source, out_stats_path):
    names = lang_names.split(",") if lang_names else [Path(p).stem for p in paths]
    if len(names) < len(paths):
        names += [Path(p).stem for p in paths[len(names):]]

    lines = [f"{'lang':>8} | {'tok/char':>9} | {'tok/word':>9} | {'<unk>%':>7} | {'chars':>7} | {'docs':>8}"]
    lines.append("-" * len(lines[0]))
    for name, p in zip(names, paths):
        n_tok = n_char = n_word = n_unk = n_docs = 0
        seen_chars = set()
        for text in _read_path(Path(p), max_lines=(max_lines_per_source or 20000)):
            enc = tokenizer.encode(text)
            n_tok += len(enc.ids)
            n_unk += sum(1 for t in enc.tokens if t == "<unk>")
            n_char += len(text)
            n_word += len(text.split())
            seen_chars.update(text)
            n_docs += 1
        tpc = n_tok / n_char if n_char else 0.0
        tpw = n_tok / n_word if n_word else 0.0
        unkp = 100.0 * n_unk / n_tok if n_tok else 0.0
        lines.append(f"{name:>8} | {tpc:9.3f} | {tpw:9.3f} | {unkp:7.3f} | "
                     f"{len(seen_chars):7d} | {n_docs:8d}")

    report = "\n".join(lines)
    print("\nPER-LANGUAGE fertility (sampled):\n" + report)
    print("\nRead: For ZHO, a high <unk>% means Chinese characters are being dropped due to the small vocabulary limit.")
    print("If ZHO <unk>% is high (>1.0%), consider using a stratified/character pre-allocation method.")
    with open(out_stats_path, "a") as f:
        f.write("\n\nPER-LANGUAGE fertility (sampled):\n" + report + "\n")


def calculate_global_stats(tokenizer, paths, args):
    counter, n_words = Counter(), 0
    n_subwords = 0
    for text in multi_iterator(paths, max_lines_per_source=(args.max_lines_per_source or 20000)):
        n_words += len(text.split())
        toks = tokenizer.encode(text).tokens
        counter.update(toks)
        n_subwords += len(toks)
    if not counter:
        print("Warning: no tokens collected — skipping global statistics.")
        return
    sorted_subwords = counter.most_common()
    n_unk = counter.get("<unk>", 0)
    oov_rate = n_unk / n_subwords if n_subwords else 0.0
    avg_splits = n_subwords / n_words if n_words else 0.0
    hapax_ratio = sum(1 for _, f in sorted_subwords if f == 1) / len(sorted_subwords)
    stats_path = Path(args.vocab_path) / "tokenizer_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"Vocabulary size: {args.vocab_size}\n")
        f.write(f"Lowercase: {args.lowercase}\n")
        f.write(f"Average splits per whitespace-word (mixed langs): {avg_splits:.3f}\n")
        f.write(f"Hapax ratio: {hapax_ratio:.4f}\n")
        f.write(f"OOV rate (<unk>/total tokens): {oov_rate:.4f}\n")
    print(f"\nGlobal: avg_splits/word={avg_splits:.3f}  hapax={hapax_ratio:.4f}  "
          f"OOV={oov_rate:.4f}  (OOV > 0 is expected for Unigram if target characters are unmapped)")
    return stats_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shared multilingual Unigram tokenizer (eng/nld/zho)')
    parser.add_argument('--input_paths', type=str, default=None,
                        help='COMMA-SEPARATED list of corpora to TRAIN on (parquet/text/dir). '
                             'For a baseline match, point this at the single balanced mixture.')
    parser.add_argument('--stats_paths', type=str, default=None,
                        help='COMMA-SEPARATED per-language corpora for the fertility report '
                             '(aligned with --lang_names). Defaults to --input_paths. Use this '
                             'to keep per-language stats when training on one mixed file.')
    parser.add_argument('--input_path', type=Path,
                        default="/home/cristiano.chesi/pr_baby_lm/data/eng/clean_train_100M",
                        help='single-corpus fallback (back-compat)')
    parser.add_argument('--lang_names', type=str, default="eng,nld,zho",
                        help='comma-separated labels aligned with --input_paths (for stats)')
    parser.add_argument('--max_lines_per_source', type=int, default=0,
                        help='cap lines read per language for balanced merges (0 = all)')
    parser.add_argument('--vocab_path', type=Path, default="./tokenizer/unigram_eng_nld_zho_16k",
                        help='output directory')
    parser.add_argument('--vocab_size', type=int, default=16384,
                        help='shared vocab size (BabyLM 2026 baseline standard is 16384)')
    parser.add_argument('--lowercase', action='store_true', default=True,
                        help='lowercase (default ON, as in your pipeline)')
    parser.add_argument('--no_lowercase', action='store_false', dest='lowercase',
                        help='preserve case (matches GPT-2/GoldFish; keeps Dutch capitalization)')
    parser.add_argument('--strip_accents', action='store_true', default=False)
    parser.add_argument('--no_strip_accents', action='store_false', dest='strip_accents')
    args = parser.parse_args()

    paths = resolve_input_paths(args)
    stats_paths = ([s for s in args.stats_paths.split(",") if s]
                   if args.stats_paths else paths)
    print("Configuration:", flush=True)
    print(f"  input_paths : {paths}")
    print(f"  languages   : {args.lang_names}")
    print(f"  vocab_size  : {args.vocab_size}")
    print(f"  lowercase   : {args.lowercase}")
    print(f"  max_lines/src: {args.max_lines_per_source or 'all'}\n")

    print("Initializing a Unigram tokenizer", flush=True)
    tokenizer, trainer, special_tokens = initialize_tokenizer(args)

    print("Training the tokenizer over all sources", flush=True)
    tokenizer.train_from_iterator(
        multi_iterator(paths, max_lines_per_source=args.max_lines_per_source), trainer)

    print("Saving the tokenizer in Transformers format", flush=True)
    save_transformers_tokenizer(tokenizer, args.vocab_path, special_tokens, args)

    print("\nReloading with PreTrainedTokenizerFast...")
    try:
        fast = PreTrainedTokenizerFast.from_pretrained(str(args.vocab_path))
        for t in ["this is a test sentence.", "dit is een testzin.", "这是一个测试句子。"]:
            ids = fast.encode(t)
            print(f"  {t!r} -> {len(ids)} ids -> {fast.decode(ids)!r}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")

    print("\nCalculating statistics...")
    stats_path = calculate_global_stats(tokenizer, paths, args)
    if stats_path:
        per_source_stats(tokenizer, stats_paths, args.lang_names, args.max_lines_per_source, stats_path)
