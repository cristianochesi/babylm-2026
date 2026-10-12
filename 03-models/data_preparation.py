#!/usr/bin/env python3
"""
BabyLM-2026 data preparation -- MERGED, multi-regimen.

One PrepareDataset, four selectable training regimens (--regimen):

  * baseline      -- the BabyLM GPT-2 baseline (babylm-baselines/strict-gpt2,
                     data_utils.py). Tokenise each blank-line DOCUMENT whole and
                     cut it into CONTIGUOUS, NON-OVERLAPPING (seq_length-2)-token
                     chunks; each datapoint is [<s>] + chunk + [</s>]. Every
                     token is a target ~ONCE per epoch; full seq_length context;
                     BOS/EOS wrapping is INTRINSIC (independent of
                     add_special_tokens). This is the submission-compliant one.

  * naturalistic  -- one example per line (utterance/sentence). No packing, no
                     overlap; short, padded sequences; BOS/EOS only if
                     add_special_tokens.

  * conversational-- each example is line_i + line_{i+1} WITHIN one document
                     (utterance pairs). Consecutive examples OVERLAP (line i+1 is
                     the response of example i and the prompt of example i+1), so
                     every interior token is a target ~TWICE per epoch. Short,
                     padded, two-utterance context.

  * sliding_window-- stride-1 windows over the concatenated token stream
                     (len = total_tokens - seq_length). Massive overlap; combine
                     with the trainer's --stride to subsample (e.g. --stride
                     seq_length recovers non-overlapping windows). Kept for
                     ablations.

Regimen-independent contract consumed by train_multilingual.py:
  * __getitem__ -> (input LongTensor, target LongTensor), target padded with -100
  * corpus_stats() -> {model_tokens, utf8_bytes, ws_words, lines, documents}
    (each surviving line/document counted ONCE -- overlap in conversational /
    sliding_window does NOT inflate the totals; the trainer applies the
    per-language Byte Premium, so this class stays language-agnostic.)

Reading: a single .txt/.parquet file OR a directory; blank-line blocks are
DOCUMENTS (a file boundary also ends a document), exactly the blocks written by
build_multilingual_dataset.py. The '.report.txt' sidecar is skipped. Parquet is
read on the first run and then served from the .cache pickle.
"""

import os
import pickle
import hashlib
import threading
import fcntl
from pathlib import Path
from typing import Optional, List, Tuple, Iterator

import torch

VALID_REGIMENS = ('baseline', 'naturalistic', 'conversational', 'sliding_window')


class PrepareDataset(torch.utils.data.Dataset):

    _cache_lock = threading.Lock()

    def __init__(
        self,
        file_path: str,
        model_name: str,
        tokenizer,
        seq_length: int,
        regimen: str = 'baseline',
        add_special_tokens: bool = False,    # honoured by the three legacy
                                             # regimens; 'baseline' always wraps
        start_token: str = '<s>',
        end_token: str = '</s>',
        pad_token: str = '<pad>',
        cache_dir: Optional[str] = None,
    ):
        if regimen not in VALID_REGIMENS:
            raise ValueError(f"regimen must be one of {VALID_REGIMENS}, got {regimen!r}")

        self.file_path = file_path
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.seq_length = int(seq_length)
        self.regimen = regimen
        self.add_special_tokens = bool(add_special_tokens)
        self.start_token, self.end_token, self.pad_token = start_token, end_token, pad_token

        # special-token ids, read from the tokenizer (NOT hardcoded). Resolved for
        # every regimen; whether they are *applied* depends on the regimen
        # (baseline: always; legacy: only if add_special_tokens).
        self.pad_token_id = self._resolve('pad_token_id', pad_token, default=0)
        self.bos_token_id = self._resolve('bos_token_id', start_token, default=None)
        self.eos_token_id = self._resolve('eos_token_id', end_token, default=None)
        # legacy aliases used by the line/pair/window __getitem__
        self.start_token_id = self.bos_token_id
        self.end_token_id = self.eos_token_id
        if self.regimen == 'baseline' and (self.bos_token_id is None or self.eos_token_id is None):
            print("Warning: <s>/</s> id not found on the tokenizer; baseline "
                  "datapoints will NOT be BOS/EOS wrapped (the baseline expects them).")

        if cache_dir is None:
            cache_dir = os.path.join(file_path, '.cache')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"{self.regimen}_{self._cache_key()}.pkl")

        if self._load_cache():
            print(f"[data] loaded {len(self):,} {self.regimen} datapoints from cache "
                  f"{self.cache_file}")
        else:
            print(f"[data] no cache; reading + tokenizing corpus ({self.regimen})...")
            self._build()
            self._save_cache()

    # ------------------------------------------------------------------ #
    #  tokenizer helpers
    # ------------------------------------------------------------------ #
    def _vocab(self) -> dict:
        try:
            if hasattr(self.tokenizer, 'get_vocab'):
                return self.tokenizer.get_vocab()
            if hasattr(self.tokenizer, 'vocab'):
                return self.tokenizer.vocab
        except Exception:
            pass
        return {}

    def _resolve(self, attr: str, token_str: str, default):
        v = getattr(self.tokenizer, attr, None)
        if v is not None:
            return int(v)
        vid = self._vocab().get(token_str)
        return int(vid) if vid is not None else default

    def _encode(self, text: str) -> List[int]:
        # add_special_tokens=False: the tokenizer's own post-processor must NOT
        # inject <s>/</s>; this dataset controls special tokens uniformly.
        try:
            try:
                e = self.tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                e = self.tokenizer.encode(text)
            if hasattr(e, 'ids'):
                return e.ids
            return e if isinstance(e, list) else list(e)
        except Exception as ex:
            print(f"Warning: encode failed: {ex}")
            return []

    # ------------------------------------------------------------------ #
    #  corpus files
    # ------------------------------------------------------------------ #
    def _files(self) -> List[str]:
        """Files the active regimen will read, RECURSIVELY (so a nested
        <root>/<lang>/<source>.txt layout from build_multilingual_dataset.py is
        read whole). baseline -> .txt only; legacy regimens -> any file
        (incl. .parquet). Hidden dirs/files (e.g. .cache) and the builder's
        '.report.txt' sidecar are always skipped. Each FILE is a document
        boundary, so 'conversational' pairs stay WITHIN one source file."""
        if os.path.isfile(self.file_path):
            return [self.file_path]
        base = Path(self.file_path)
        out = []
        for p in sorted(base.rglob('*')):
            if not p.is_file():
                continue
            rel = p.relative_to(base)
            if any(part.startswith('.') for part in rel.parts):
                continue                                   # skip .cache, hidden
            if p.name.endswith('.report.txt'):
                continue                                   # skip sampling report
            if self.regimen == 'baseline':
                if p.name.endswith('.txt'):
                    out.append(str(p))
            else:
                out.append(str(p))
        return out

    # ------------------------------------------------------------------ #
    #  build dispatch
    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        if self.regimen == 'baseline':
            self._build_baseline()
        else:
            self._build_legacy()

    # ---- baseline: per-document non-overlapping (seq_length-2) chunks ---- #
    def _iter_documents(self) -> Iterator[Tuple[str, int, int, int]]:
        """Yield (doc_text, n_bytes, n_ws_words, n_lines) per blank-line block.
        A file boundary also ends the current document."""
        for path in self._files():
            buf: List[str] = []
            try:
                if path.endswith('.parquet'):
                    for ln in self._read_parquet_lines(path):
                        if ln.strip() == '':
                            if buf:
                                yield self._finish(buf); buf = []
                        else:
                            buf.append(ln.rstrip('\n'))
                    if buf:
                        yield self._finish(buf); buf = []
                    continue
                with open(path, 'r', encoding='utf-8') as f:
                    for raw in f:
                        if raw.strip() == '':
                            if buf:
                                yield self._finish(buf); buf = []
                        else:
                            buf.append(raw.rstrip('\n'))
                if buf:
                    yield self._finish(buf); buf = []
            except Exception as ex:
                print(f"Warning: read failed {path}: {ex}")

    @staticmethod
    def _finish(lines: List[str]) -> Tuple[str, int, int, int]:
        # text keeps "\n" between lines (the baseline tokenises the joined
        # document, so model_tokens includes newline tokens), but the byte/word
        # totals count CONTENT only -- inter-line "\n" bytes are excluded -- so
        # corpus_stats matches the per-line legacy regimens exactly.
        text = '\n'.join(lines)
        content_bytes = sum(len(l.encode('utf-8')) for l in lines)
        return (text, content_bytes,
                sum(len(l.split()) for l in lines), len(lines))

    def _build_baseline(self) -> None:
        chunk_size = max(1, self.seq_length - 2)     # reserve <s> and </s>
        self.chunks: List[List[int]] = []
        tot_tokens = tot_bytes = tot_words = tot_lines = n_docs = 0
        for text, nb, nw, nl in self._iter_documents():
            ids = self._encode(text)
            if not ids:
                continue
            n_docs += 1
            tot_bytes += nb; tot_words += nw; tot_lines += nl
            tot_tokens += len(ids)
            for i in range(0, len(ids), chunk_size):
                self.chunks.append(ids[i:i + chunk_size])
        self._stats = {'model_tokens': int(tot_tokens), 'utf8_bytes': int(tot_bytes),
                       'ws_words': int(tot_words), 'lines': int(tot_lines),
                       'documents': int(n_docs)}
        print(f"[data] baseline: {len(self.chunks):,} datapoints "
              f"(<=seq_length-2={chunk_size} tokens + <s>/</s>) from {n_docs} documents; "
              f"{tot_tokens:,} tokens, {tot_words:,} ws-words")

    # ---- legacy: per-line read with doc ids, then regimen-specific build ---- #
    def _read_parquet_lines(self, path: str, text_column: str = 'text') -> List[str]:
        """Read the `text` column of a parquet file as line-strings (each row
        split on newlines, readlines-style), so blank-line document structure is
        honoured identically to .txt."""
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=[text_column])
        lines = []
        for value in table[text_column]:
            s = value.as_py()
            if s is None:
                continue
            for ln in str(s).split('\n'):
                lines.append(ln + '\n')
        return lines

    def _process_and_tokenize(self) -> None:
        """Read all lines, assign a DOCUMENT ID per non-empty line (new document
        at every file boundary and after every blank line), drop empty/empty-
        tokenizing lines keeping doc ids aligned. Sets self.lines, self.line_doc_id,
        self.total_model_tokens."""
        raw = []                       # None marks a file boundary
        for path in self._files():
            try:
                raw.append(None)
                if path.endswith('.parquet'):
                    raw.extend(self._read_parquet_lines(path))
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        raw.extend(f.readlines())
            except Exception as e:
                print(f"Warning: Error reading file {os.path.basename(path)}: {e}")
                continue

        clean_lines, clean_doc_ids = [], []
        doc_id, new_doc = -1, True
        for item in raw:
            if item is None:
                new_doc = True; continue
            s = item.strip()
            if not s:
                new_doc = True; continue
            if new_doc:
                doc_id += 1; new_doc = False
            clean_lines.append(s)
            clean_doc_ids.append(doc_id)

        self.lines, self.line_doc_id = [], []
        num_tokens = 0
        for line, did in zip(clean_lines, clean_doc_ids):
            ids = self._encode(line)
            if ids:
                num_tokens += len(ids)
                self.lines.append(line)
                self.line_doc_id.append(did)
        self.total_model_tokens = int(num_tokens)
        n_docs = len(set(self.line_doc_id)) if self.line_doc_id else 0
        print(f"Filtered dataset: {len(self.lines)} valid lines across {n_docs} documents "
              f"after removing empty tokenizations ({num_tokens} tokens)")

    def _build_legacy(self) -> None:
        self._process_and_tokenize()
        if self.regimen == 'naturalistic':
            self._prepare_naturalistic()
        elif self.regimen == 'conversational':
            self._prepare_conversational()
        else:
            self._prepare_sliding_window()
        # regimen-independent corpus stats (each surviving line counted once)
        lines = self.lines
        self._stats = {
            'model_tokens': int(self.total_model_tokens),
            'utf8_bytes': int(sum(len(l.encode('utf-8')) for l in lines)),
            'ws_words': int(sum(len(l.split()) for l in lines)),
            'lines': len(lines),
            'documents': len(set(self.line_doc_id)) if self.line_doc_id else 0,
        }

    def _prepare_naturalistic(self):
        """
        Accumulate consecutive utterances from the same document until
        seq_length is reached.

        No overlap.
        """
        self.tokenized_lines = []

        current = []
        current_doc = None
        target = self.seq_length

        for line, did in zip(self.lines, self.line_doc_id):
            ids = self._encode(line)
            if not ids:
                continue

            if current_doc is None:
                current_doc = did

            # never cross document boundaries
            if did != current_doc:
                if current:
                    self.tokenized_lines.append(current)

                current = ids
                current_doc = did
                continue

            if len(current) + len(ids) <= target:
                current.extend(ids)
            else:
                if current:
                    self.tokenized_lines.append(current)

                current = ids

        if current:
            self.tokenized_lines.append(current)

        print(
            f"Naturalistic regimen: "
            f"{len(self.tokenized_lines)} packed sequences "
            f"(target={target} tokens)"
        )

    def _prepare_conversational(self):
        """
        Pack consecutive utterances up to seq_length.

        The final utterance of a chunk is carried forward as the first
        utterance of the next chunk (within the same document) to preserve
        conversational continuity.

        Example:

            [u1 u2 u3 u4] -> chunk A
                     |
                     +---- overlap

            [u4 u5 u6 u7] -> chunk B
        """

        self.tokenized_pairs = []

        target = self.seq_length

        current_chunk = []
        current_doc = None

        last_utterance = None

        for line, did in zip(self.lines, self.line_doc_id):

            ids = self._encode(line)
            if not ids:
                continue

            if current_doc is None:
                current_doc = did

            # document boundary
            if did != current_doc:

                if current_chunk:
                    self.tokenized_pairs.append(current_chunk)

                current_chunk = []
                current_doc = did
                last_utterance = None

            # still fits
            if len(current_chunk) + len(ids) <= target:

                current_chunk.extend(ids)
                last_utterance = ids

            else:
                # flush old chunk
                if current_chunk:
                    self.tokenized_pairs.append(current_chunk)

                # start next chunk with overlap
                current_chunk = []

                if last_utterance is not None:
                    current_chunk.extend(last_utterance)

                current_chunk.extend(ids)

                last_utterance = ids

        if current_chunk:
            self.tokenized_pairs.append(current_chunk)

        print(
            f"Conversational regimen: "
            f"{len(self.tokenized_pairs)} packed conversations "
            f"(target={target} tokens, 1-utterance overlap)"
        )

    def _prepare_sliding_window(self) -> None:
        all_tokens = []
        for line in self.lines:
            ids = self._encode(line)
            if ids:
                all_tokens.extend(ids)
        self.all_tokens = all_tokens
        print(f"Sliding window regimen: {len(self.all_tokens)} total tokens")

    # ------------------------------------------------------------------ #
    #  stats for the trainer's byte-premium accounting
    # ------------------------------------------------------------------ #
    def corpus_stats(self) -> dict:
        return getattr(self, '_stats',
                       {'model_tokens': 0, 'utf8_bytes': 0, 'ws_words': 0,
                        'lines': 0, 'documents': 0})

    # ------------------------------------------------------------------ #
    #  cache (atomic, file-locked, regimen-aware)
    # ------------------------------------------------------------------ #
    def _cache_key(self) -> str:
        finfo = []
        for p in self._files():
            try:
                st = os.stat(p)
                finfo.append((os.path.basename(p), int(st.st_mtime), st.st_size))
            except Exception:
                pass
        s = (f"{self.model_name}|{self.regimen}|sl{self.seq_length}|"
             f"ast{self.add_special_tokens}|"
             f"bos{self.bos_token_id}|eos{self.eos_token_id}|pad{self.pad_token_id}|"
             f"vocab{len(self._vocab())}|{finfo}")
        return hashlib.md5(s.encode()).hexdigest()[:16]

    def _payload(self) -> dict:
        if self.regimen == 'baseline':
            return {'chunks': self.chunks}
        if self.regimen == 'naturalistic':
            return {'tokenized_lines': self.tokenized_lines}
        if self.regimen == 'conversational':
            return {'tokenized_pairs': self.tokenized_pairs}
        return {'all_tokens': self.all_tokens}

    def _restore(self, d: dict) -> None:
        if self.regimen == 'baseline':
            self.chunks = d['chunks']
        elif self.regimen == 'naturalistic':
            self.tokenized_lines = d['tokenized_lines']
        elif self.regimen == 'conversational':
            self.tokenized_pairs = d['tokenized_pairs']
        else:
            self.all_tokens = d['all_tokens']

    def _load_cache(self) -> bool:
        if not os.path.exists(self.cache_file):
            return False
        try:
            with open(self.cache_file, 'rb') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    d = pickle.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            if d.get('regimen') != self.regimen:
                return False
            self._restore(d)
            self._stats = d['stats']
            return True
        except Exception as ex:
            print(f"[data] cache load failed ({ex}); rebuilding")
            return False

    def _save_cache(self) -> None:
        d = {'regimen': self.regimen, 'stats': self._stats}
        d.update(self._payload())
        tmp = self.cache_file + '.tmp'
        try:
            with open(tmp, 'wb') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            os.replace(tmp, self.cache_file)
            print(f"[data] cached -> {self.cache_file}")
        except Exception as ex:
            print(f"[data] cache save failed: {ex}")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    #  Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        if self.regimen == 'baseline':
            return len(self.chunks)
        if self.regimen == 'naturalistic':
            return len(self.tokenized_lines)
        if self.regimen == 'conversational':
            return len(self.tokenized_pairs)
        return max(0, len(self.all_tokens) - self.seq_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.regimen == 'baseline':
            return self._getitem_baseline(idx)
        return self._getitem_legacy(idx)

    # ---- baseline datapoint: [<s>] + chunk + [</s>], padded to seq_length-1 ---- #
    def _getitem_baseline(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = list(self.chunks[idx])
        if self.bos_token_id is not None:
            seq = [self.bos_token_id] + seq
        if self.eos_token_id is not None:
            seq = seq + [self.eos_token_id]
        if len(seq) < 2:
            seq = seq + [self.pad_token_id] * (2 - len(seq))

        inp, tgt = seq[:-1], seq[1:]
        L = self.seq_length - 1
        if len(inp) > L:
            inp, tgt = inp[:L], tgt[:L]
        elif len(inp) < L:
            pad = L - len(inp)
            inp = inp + [self.pad_token_id] * pad
            tgt = tgt + [-100] * pad
        return (torch.tensor(inp, dtype=torch.long),
                torch.tensor(tgt, dtype=torch.long))

    # ---- legacy datapoint: line / pair / window, padded to seq_length ---- #
    def _getitem_legacy(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            if self.regimen == 'naturalistic':
                sequence = list(self.tokenized_lines[idx])
            elif self.regimen == 'conversational':
                sequence = list(self.tokenized_pairs[idx])
            else:  # sliding_window
                end = idx + self.seq_length + 1
                if end > len(self.all_tokens):
                    avail = len(self.all_tokens) - idx
                    sequence = (list(self.all_tokens[idx:idx + avail]) if avail > 0
                                else [self.pad_token_id] * 2)
                else:
                    sequence = list(self.all_tokens[idx:end])

            if not sequence:
                sequence = [self.pad_token_id, self.pad_token_id]

            if self.add_special_tokens:
                if self.start_token_id is not None:
                    sequence = [self.start_token_id] + sequence
                if self.end_token_id is not None:
                    sequence = sequence + [self.end_token_id]

            if len(sequence) < 2:
                sequence = sequence + [self.pad_token_id] * (2 - len(sequence))

            inp, tgt = sequence[:-1], sequence[1:]
            if len(inp) > self.seq_length:
                inp, tgt = inp[:self.seq_length], tgt[:self.seq_length]
            elif len(inp) < self.seq_length:
                pad = self.seq_length - len(inp)
                inp = inp + [self.pad_token_id] * pad
                tgt = tgt + [-100] * pad
            return (torch.tensor(inp, dtype=torch.long),
                    torch.tensor(tgt, dtype=torch.long))
        except Exception as e:
            print(f"Warning: Error in __getitem__ at index {idx}: {e}")
            return (torch.tensor([self.pad_token_id] * self.seq_length, dtype=torch.long),
                    torch.tensor([-100] * self.seq_length, dtype=torch.long))