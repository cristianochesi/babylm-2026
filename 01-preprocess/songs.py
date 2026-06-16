import sys
import re
import os
import utils
import ftfy
from smart_open import open

d_in = sys.argv[1]
d_out = sys.argv[2]
target_file = "songs.txt"
text_original = ""
text_cleaned = ""
text_cleaned_lines = []
text_original_lines = []
n_files = 0
utils.profiling_init(utils)
print("\n===SONGS data processing===\nOpening files in '" + d_in + "':")


def preprocess(line):
    line = re.sub(r'\b(?:TUTTI|SOLISTA|CORO):', '', line)  # rimozione occorrenze di "TUTTI:", "SOLISTA:", "CORO:"
    line = re.sub(r'\b(?:SOLISTA 1|SOLISTA 2|SOLISTI 1 E 2)\b', '', line)  # rimozione termini specifici
    line = re.sub(r'\b(?:ALEXANDROS|DARIYA|ELISA)\b', '', line)  # rimozione termini specifici
    line = re.sub(r'([aeiouàèéìòùáíóúAEIOU])\1{2,}', r'\1', line)  # sequenze di più di tre vocali vanno ridotte a una (ooo -> o)
    line = re.sub(r'[ |­]', '', line)
    line = line.lower()
    line = utils.remove_tags(line)
    line = utils.remove_quotes(line)
    line = utils.remove_symbols(line)
    line = utils.space_punctuation(line)
    # line = utils.add_full_stop(line.strip())
    line = utils.remove_multiple_spacing(line)
    if utils.is_empty_line(line):
        return ""
    else:
        return utils.remove_last_newline(line).strip() + "\n"


for file in os.listdir(d_in):
    print(".", end="")
    n_files += 1
    with open(d_in + file, encoding='utf8') as f:
        for line in f:
           text_original_lines.append(line + "\n")
           if len(line) > 0 and not utils.is_empty_line(line) and not re.search(r'\|', line):
                text_cleaned_lines.append(preprocess(line))

text_original = ''.join(text_original_lines)
text_cleaned = ''.join(text_cleaned_lines)

utils.report(n_files, text_original, text_cleaned)
utils.save(d_out, target_file, text_original, text_cleaned)
utils.profiling_end(utils)