import sys
import os
import utils
from smart_open import open

file = sys.argv[1]

print("\n===CLEANED CORPUS info===\nOpening file '" + file + "':")



with open(file, encoding='utf8') as f:
    text_original = f.read()

types, tokens, sentences, percentile, min, max = utils.get_corpus_info(text_original)
print("\nTypes: " + str(types) + "\nTokens: " + str(tokens) + "\nTTR: " + str(round(types / tokens, 2)) + "\nSentences (# of lines): " + str(sentences) + "\nWord per sentence: " + str(round(tokens/sentences,0)) + "\n85% of sentences captured with length (min=" + str(min) + " max=" + str(max) +"): " + str(percentile))