# Tokenizers used in the BabyLM 2026 Multilingual shared task #

This folder include the script used to build the [MorPiece](https://github.com/cristianochesi/morpiece) (MoP) and the ParadigmFinder (PaF) tokenizers used for the challenge.

Below a sample batch script to train MoP (pull the MorPiece git into ./morpiece/ folder):

```
# Hyperparameters
SIZE="16k"
CUTOFF="100"
MIN_FREQ="10"
MIN_SUFF_STEM="3"

BENCHMARK_EN="./benchmarks/sigmorphon2022_eng_adapted.csv"
BENCHMARK_NL="./benchmarks/sigmorphon2022_nld_adapted.csv"
BENCHMARK_CN="./benchmarks/sigmorphon2022_zho_adapted.csv"

DATA="../data/multilingual_selection"
ENG="$DATA/eng"
NLD="$DATA/nld"
ZHO="$DATA/zho"
VOCAB_SIZE=16640
# VOCAB_SIZE=16384
# VOCAB_SIZE=32768
VOCAB_PATH="./tokenizer/MoP_16K_ENG_NLD_ZHO_native_glue_fix"

srun python -u morpiece/build_multilingual_morpiece.py \
     --mode soft --budget $VOCAB_SIZE \
     --hf_pipeline native_bf \
     --glue_morphemes \
     --lang eng:$ENG \
     --lang nld:$NLD \
     --lang zho:$ZHO \
     --zho_min_frequency 5 --char_coverage_topk 5000 \
     --cutoff $CUTOFF --min_frequency $MIN_FREQ --min_suffix_stems $MIN_SUFF_STEM \
     --output_dir $VOCAB_PATH \
     # --hf_pipeline unigram --unigram_cluster_power 2.0 \

srun python test_tokenizer.py --tokenizer_folder $VOCAB_PATH --benchmark_file $BENCHMARK_EN
srun python test_tokenizer.py --tokenizer_folder $VOCAB_PATH --benchmark_file $BENCHMARK_NL
srun python test_tokenizer.py --tokenizer_folder $VOCAB_PATH --benchmark_file $BENCHMARK_CN
```
