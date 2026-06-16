### Preprocess Folder

This folder contains simple scripts to preprocess the raw BabyLM 2026 data.

1. extract `.txt` files and clean them using the [baseline BabyLM 2026 scripts](https://github.com/babylm-org/babylm-baselines/tree/main/strict-gpt2/experiment_scripts).
2. place the raw txt data obtained in `../data/<lang>/` by distributing them into the appropriate sub-directory (each sub-directory can contain one or more simple-text, UTF-8, files):
`childes`, `subtitles`, `conversations` (this folder can include bnc_spoken corpus), `gutenberg` and `wikipedia`.
3. Distribute them into the appropriate sub-folder (each sub-folder can contain one or more simple-text, UTF-8, files):
`childes`, `subtitles`, `conversations` (this folder can include bnc_spoken corpus), `gutenberg` (this folder can contatin any book/textual/educational file) and `wikipedia`.
4. Set the correct folders in the `clean.sbatch` script and run it (e.g., ```DATA_DIR_SRC="../data/babylm_2026_txt/nld/original/"```, ```DATA_DIR_PROCESSED="../data/babylm_2026_txt/nld/processed/"```)
```bash
./clean.sbatch
```
5. (Optional) To create a multilingual corpus, run the following command (remember to set the correct DATA folder; You can also specify the amunt of child-directed-speech to be included and a max % of each type of text to add to the mixed training corpus you want to create)
```bash
./build_multilingual.sbatch
```

Required python packages (to be installed before running the script):

 - `smart-open`
 - `normalize`
 - `ftfy`
