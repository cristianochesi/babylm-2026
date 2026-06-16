#!/bin/bash
#SBATCH --job-name=BabyLM2026_corpus_cleaning
#SBATCH -e "results/%x-%j.err"
#SBATCH -o "results/%x-%j.out"
#SBATCH --partition=cpuq
#SBATCH --chdir=/home/your_home
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-core=1
#SBATCH --nodes=1

date

conda init bash
source /home/your_home/.bashrc
conda activate your_env_py3_12_torch2_91_CUDA_12_8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

DATA_DIR_SRC="../data/babylm_2026_txt/nld/original/"
DATA_DIR_PROCESSED="../data/babylm_2026_txt/nld/processed/"

mkdir -p $DATA_DIR_PROCESSED

python childes.py $DATA_DIR_SRC"childes/" $DATA_DIR_PROCESSED
python subtitles.py $DATA_DIR_SRC"subtitles/" $DATA_DIR_PROCESSED
python conversations_eng.py $DATA_DIR_SRC"conversations/" $DATA_DIR_PROCESSED
python gutenberg.py $DATA_DIR_SRC"gutenberg/" $DATA_DIR_PROCESSED
python wikipedia.py $DATA_DIR_SRC"wikipedia/" $DATA_DIR_PROCESSED

cat $DATA_DIR_PROCESSED"childes.txt" $DATA_DIR_PROCESSED"subtitles.txt" $DATA_DIR_PROCESSED"conversations.txt" $DATA_DIR_PROCESSED"gutenberg.txt" $DATA_DIR_PROCESSED"wikipedia.txt" > $DATA_DIR_PROCESSED"all.txt"

python corpus_info.py $DATA_DIR_PROCESSED"all.txt"

date
echo "end of cleaning job"
