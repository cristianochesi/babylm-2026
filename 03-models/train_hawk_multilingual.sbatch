#!/bin/bash
#SBATCH --job-name=ML_HAWK
#SBATCH -e "results/%x-%j_err.txt"
#SBATCH -o "results/%x-%j_out.txt"
#SBATCH --partition=gpuq
#SBATCH --nodelist=gp0x
#SBATCH --account=xxx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --chdir=/home/your_home

date
source /home/your_home/.bashrc
conda activate env_py3_12_torch2_91_CUDA_12_8

TOK=/home/your_home/your_tokenizer/tokenizer.json
ROOT=/home/your_home/your_babylm_folder/data/multilingual_selection

python train_multilingual.py \
    --arch hawk \
    --tokenizer "$TOK" --corpus_root "$ROOT" \
    --regimen conversational --seq_length 512 \
    --n_embd 704 --n_layer 12 --rnn_width 768 --conv_kernel 4 --mlp_expansion 3 --dropout 0.1 \
    --max_position_embeddings 1024 \
    --batch_size 16 --accumulation_steps 4 \
    --learning_rate 5e-4 --min_lr 5e-5 --warmup_ratio 0.01 \
    --weight_decay 0.1 --beta1 0.9 --beta2 0.999 --grad_clip 1.0 \
    --use_amp --amp_dtype bf16 \
    --max_tokens_exposure 0 --epochs 5 \
    --save_at_words babylm --budget_millions 1000 \
    --output_dir checkpoints/ml_hawk --log_interval 50
date; echo "end Hawk-multilingual training"
