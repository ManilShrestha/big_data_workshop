#!/bin/bash
# Quick test training run (1000 samples, 1 epoch, ~10 minutes)

set -e

cd "$(dirname "$0")/../LLaMA-Factory"

echo "========================================="
echo "Quick Test Training"
echo "========================================="
echo "Model: Qwen2.5-7B-Instruct"
echo "Samples: 1000"
echo "Epochs: 1"
echo "Expected time: ~10 minutes"
echo "========================================="
echo ""

python src/train.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset variant7_all_hops \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --output_dir ../models/qwen2.5-7b-lora-test \
    --overwrite_cache true \
    --overwrite_output_dir true \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_ratio 0.1 \
    --save_steps 100 \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --max_samples 1000 \
    --plot_loss true \
    --bf16 true

echo ""
echo "========================================="
echo "Test Training Complete!"
echo "========================================="
echo "Model saved to: variant7/models/qwen2.5-7b-lora-test"
echo ""
echo "Next: Run full training with scripts/train_full.sh"
echo ""
