#!/bin/bash
# Full training run (38K samples, 3 epochs, ~2-4 hours)

set -e

cd "$(dirname "$0")/../LLaMA-Factory"

echo "========================================="
echo "Full Training Run"
echo "========================================="
echo "Model: Qwen2.5-7B-Instruct"
echo "Samples: 38,645"
echo "Epochs: 3"
echo "Expected time: ~2-4 hours"
echo "========================================="
echo ""
echo "Starting training..."
echo "Monitor progress: tail -f ../models/qwen2.5-7b-lora-variant7/trainer_log.jsonl"
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
    --output_dir ../models/qwen2.5-7b-lora-variant7 \
    --overwrite_cache true \
    --overwrite_output_dir true \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_ratio 0.1 \
    --save_steps 500 \
    --eval_steps 500 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --val_size 0.1 \
    --per_device_eval_batch_size 4 \
    --eval_strategy steps \
    --plot_loss true \
    --bf16 true

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo "Model saved to: variant7/models/qwen2.5-7b-lora-variant7"
echo ""
echo "Next steps:"
echo "  1. Test inference: python scripts/test_inference.py"
echo "  2. Evaluate on test set: python scripts/evaluate_variant7.py"
echo ""
