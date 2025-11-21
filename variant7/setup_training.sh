#!/bin/bash
# Setup script for Variant7 LoRA fine-tuning

set -e  # Exit on error

echo "========================================="
echo "Variant7 Training Setup"
echo "========================================="

# Check if we're in the right directory
if [ ! -f "extract_lora_training_data.py" ]; then
    echo "Error: Must run from variant7/ directory"
    exit 1
fi

# Step 1: Clone LLaMA-Factory
echo ""
echo "[1/4] Cloning LLaMA-Factory..."
if [ -d "LLaMA-Factory" ]; then
    echo "  LLaMA-Factory already exists, skipping..."
else
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    echo "  ✓ Cloned successfully"
fi

# Step 2: Install dependencies
echo ""
echo "[2/4] Installing dependencies..."
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --quiet
echo "  ✓ Dependencies installed"

# Step 3: Verify installation
echo ""
echo "[3/4] Verifying installation..."
python -c "import transformers; import peft; print(f'  ✓ Transformers: {transformers.__version__}'); print(f'  ✓ PEFT: {peft.__version__}')"

# Step 4: Register dataset
echo ""
echo "[4/4] Registering training dataset..."

# Check if dataset already registered
if grep -q "variant7_all_hops" data/dataset_info.json; then
    echo "  Dataset already registered, skipping..."
else
    # Backup original
    cp data/dataset_info.json data/dataset_info.json.bak

    # Add our dataset using Python
    python << 'EOF'
import json

with open('data/dataset_info.json', 'r') as f:
    data = json.load(f)

# Add variant7 dataset
data['variant7_all_hops'] = {
    "file_name": "../../training_data/all_hops_test.jsonl",
    "formatting": "sharegpt",
    "columns": {
        "messages": "messages"
    }
}

with open('data/dataset_info.json', 'w') as f:
    json.dump(data, f, indent=2)

print("  ✓ Dataset registered")
EOF
fi

cd ..

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Quick test (10 min):  bash scripts/train_quick_test.sh"
echo "  2. Full training (2-4h): bash scripts/train_full.sh"
echo "  3. Monitor training:     tail -f models/qwen2.5-7b-lora-*/trainer_log.jsonl"
echo ""
