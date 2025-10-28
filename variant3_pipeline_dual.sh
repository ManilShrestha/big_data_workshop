#!/bin/bash

# Variant 3 Dual Training Pipeline
# Runs all steps sequentially to train EdgeScorerDual model

set -e  # Exit on error

echo "================================================================================"
echo "Variant 3 Dual: Complete Training Pipeline"
echo "================================================================================"
echo ""
echo "This pipeline will:"
echo "  1. Generate training data from BFS shortest paths (~674K samples)"
echo "  2. Generate text embeddings for entities and relations (~50K entities)"
echo "  3. Cache question embeddings (~unique questions)"
echo "  4. Train EdgeScorerDual model with dual embeddings"
echo ""
echo "Estimated time: ~2-3 hours"
echo "Estimated cost: ~$0.004 for OpenAI embeddings"
echo ""
echo "================================================================================"
echo ""

# Ask for confirmation
read -p "Proceed with full pipeline? (yes/no): " response
if [[ ! "$response" =~ ^[Yy]([Ee][Ss])?$ ]]; then
    echo "Aborted by user"
    exit 0
fi

echo ""
echo "================================================================================"
echo "Step 1/4: Generate Training Data"
echo "================================================================================"
echo ""
python variant3_create_training_data.py

echo ""
echo "================================================================================"
echo "Step 2/4: Generate Text Embeddings (Entities + Relations)"
echo "================================================================================"
echo ""
python variant3_generate_text_embeddings.py

echo ""
echo "================================================================================"
echo "Step 3/4: Cache Question Embeddings"
echo "================================================================================"
echo ""
python variant3_cache_embeddings_dual.py

echo ""
echo "================================================================================"
echo "Step 4/4: Train EdgeScorerDual Model"
echo "================================================================================"
echo ""
python variant3_train_edge_scorer_dual.py

echo ""
echo "================================================================================"
echo "Pipeline Complete!"
echo "================================================================================"
echo ""
echo "Trained model saved to: models/variant3_edge_scorer_dual_best.pt"
echo "Training history saved to: models/variant3_training_history_dual.json"
echo ""
echo "Next steps:"
echo "  - Evaluate model on test set"
echo "  - Integrate into QA system (variant 3 runner)"
echo "  - Compare with baseline variants"
echo ""
