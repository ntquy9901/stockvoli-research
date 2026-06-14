#!/bin/bash

# 🚀 Prepare Colab Upload Script
# This script creates a ZIP file with all necessary files for Google Colab

echo "🔧 Preparing files for Google Colab upload..."

# Create temporary directory structure
mkdir -p /tmp/colab_upload/configs
mkdir -p /tmp/colab_upload/data/processed
mkdir -p /tmp/colab_upload/src
mkdir -p /tmp/colab_upload/experiments

# Copy configuration files
echo "📋 Copying configuration files..."
cp configs/config.yaml /tmp/colab_upload/configs/

# Copy processed data files
echo "📊 Copying processed data files (30 stocks)..."
cp data/processed/*_processed.csv /tmp/colab_upload/data/processed/

# Copy source code
echo "💻 Copying source code..."
cp src/model_training_fixed.py /tmp/colab_upload/src/

# Copy the Colab notebook
echo "📓 Copying Colab notebook..."
cp colab_timesfm_training.ipynb /tmp/colab_upload/

# Create README
cat > /tmp/colab_upload/README.md << 'EOF'
# TimesFM Training on Google Colab

This package contains everything needed to train TimesFM 2.5 on Google Colab.

## Quick Start:
1. Upload this ZIP to Google Drive
2. Extract in Google Drive (should create stockvoli-research folder)
3. Open colab_timesfm_training.ipynb in Colab
4. Run: Runtime → Run all
5. Wait ~3 hours for training completion

## Files Included:
- configs/config.yaml - Training configuration
- data/processed/*.csv - 30 VN30 stocks processed data
- src/model_training_fixed.py - Training script
- colab_timesfm_training.ipynb - Colab notebook

## Expected Results:
- Training time: ~3 hours on T4 GPU
- Final R²: ~0.5-0.6
- QLIKE: < 1.0

Good luck! 🚀
EOF

# Create ZIP file
echo "📦 Creating ZIP file..."
cd /tmp
zip -r "stockvoli-colab-ready.zip" colab_upload/
cd - > /dev/null

# Move to project directory
mv /tmp/stockvoli-colab-ready.zip D:/bmad-projects/stockvoli-research/

# Cleanup
rm -rf /tmp/colab_upload

echo "✅ Setup complete!"
echo ""
echo "📦 Created: stockvoli-colab-ready.zip"
echo ""
echo "📊 Package contents:"
unzip -l stockvoli-colab-ready.zip | tail -20
echo ""
echo "🚀 Next steps:"
echo "1. Upload 'stockvoli-colab-ready.zip' to Google Drive"
echo "2. Open colab.google.com"
echo "3. Upload 'colab_timesfm_training.ipynb' (from the ZIP)"
echo "4. Run: Runtime → Run all"
echo ""
echo "🎯 Your training will start automatically!"
echo "   Expected completion time: ~3 hours on T4 GPU"
