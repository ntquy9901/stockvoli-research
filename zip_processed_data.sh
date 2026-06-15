#!/bin/bash
# Script to package processed data for Google Colab upload
# This creates a zip file of all processed VN30 stock data

echo "📦 Packaging processed VN30 data for Google Colab upload..."
echo ""

# Check if processed data exists
if [ ! -d "data/processed" ]; then
    echo "❌ Error: data/processed directory not found!"
    exit 1
fi

# Count processed files
file_count=$(ls data/processed/*.csv 2>/dev/null | wc -l)
echo "✅ Found $file_count processed stock files"

if [ $file_count -lt 30 ]; then
    echo "⚠️  Warning: Expected 30 files, found $file_count"
fi

# Create zip file
echo "📦 Creating zip file: vn30_processed_data.zip"
zip -r vn30_processed_data.zip data/processed/*.csv

# Get file size
file_size=$(du -h vn30_processed_data.zip | cut -f1)
echo "✅ Zip file created: vn30_processed_data.zip ($file_size)"
echo ""

echo "📤 Upload Instructions:"
echo "1. Download vn30_processed_data.zip"
echo "2. Extract to Google Drive: My Drive/TimesFM_VN30/"
echo "3. Ensure folder structure: TimesFM_VN30/{TICKER}_processed.csv"
echo ""
echo "🚀 Ready for Google Colab testing!"

# List expected files
echo ""
echo "📋 Expected files after extraction:"
ls data/processed/*.csv | xargs -n 1 basename | sort