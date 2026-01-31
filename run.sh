#!/bin/bash

echo "🔥 Macro-Event Market Reaction Engine"
echo "====================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

echo "✅ pip found"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate 2>/dev/null

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
    echo ""
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create necessary directories
mkdir -p data
mkdir -p models

echo "🚀 Starting Streamlit app..."
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Run Streamlit
streamlit run app/main.py