#!/bin/bash

# 🌾 Crop Recommendation System Launcher
echo "🌾 Starting Crop Recommendation System..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if dataset exists
if [ ! -f "Crop_recommendation.csv.xls" ]; then
    echo "❌ Error: Dataset file 'Crop_recommendation.csv.xls' not found!"
    echo "Please ensure the dataset is in the current directory."
    exit 1
fi

# Start Streamlit app
echo "🚀 Launching Streamlit application..."
echo "📱 Your app will open in the browser automatically"
echo "🔗 Or visit: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the application"
echo "=================================="

streamlit run streamlit_app.py