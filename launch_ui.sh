#!/bin/bash
# AI Trader UI Launcher
# This script activates the virtual environment and launches the Streamlit UI

echo "🤖 AI Trader - Starting UI..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    echo "💡 Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if streamlit is available in venv
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Launch the UI
echo "🚀 Launching AI Trader UI..."
echo "📱 The app will open in your default browser"
echo "🔗 If it doesn't open automatically, visit: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

streamlit run ui.py
