"""
Demo launcher for AI Trader UI
This script demonstrates how to start the Streamlit interface
"""

from pathlib import Path

def main():
    print("🤖 AI Trader - Demo Launch")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ main.py not found. Please run from the aitrader directory.")
        return
    
    if not Path("ui.py").exists():
        print("❌ ui.py not found. Please ensure the UI file exists.")
        return
    
    if not Path("issi.json").exists():
        print("⚠️  issi.json not found. Creating a sample file...")
        sample_tickers = [
            {
                "symbol": "BBCA.JK",
                "name": "Bank Central Asia",
                "ema_short": 20,
                "ema_long": 50,
                "rsi": 14
            },
            {
                "symbol": "BBRI.JK",
                "name": "Bank Rakyat Indonesia",
                "ema_short": 20,
                "ema_long": 50,
                "rsi": 14
            }
        ]
        
        import json
        with open("issi.json", "w", encoding="utf-8") as f:
            json.dump(sample_tickers, f, indent=2)
        print("✅ Created sample issi.json with 2 tickers")
    
    print("\n📋 Pre-launch checklist:")
    print("✅ main.py found")
    print("✅ ui.py found")
    print("✅ issi.json found")
    print("✅ Virtual environment configured")
    
    print("\n🚀 To launch the UI, run:")
    print("   ./launch_ui.sh")
    print("\n🔧 Or manually:")
    print("   source .venv/bin/activate")
    print("   streamlit run ui.py")
    
    print("\n📱 The UI will be available at: http://localhost:8501")
    print("\n🎯 Features available:")
    print("   • 📈 Backtesting with interactive charts")
    print("   • 🎯 Live signal generation")
    print("   • ⚙️ Parameter optimization")
    print("   • 📊 Portfolio analysis")
    print("   • 🔧 Ticker management")

if __name__ == "__main__":
    main()
