"""
Production-ready version of the UI with security enhancements
"""

import streamlit as st
import os
import json
from pathlib import Path

# Production configuration
if os.getenv('STREAMLIT_ENV') == 'production':
    # Hide Streamlit style in production
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Environment variable handling
def get_api_key():
    """Get API key from environment variable in production"""
    return os.getenv('GROQ_API_KEY', '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')

def load_tickers_safe():
    """Load tickers with error handling and fallback"""
    try:
        ticker_file = Path('./issi.json')
        if not ticker_file.exists():
            # Create default tickers if file doesn't exist
            default_tickers = [
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
            with open('./issi.json', 'w', encoding='utf-8') as f:
                json.dump(default_tickers, f, indent=2)
            
            st.info("🔄 Created default ticker configuration")
        
        # Import the main UI after ensuring config exists
        from ui import main as ui_main
        return ui_main()
        
    except Exception as e:
        st.error(f"❌ Error loading application: {str(e)}")
        st.info("Please ensure all required files are present")
        return None

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="AI Trader - Production",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Production warning
    if os.getenv('STREAMLIT_ENV') == 'production':
        st.sidebar.success("🌐 Production Environment")
    else:
        st.sidebar.info("🔧 Development Environment")
    
    # Load and run main UI
    load_tickers_safe()

if __name__ == "__main__":
    main()
