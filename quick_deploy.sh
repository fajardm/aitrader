#!/bin/bash
# AI Trader - Simple Deploy Setup

echo "🚀 Quick Deploy Setup for AI Trader"
echo "==================================="

# 1. Setup git if needed
if [ ! -d ".git" ]; then
    git init
    git branch -M main
fi

# 2. Create .gitignore
cat > .gitignore << 'EOF'
.streamlit/secrets.toml
.env
__pycache__/
*.pyc
.venv/
.DS_Store
EOF

# 3. Add and commit
git add .
git commit -m "AI Trader UI ready for deployment" || echo "Already committed"

echo ""
echo "✅ READY TO DEPLOY!"
echo ""
echo "🎯 RECOMMENDED: Streamlit Cloud (100% Free)"
echo "1. Push to GitHub: git remote add origin YOUR_GITHUB_URL"
echo "2. git push -u origin main"
echo "3. Go to https://share.streamlit.io"
echo "4. Connect your repo → Deploy!"
echo ""
echo "🔑 Add your Groq API key in Streamlit Cloud secrets:"
echo '   [groq]'
echo '   api_key = "your_key_here"'
echo ""
echo "🌐 Your app will be live at:"
echo "   https://USERNAME-aitrader-ui-main.streamlit.app"
