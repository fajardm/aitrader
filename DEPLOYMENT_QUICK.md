# Deploy AI Trader to Free Serverless Platforms

## 🚀 Quick Deploy Options

### 1. Streamlit Community Cloud (Recommended - FREE)
**Best for Streamlit apps, completely free**

1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Connect your GitHub repo
4. Deploy instantly!

**Setup:**
```bash
# Add to your repo
git add .
git commit -m "Add Streamlit UI"
git push origin main
```

### 2. Railway (FREE tier: 500 hours/month)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### 3. Render (FREE tier available)
```bash
# Just connect your GitHub repo at render.com
# Auto-deploys on git push
```

## ⚡ Fastest Option: Streamlit Cloud

1. **Create GitHub repo** (if not done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/aitrader.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Select your GitHub repo
   - Main file: `ui.py`
   - Click "Deploy!"

3. **Done!** Your app will be live at: `https://yourusername-aitrader-ui-main.streamlit.app`

## 🔧 Production Requirements

Create `.streamlit/secrets.toml` for sensitive data:
```toml
[groq]
api_key = "your_groq_api_key_here"
```

Update `ui.py` to use secrets:
```python
# Add at top of ui.py
import streamlit as st
api_key = st.secrets["groq"]["api_key"]
```

## 📝 Current Status
Your UI is ready to deploy! Just push to GitHub and use Streamlit Cloud - it's the easiest and completely free option.
