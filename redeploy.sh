#!/bin/bash
# Redeploy AI Trader after git push

echo "🔄 AI Trader - Redeploy Helper"
echo "============================="

echo "✅ Code pushed to GitHub successfully!"
echo ""

# Check which platforms might be deployed
echo "🚀 Redeployment options:"
echo ""

# Streamlit Cloud
echo "1️⃣  STREAMLIT CLOUD:"
echo "   • Should auto-deploy in 1-2 minutes"
echo "   • Check: https://share.streamlit.io/deploy"
echo "   • Manual reboot: Go to app dashboard → 'Reboot'"
echo ""

# Railway
if command -v railway &> /dev/null; then
    echo "2️⃣  RAILWAY:"
    echo "   • Auto-deploys on push"
    echo "   • Manual: railway redeploy"
    echo "   • Status: railway status"
    echo ""
fi

# Render
echo "3️⃣  RENDER:"
echo "   • Auto-deploys on push"
echo "   • Manual: Go to render.com dashboard → 'Manual Deploy'"
echo ""

# Heroku
if command -v heroku &> /dev/null; then
    echo "4️⃣  HEROKU:"
    echo "   • Manual: heroku ps:restart -a your-app-name"
    echo ""
fi

echo "🔗 Check deployment status:"
echo "   • Streamlit Cloud: https://share.streamlit.io"
echo "   • Railway: railway logs"
echo "   • Render: Check dashboard"
echo ""

echo "⏱️  Typical deployment times:"
echo "   • Streamlit Cloud: 30-60 seconds"
echo "   • Railway: 1-2 minutes"
echo "   • Render: 2-3 minutes"
