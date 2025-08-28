#!/bin/bash

# AI Trader - Production Deployment Script
# Usage: ./deploy.sh [platform]
# Platforms: streamlit, railway, render, heroku

set -e  # Exit on error

PLATFORM=${1:-streamlit}

echo "🚀 AI Trader - Production Deployment"
echo "Target Platform: $PLATFORM"
echo "=================================="

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."
    
    if [ ! -f "ui.py" ]; then
        echo "❌ ui.py not found"
        exit 1
    fi
    
    if [ ! -f "requirements.txt" ]; then
        echo "❌ requirements.txt not found"
        exit 1
    fi
    
    if [ ! -f "issi.json" ]; then
        echo "⚠️  issi.json not found, creating sample..."
        cat > issi.json << EOF
[
  {
    "symbol": "BBCA.JK",
    "name": "Bank Central Asia",
    "ema_short": 20,
    "ema_long": 50,
    "rsi": 14
  }
]
EOF
    fi
    
    echo "✅ Prerequisites check completed"
}

# Git setup and push
setup_git() {
    echo "📝 Setting up Git repository..."
    
    if [ ! -d ".git" ]; then
        git init
        echo "✅ Git repository initialized"
    fi
    
    # Add all files
    git add .
    git commit -m "Deploy AI Trader UI - $(date)" || echo "No changes to commit"
    
    # Check if remote exists
    if ! git remote get-url origin > /dev/null 2>&1; then
        echo "⚠️  No git remote found. Please add your repository:"
        echo "   git remote add origin https://github.com/yourusername/aitrader.git"
        echo "   git push -u origin main"
        exit 1
    fi
    
    git push origin main
    echo "✅ Code pushed to repository"
}

# Deploy to Streamlit Cloud
deploy_streamlit() {
    echo "🌐 Deploying to Streamlit Cloud..."
    echo "📋 Manual steps required:"
    echo "1. Visit https://share.streamlit.io"
    echo "2. Click 'New app'"
    echo "3. Connect your GitHub account"
    echo "4. Select your repository"
    echo "5. Main file path: ui.py"
    echo "6. Click 'Deploy!'"
    echo ""
    echo "🔗 Your app will be available at:"
    echo "   https://yourusername-aitrader-ui-xxxxxx.streamlit.app"
}

# Deploy to Railway
deploy_railway() {
    echo "🚂 Deploying to Railway..."
    
    # Check if Railway CLI is installed
    if ! command -v railway &> /dev/null; then
        echo "Installing Railway CLI..."
        npm install -g @railway/cli
    fi
    
    # Login and deploy
    railway login
    railway init
    railway up
    
    echo "✅ Deployed to Railway!"
    echo "🔗 Check your dashboard at https://railway.app"
}

# Deploy to Render
deploy_render() {
    echo "🎨 Deploying to Render..."
    echo "📋 Manual steps required:"
    echo "1. Visit render.com"
    echo "2. Click 'New +' → 'Web Service'"
    echo "3. Connect your GitHub repository"
    echo "4. Configure:"
    echo "   - Name: aitrader-ui"
    echo "   - Environment: Python 3"
    echo "   - Build Command: pip install -r requirements.txt"
    echo "   - Start Command: streamlit run ui.py --server.headless true --server.port \$PORT --server.address 0.0.0.0"
    echo "5. Add environment variables if needed"
    echo "6. Click 'Create Web Service'"
}

# Deploy to Heroku
deploy_heroku() {
    echo "🟣 Deploying to Heroku..."
    
    # Check if Heroku CLI is installed
    if ! command -v heroku &> /dev/null; then
        echo "❌ Heroku CLI not found. Install from https://devcenter.heroku.com/articles/heroku-cli"
        exit 1
    fi
    
    # Login and create app
    heroku login
    heroku create aitrader-ui-$(date +%s) || echo "App might already exist"
    
    # Deploy
    git push heroku main
    
    echo "✅ Deployed to Heroku!"
    heroku open
}

# Security check
security_check() {
    echo "🔐 Running security checks..."
    
    # Check for hardcoded secrets
    if grep -r "gsk_" *.py > /dev/null 2>&1; then
        echo "⚠️  Warning: Potential API keys found in code"
        echo "   Consider using environment variables"
    fi
    
    # Check for debug mode
    if grep -r "debug.*=.*True" *.py > /dev/null 2>&1; then
        echo "⚠️  Warning: Debug mode might be enabled"
    fi
    
    echo "✅ Security check completed"
}

# Main deployment flow
main() {
    echo "Starting deployment process..."
    
    check_prerequisites
    security_check
    setup_git
    
    case $PLATFORM in
        "streamlit")
            deploy_streamlit
            ;;
        "railway")
            deploy_railway
            ;;
        "render")
            deploy_render
            ;;
        "heroku")
            deploy_heroku
            ;;
        *)
            echo "❌ Unknown platform: $PLATFORM"
            echo "Available platforms: streamlit, railway, render, heroku"
            exit 1
            ;;
    esac
    
    echo ""
    echo "🎉 Deployment process completed!"
    echo "📚 Check DEPLOYMENT_GUIDE.md for detailed instructions"
    echo "🔧 Check PRODUCTION_GUIDE.md for optimization tips"
}

# Help message
show_help() {
    echo "AI Trader Deployment Script"
    echo ""
    echo "Usage: $0 [platform]"
    echo ""
    echo "Platforms:"
    echo "  streamlit  Deploy to Streamlit Cloud (default, easiest)"
    echo "  railway    Deploy to Railway (great performance)"
    echo "  render     Deploy to Render (reliable)"
    echo "  heroku     Deploy to Heroku (classic)"
    echo ""
    echo "Examples:"
    echo "  $0                  # Deploy to Streamlit Cloud"
    echo "  $0 railway         # Deploy to Railway"
    echo "  $0 render          # Deploy to Render"
}

# Handle arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# Run main deployment
main
