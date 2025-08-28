# Production Deployment Guide - Free Serverless Options

## 🚀 Deployment Options

### 1. Streamlit Cloud (Recommended) ⭐
**Free tier**: Unlimited public apps, 1GB resources
**Best for**: Streamlit apps (obviously!)

### 2. Railway
**Free tier**: 500 hours/month, $5 credit monthly
**Best for**: Any Python app, great for demos

### 3. Render
**Free tier**: 750 hours/month
**Best for**: Web services, automatic SSL

### 4. Heroku (Limited Free)
**Free tier**: 1000 dyno hours/month
**Best for**: Established platform, many add-ons

### 5. Google Cloud Run
**Free tier**: 2 million requests/month
**Best for**: Containerized apps, enterprise-ready

---

## 🌐 Option 1: Streamlit Cloud (Easiest) ⭐

### Prerequisites
1. GitHub account
2. Public GitHub repository

### Step-by-Step Deployment

#### 1. Prepare Repository
```bash
# Initialize git if not done
git init
git add .
git commit -m "Initial commit - AI Trader UI"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/aitrader.git
git branch -M main
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud
1. Visit https://share.streamlit.io
2. Click "New app"
3. Connect your GitHub account
4. Select repository: `yourusername/aitrader`
5. Main file path: `ui.py`
6. Click "Deploy!"

#### 3. Environment Variables (Optional)
- Add `GROQ_API_KEY` in advanced settings
- Set `STREAMLIT_ENV=production`

### Features
- ✅ Automatic SSL certificate
- ✅ Custom domain support (.streamlit.app)
- ✅ Auto-redeploy on git push
- ✅ Built-in secrets management
- ✅ No configuration files needed

### Limitations
- Must be public repository
- 1GB RAM limit
- Community support only

---

## 🚂 Option 2: Railway (Great Performance)

### Prerequisites
1. GitHub account
2. Railway account (railway.app)

### Step-by-Step Deployment

#### 1. Prepare Files
Ensure you have:
- ✅ `requirements.txt`
- ✅ `railway.toml` (created above)
- ✅ `ui.py`

#### 2. Deploy on Railway
```bash
# Method 1: Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up

# Method 2: Web Dashboard
# 1. Visit railway.app
# 2. Connect GitHub
# 3. Select repository
# 4. Deploy automatically
```

#### 3. Configure Environment
```bash
# Set environment variables
railway variables set GROQ_API_KEY=your_api_key
railway variables set STREAMLIT_ENV=production
```

#### 4. Custom Domain (Optional)
- Go to Settings → Networking
- Add custom domain
- Update DNS records

### Features
- ✅ 500 hours/month free
- ✅ $5 credit monthly
- ✅ Custom domains
- ✅ Environment variables
- ✅ Auto-scaling
- ✅ Database add-ons

### Performance
- 1GB RAM, 1 vCPU
- Global CDN
- Automatic SSL

---

## 🎨 Option 3: Render (Reliable)

### Prerequisites
1. GitHub account
2. Render account (render.com)

### Step-by-Step Deployment

#### 1. Prepare Repository
Ensure you have:
- ✅ `requirements.txt`
- ✅ `render.yaml` (created above)

#### 2. Deploy on Render
1. Visit render.com
2. Click "New +" → "Web Service"
3. Connect GitHub repository
4. Configure:
   - **Name**: aitrader-ui
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run ui.py --server.headless true --server.port $PORT --server.address 0.0.0.0`

#### 3. Environment Variables
Add in dashboard:
```
GROQ_API_KEY=your_api_key
STREAMLIT_ENV=production
PYTHON_VERSION=3.11
```

#### 4. SSL & Domain
- Automatic SSL certificate
- Custom domain in settings
- Built-in CDN

### Features
- ✅ 750 hours/month free
- ✅ Automatic SSL
- ✅ Custom domains
- ✅ Health checks
- ✅ Zero-downtime deploys
- ✅ Built-in monitoring

### Performance
- 512MB RAM, 0.1 vCPU (free)
- Global CDN
- SSD storage

---

## ⚡ Option 4: Google Cloud Run (Scalable)

### Prerequisites
1. Google Cloud account
2. Docker knowledge

### Benefits
- Pay per request
- Auto-scaling
- Enterprise features
- Global deployment

### Configuration Files Needed ⬇️

---

## 📋 Quick Comparison

| Platform | Ease | Performance | Limits | SSL | Custom Domain |
|----------|------|-------------|--------|-----|---------------|
| Streamlit Cloud | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Public only | ✅ | ✅ |
| Railway | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 500h/month | ✅ | ✅ |
| Render | ⭐⭐⭐⭐ | ⭐⭐⭐ | 750h/month | ✅ | ✅ |
| Google Cloud Run | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 2M req/month | ✅ | ✅ |

## 🎯 Recommendation

**For beginners**: Use Streamlit Cloud
**For production**: Use Railway or Google Cloud Run
**For reliability**: Use Render
