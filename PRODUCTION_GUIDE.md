# Production Security & Best Practices

## 🔐 Security Checklist

### Environment Variables
```bash
# Set these in your deployment platform
GROQ_API_KEY=your_actual_api_key_here
STREAMLIT_ENV=production
ALLOWED_HOSTS=your-domain.com
```

### API Key Security
❌ **DON'T**: Hard-code API keys in source code
✅ **DO**: Use environment variables

```python
# In your code
import os
api_key = os.getenv('GROQ_API_KEY')
```

### Production Configuration
Create `.streamlit/secrets.toml` for sensitive data:
```toml
# This file is NOT committed to git
[general]
api_key = "your_secret_key"
database_url = "your_db_connection"
```

## 📊 Performance Optimization

### Memory Management
```python
# Add to ui.py
import gc
import streamlit as st

# Clear cache periodically
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    # Your data loading logic
    pass

# Manual garbage collection for long processes
def cleanup_memory():
    gc.collect()
```

### Caching Strategy
```python
# Cache expensive operations
@st.cache_data(ttl=1800)  # 30 minutes
def load_ticker_data(symbol):
    return yf.download(symbol)

@st.cache_resource
def load_model():
    # Load ML models once
    return model
```

### Database Considerations
For production, consider adding:
- PostgreSQL for data storage
- Redis for caching
- Time-series database for market data

## 🚀 Deployment Security

### 1. Environment Variables
```bash
# Railway
railway variables set GROQ_API_KEY=xxx

# Render
# Set in dashboard under Environment

# Google Cloud Run
gcloud run services update aitrader-ui \
  --set-env-vars GROQ_API_KEY=xxx
```

### 2. HTTPS Only
All platforms provide automatic HTTPS:
- Streamlit Cloud: ✅ Auto
- Railway: ✅ Auto  
- Render: ✅ Auto
- Google Cloud Run: ✅ Auto

### 3. Access Control
```python
# Add authentication if needed
def check_password():
    def password_entered():
        if st.session_state["password"] == os.getenv("APP_PASSWORD"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", 
                     on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", 
                     on_change=password_entered, key="password")
        st.error("Password incorrect")
        return False
    else:
        return True
```

## 📈 Monitoring & Logging

### Application Metrics
```python
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track performance
def track_performance(func_name):
    start_time = time.time()
    # Your function here
    execution_time = time.time() - start_time
    logger.info(f"{func_name} took {execution_time:.2f} seconds")
```

### Health Checks
```python
# Add to ui.py
import streamlit as st
import requests

def health_check():
    """Simple health check endpoint"""
    try:
        # Test database connection
        # Test API connectivity
        return {"status": "healthy", "timestamp": time.time()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## 🔧 Production Optimizations

### 1. Reduce Memory Usage
```python
# Use generators instead of lists
def process_large_dataset():
    for chunk in pd.read_csv('large_file.csv', chunksize=1000):
        yield process_chunk(chunk)
```

### 2. Lazy Loading
```python
# Load components only when needed
@st.cache_resource
def get_optimization_engine():
    import optuna  # Import only when needed
    return optuna.create_study()
```

### 3. Error Boundaries
```python
def safe_execute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.exception("Error in safe_execute")
        return None
```

## 📱 Mobile Optimization

### Responsive Design
```python
# Detect mobile devices
import streamlit as st

# Mobile-friendly layouts
col1, col2 = st.columns([1, 1] if not is_mobile() else [1])

def is_mobile():
    # Detect mobile user agents (basic)
    return st.session_state.get('mobile', False)
```

### Touch-Friendly UI
```python
# Larger buttons for mobile
if is_mobile():
    button_style = """
    <style>
    .stButton button {
        height: 50px;
        font-size: 18px;
    }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
```

## 💾 Backup & Recovery

### Data Backup
```python
# Automated backups
import schedule
import time

def backup_config():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    shutil.copy('issi.json', f'backups/issi_{timestamp}.json')

# Schedule daily backups
schedule.every().day.at("02:00").do(backup_config)
```

### Configuration Management
```python
# Version control for configurations
import json
import hashlib

def save_config_with_version(config):
    config_str = json.dumps(config, sort_keys=True)
    version_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    with open(f'configs/config_{version_hash}.json', 'w') as f:
        json.dump(config, f, indent=2)
```

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Railway
      run: |
        npm install -g @railway/cli
        railway deploy
      env:
        RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
```

## 📊 Cost Optimization

### Free Tier Limits
- **Streamlit Cloud**: Unlimited apps, 1GB RAM
- **Railway**: 500 hours/month, $5 credit
- **Render**: 750 hours/month
- **Google Cloud Run**: 2M requests/month

### Usage Monitoring
```python
# Track resource usage
import psutil
import streamlit as st

def display_system_metrics():
    if st.sidebar.checkbox("Show System Metrics"):
        st.sidebar.metric("CPU Usage", f"{psutil.cpu_percent()}%")
        st.sidebar.metric("Memory Usage", 
                         f"{psutil.virtual_memory().percent}%")
```

## 🎯 Deployment Checklist

Before going live:
- [ ] API keys in environment variables
- [ ] Remove debug/test code
- [ ] Enable caching for expensive operations
- [ ] Test with sample data
- [ ] Verify mobile responsiveness
- [ ] Setup error logging
- [ ] Configure custom domain (if needed)
- [ ] Test all major features
- [ ] Setup monitoring/alerts
- [ ] Document deployment process

## 🆘 Troubleshooting

### Common Issues
1. **Memory errors**: Reduce data size, add caching
2. **Slow loading**: Optimize data loading, add progress bars
3. **API timeouts**: Add retry logic, error handling
4. **Mobile issues**: Test responsive design

### Debug Mode
```python
# Add debug panel for development
if os.getenv('DEBUG', 'False').lower() == 'true':
    st.sidebar.subheader("🐛 Debug Info")
    st.sidebar.json(st.session_state)
```
