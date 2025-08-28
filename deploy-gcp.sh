# Google Cloud Run deployment script

# 1. Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/aitrader-ui

# 2. Deploy to Cloud Run
gcloud run deploy aitrader-ui \
    --image gcr.io/YOUR_PROJECT_ID/aitrader-ui \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --port 8080 \
    --max-instances 10
