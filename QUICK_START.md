# Quick Start: Deploy to Google Cloud Run

## Prerequisites Checklist
- [ ] Google Cloud account with billing enabled
- [ ] gcloud CLI installed and authenticated
- [ ] Project created in Google Cloud Console

## Fastest Deployment (3 Steps)

### 1. Authenticate and Set Project
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable Required APIs
```bash
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com
```

### 3. Deploy
```bash
# Windows (PowerShell)
.\deploy-to-cloud-run.ps1

# Linux/macOS
chmod +x deploy-to-cloud-run.sh
./deploy-to-cloud-run.sh

# Or manually:
gcloud run deploy calculator-mcp-server \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2
```

## That's It! 🎉

Your service will be available at: `https://calculator-mcp-server-XXXXX.run.app`

## Next Steps

1. **Set Environment Variables** (if needed):
```bash
gcloud run services update calculator-mcp-server \
  --region us-central1 \
  --update-env-vars "SUPABASE_URL=your-url,SUPABASE_KEY=your-key"
```

2. **Test Your Deployment**:
```bash
# Get your service URL
SERVICE_URL=$(gcloud run services describe calculator-mcp-server --region us-central1 --format 'value(status.url)')

# Test health endpoint
curl $SERVICE_URL/api/health
```

3. **View Logs**:
```bash
gcloud run services logs read calculator-mcp-server --region us-central1
```

## Need More Details?

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for comprehensive instructions.

