# Google Cloud Run Deployment Guide

This guide will walk you through deploying your calculator-mcp-server application to Google Cloud Run.

## Prerequisites

1. **Google Cloud Account**: Sign up at [cloud.google.com](https://cloud.google.com)
2. **Google Cloud SDK (gcloud)**: Install from [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install)
3. **Docker**: Install from [docker.com](https://www.docker.com/get-started) (optional, for local testing)
4. **Project Access**: You need a Google Cloud project with billing enabled

## Step 1: Install and Configure Google Cloud SDK

### Install gcloud CLI

**Windows (PowerShell):**
```powershell
# Download and run the installer from:
# https://cloud.google.com/sdk/docs/install
```

**macOS:**
```bash
# Using Homebrew
brew install --cask google-cloud-sdk

# Or download from:
# https://cloud.google.com/sdk/docs/install
```

**Linux:**
```bash
# Download and run the installer
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### Authenticate and Initialize

```bash
# Login to your Google account
gcloud auth login

# Set your project (replace PROJECT_ID with your actual project ID)
gcloud config set project PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## Step 2: Configure Environment Variables

Create a file to store your environment variables. You'll need to set these in Cloud Run:

**Required (if using Supabase):**
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase service role key

**Optional:**
- `ADMIN_USERNAME`: Admin username (defaults to "admin")
- `ADMIN_PASSWORD`: Admin password (defaults to "changeme123")
- `OPENAI_API_KEY`: OpenAI API key (if using OpenAI features)

## Step 3: Build and Deploy

### Option A: Deploy from Source (Recommended)

This method builds the container in the cloud:

```bash
# Navigate to your project directory
cd D:\Data\dev\calculator-mcp-server

# Deploy to Cloud Run
gcloud run deploy calculator-mcp-server \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --set-env-vars "SUPABASE_URL=your-supabase-url,SUPABASE_KEY=your-supabase-key"
```

**Explanation of flags:**
- `--source .`: Build from current directory
- `--platform managed`: Use fully managed Cloud Run
- `--region us-central1`: Choose your preferred region
- `--allow-unauthenticated`: Allow public access (remove for private)
- `--port 8080`: Port your app listens on
- `--memory 2Gi`: Memory allocation (adjust based on needs)
- `--cpu 2`: CPU allocation
- `--timeout 300`: Request timeout in seconds
- `--max-instances 10`: Maximum concurrent instances

### Option B: Build Locally and Push

If you prefer to build locally first:

```bash
# Set your project ID
export PROJECT_ID=your-project-id

# Build the container
gcloud builds submit --tag gcr.io/$PROJECT_ID/calculator-mcp-server

# Deploy the container
gcloud run deploy calculator-mcp-server \
  --image gcr.io/$PROJECT_ID/calculator-mcp-server \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2
```

## Step 4: Set Environment Variables (After Initial Deployment)

If you didn't set environment variables during deployment, set them now:

```bash
gcloud run services update calculator-mcp-server \
  --region us-central1 \
  --update-env-vars "SUPABASE_URL=your-supabase-url,SUPABASE_KEY=your-supabase-key,ADMIN_USERNAME=admin,ADMIN_PASSWORD=your-secure-password"
```

Or set them individually:

```bash
gcloud run services update calculator-mcp-server \
  --region us-central1 \
  --set-env-vars SUPABASE_URL=your-supabase-url \
  --set-env-vars SUPABASE_KEY=your-supabase-key \
  --set-env-vars ADMIN_USERNAME=admin \
  --set-env-vars ADMIN_PASSWORD=your-secure-password
```

## Step 5: Configure CORS (If Needed)

Your app already has CORS configured in `app/main.py`. If you need to add your Cloud Run URL:

1. Get your Cloud Run service URL:
```bash
gcloud run services describe calculator-mcp-server --region us-central1 --format 'value(status.url)'
```

2. Update `app/main.py` to include your Cloud Run URL in the `origins` list:
```python
origins = [
    "http://localhost:5174",
    "http://localhost:5173",
    "https://risolutorematematico.it",
    "https://www.risolutorematematico.it",
    "https://your-service-url.run.app",  # Add your Cloud Run URL
]
```

3. Redeploy:
```bash
gcloud run deploy calculator-mcp-server --source . --region us-central1
```

## Step 6: Verify Deployment

1. **Get your service URL:**
```bash
gcloud run services describe calculator-mcp-server \
  --region us-central1 \
  --format 'value(status.url)'
```

2. **Test the health endpoint:**
```bash
curl https://your-service-url.run.app/api/health
```

3. **View logs:**
```bash
gcloud run services logs read calculator-mcp-server --region us-central1
```

## Step 7: Set Up Custom Domain (Optional)

1. **Map a custom domain:**
```bash
gcloud run domain-mappings create \
  --service calculator-mcp-server \
  --domain your-domain.com \
  --region us-central1
```

2. **Follow the DNS configuration instructions** provided by the command output.

## Step 8: Continuous Deployment (Optional)

### Using Cloud Build Triggers

1. **Create a trigger:**
```bash
gcloud builds triggers create github \
  --name="deploy-calculator-server" \
  --repo-name="your-repo-name" \
  --repo-owner="your-github-username" \
  --branch-pattern="^main$" \
  --build-config="cloudbuild.yaml"
```

2. **Create `cloudbuild.yaml` in your project root:**
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/calculator-mcp-server', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/calculator-mcp-server']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'calculator-mcp-server'
      - '--image'
      - 'gcr.io/$PROJECT_ID/calculator-mcp-server'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
```

## Troubleshooting

### Container fails to start
- Check logs: `gcloud run services logs read calculator-mcp-server --region us-central1`
- Verify PORT environment variable is set correctly
- Ensure all dependencies are in `requirements.txt`

### Out of memory errors
- Increase memory: `gcloud run services update calculator-mcp-server --memory 4Gi --region us-central1`

### Timeout errors
- Increase timeout: `gcloud run services update calculator-mcp-server --timeout 600 --region us-central1`

### CORS errors
- Verify your frontend URL is in the CORS origins list
- Check that CORS middleware is properly configured

### Environment variables not working
- Verify variables are set: `gcloud run services describe calculator-mcp-server --region us-central1`
- Check for typos in variable names
- Ensure no extra spaces in values

## Cost Optimization

1. **Set minimum instances to 0** (scales to zero when not in use):
```bash
gcloud run services update calculator-mcp-server \
  --min-instances 0 \
  --region us-central1
```

2. **Set maximum instances** to control costs:
```bash
gcloud run services update calculator-mcp-server \
  --max-instances 5 \
  --region us-central1
```

3. **Use appropriate memory/CPU** for your workload:
```bash
gcloud run services update calculator-mcp-server \
  --memory 1Gi \
  --cpu 1 \
  --region us-central1
```

## Security Best Practices

1. **Use Secret Manager for sensitive data:**
```bash
# Create a secret
echo -n "your-secret-value" | gcloud secrets create supabase-key --data-file=-

# Grant Cloud Run access
gcloud secrets add-iam-policy-binding supabase-key \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Use in deployment
gcloud run deploy calculator-mcp-server \
  --update-secrets SUPABASE_KEY=supabase-key:latest \
  --region us-central1
```

2. **Require authentication** (remove `--allow-unauthenticated`):
```bash
gcloud run services update calculator-mcp-server \
  --no-allow-unauthenticated \
  --region us-central1
```

3. **Set up IAM policies** to control who can invoke the service.

## Quick Reference Commands

```bash
# Deploy
gcloud run deploy calculator-mcp-server --source . --region us-central1

# Update environment variables
gcloud run services update calculator-mcp-server \
  --update-env-vars "KEY=value" \
  --region us-central1

# View logs
gcloud run services logs read calculator-mcp-server --region us-central1 --limit 50

# Get service URL
gcloud run services describe calculator-mcp-server --region us-central1 --format 'value(status.url)'

# Delete service
gcloud run services delete calculator-mcp-server --region us-central1
```

## Next Steps

1. Set up monitoring and alerts in Cloud Console
2. Configure custom domain if needed
3. Set up CI/CD pipeline for automatic deployments
4. Review and optimize resource allocation based on usage
5. Set up backup and disaster recovery procedures

