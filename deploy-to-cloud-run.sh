#!/bin/bash

# Google Cloud Run Deployment Script
# This script automates the deployment process

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="calculator-mcp-server"
REGION="us-central1"
PORT="8080"
MEMORY="2Gi"
CPU="2"
TIMEOUT="300"
MAX_INSTANCES="10"

echo -e "${GREEN}=== Google Cloud Run Deployment Script ===${NC}\n"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed.${NC}"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}You are not authenticated. Please run: gcloud auth login${NC}"
    exit 1
fi

# Get current project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: No project set. Please run: gcloud config set project PROJECT_ID${NC}"
    exit 1
fi

echo -e "Project ID: ${GREEN}$PROJECT_ID${NC}"
echo -e "Service Name: ${GREEN}$SERVICE_NAME${NC}"
echo -e "Region: ${GREEN}$REGION${NC}\n"

# Enable required APIs
echo -e "${YELLOW}Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com --quiet
gcloud services enable run.googleapis.com --quiet
gcloud services enable containerregistry.googleapis.com --quiet

# Ask about environment variables
echo -e "\n${YELLOW}Environment Variables Configuration:${NC}"
read -p "Do you want to set environment variables? (y/n): " SET_ENV

ENV_VARS=""
if [ "$SET_ENV" = "y" ]; then
    read -p "SUPABASE_URL (optional, press Enter to skip): " SUPABASE_URL
    read -p "SUPABASE_KEY (optional, press Enter to skip): " SUPABASE_KEY
    read -p "ADMIN_USERNAME (optional, press Enter to skip): " ADMIN_USERNAME
    read -p "ADMIN_PASSWORD (optional, press Enter to skip): " ADMIN_PASSWORD
    read -p "OPENAI_API_KEY (optional, press Enter to skip): " OPENAI_API_KEY
    
    # Build environment variables string
    if [ -n "$SUPABASE_URL" ]; then
        ENV_VARS="SUPABASE_URL=$SUPABASE_URL"
    fi
    if [ -n "$SUPABASE_KEY" ]; then
        if [ -n "$ENV_VARS" ]; then
            ENV_VARS="$ENV_VARS,SUPABASE_KEY=$SUPABASE_KEY"
        else
            ENV_VARS="SUPABASE_KEY=$SUPABASE_KEY"
        fi
    fi
    if [ -n "$ADMIN_USERNAME" ]; then
        if [ -n "$ENV_VARS" ]; then
            ENV_VARS="$ENV_VARS,ADMIN_USERNAME=$ADMIN_USERNAME"
        else
            ENV_VARS="ADMIN_USERNAME=$ADMIN_USERNAME"
        fi
    fi
    if [ -n "$ADMIN_PASSWORD" ]; then
        if [ -n "$ENV_VARS" ]; then
            ENV_VARS="$ENV_VARS,ADMIN_PASSWORD=$ADMIN_PASSWORD"
        else
            ENV_VARS="ADMIN_PASSWORD=$ADMIN_PASSWORD"
        fi
    fi
    if [ -n "$OPENAI_API_KEY" ]; then
        if [ -n "$ENV_VARS" ]; then
            ENV_VARS="$ENV_VARS,OPENAI_API_KEY=$OPENAI_API_KEY"
        else
            ENV_VARS="OPENAI_API_KEY=$OPENAI_API_KEY"
        fi
    fi
fi

# Build deployment command
DEPLOY_CMD="gcloud run deploy $SERVICE_NAME \
  --source . \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port $PORT \
  --memory $MEMORY \
  --cpu $CPU \
  --timeout $TIMEOUT \
  --max-instances $MAX_INSTANCES"

# Add environment variables if provided
if [ -n "$ENV_VARS" ]; then
    DEPLOY_CMD="$DEPLOY_CMD --set-env-vars \"$ENV_VARS\""
fi

# Confirm deployment
echo -e "\n${YELLOW}Ready to deploy with the following configuration:${NC}"
echo "  Service: $SERVICE_NAME"
echo "  Region: $REGION"
echo "  Memory: $MEMORY"
echo "  CPU: $CPU"
echo "  Timeout: ${TIMEOUT}s"
echo "  Max Instances: $MAX_INSTANCES"
if [ -n "$ENV_VARS" ]; then
    echo "  Environment Variables: Set"
fi

read -p "\nProceed with deployment? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ]; then
    echo -e "${YELLOW}Deployment cancelled.${NC}"
    exit 0
fi

# Deploy
echo -e "\n${GREEN}Deploying to Cloud Run...${NC}"
eval $DEPLOY_CMD

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo -e "\n${GREEN}✓ Deployment successful!${NC}"
echo -e "Service URL: ${GREEN}$SERVICE_URL${NC}"
echo -e "\nTest the health endpoint:"
echo -e "  ${YELLOW}curl $SERVICE_URL/api/health${NC}"


