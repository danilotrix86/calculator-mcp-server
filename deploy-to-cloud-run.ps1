# Google Cloud Run Deployment Script for PowerShell
# This script automates the deployment process on Windows

param(
    [string]$ServiceName = "calculator-mcp-server",
    [string]$Region = "europe-west12",
    [string]$Port = "8080",
    [string]$Memory = "2Gi",
    [string]$Cpu = "2",
    [string]$Timeout = "300",
    [string]$MaxInstances = "10"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Google Cloud Run Deployment Script ===" -ForegroundColor Green
Write-Host ""

# Check if gcloud is installed
try {
    $null = Get-Command gcloud -ErrorAction Stop
} catch {
    Write-Host "Error: gcloud CLI is not installed." -ForegroundColor Red
    Write-Host "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
}

# Check if user is authenticated
try {
    $activeAccount = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>&1 | Out-String
    if (-not $activeAccount -or $activeAccount.Trim() -eq "") {
        Write-Host "You are not authenticated. Please run: gcloud auth login" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "You are not authenticated. Please run: gcloud auth login" -ForegroundColor Yellow
    exit 1
}

# Get current project
try {
    $projectId = gcloud config get-value project 2>&1 | Out-String
    $projectId = $projectId.Trim()
    if (-not $projectId -or $projectId -eq "") {
        Write-Host "Error: No project set. Please run: gcloud config set project PROJECT_ID" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Error: No project set. Please run: gcloud config set project PROJECT_ID" -ForegroundColor Red
    exit 1
}

Write-Host "Project ID: $projectId" -ForegroundColor Green
Write-Host "Service Name: $ServiceName" -ForegroundColor Green
Write-Host "Region: $Region" -ForegroundColor Green
Write-Host ""

# Enable required APIs
Write-Host "Enabling required APIs..." -ForegroundColor Yellow
gcloud services enable cloudbuild.googleapis.com --quiet
gcloud services enable run.googleapis.com --quiet
gcloud services enable containerregistry.googleapis.com --quiet

# Ask about environment variables
Write-Host ""
Write-Host "Environment Variables Configuration:" -ForegroundColor Yellow
$setEnv = Read-Host "Do you want to set environment variables? (y/n)"

$envVars = @()
if ($setEnv -eq "y") {
    $supabaseUrl = Read-Host "SUPABASE_URL (optional, press Enter to skip)"
    $supabaseKey = Read-Host "SUPABASE_KEY (optional, press Enter to skip)"
    $adminUsername = Read-Host "ADMIN_USERNAME (optional, press Enter to skip)"
    $adminPassword = Read-Host "ADMIN_PASSWORD (optional, press Enter to skip)"
    $openaiKey = Read-Host "OPENAI_API_KEY (optional, press Enter to skip)"
    $openaiModel = Read-Host "OPENAI_MODEL (optional, press Enter to skip)"
    
    if ($supabaseUrl) { $envVars += "SUPABASE_URL=$supabaseUrl" }
    if ($supabaseKey) { $envVars += "SUPABASE_KEY=$supabaseKey" }
    if ($adminUsername) { $envVars += "ADMIN_USERNAME=$adminUsername" }
    if ($adminPassword) { $envVars += "ADMIN_PASSWORD=$adminPassword" }
    if ($openaiKey) { $envVars += "OPENAI_API_KEY=$openaiKey" }
    if ($openaiModel) { $envVars += "OPENAI_MODEL=$openaiModel" }
    
}

# Build deployment command - we'll construct it as a string to avoid array expansion issues
$deployCmd = "gcloud run deploy $ServiceName --source . --platform managed --region $Region --allow-unauthenticated --port $Port --memory $Memory --cpu $Cpu --timeout $Timeout --max-instances $MaxInstances"

# Add environment variables if provided
if ($envVars.Count -gt 0) {
    $envVarsString = $envVars -join ","
    $deployCmd += " --set-env-vars `"$envVarsString`""
}

# Confirm deployment
Write-Host ""
Write-Host "Ready to deploy with the following configuration:" -ForegroundColor Yellow
Write-Host "  Service: $ServiceName"
Write-Host "  Region: $Region"
Write-Host "  Memory: $Memory"
Write-Host "  CPU: $Cpu"
Write-Host "  Timeout: ${Timeout}s"
Write-Host "  Max Instances: $MaxInstances"
if ($envVars.Count -gt 0) {
    Write-Host "  Environment Variables: Set"
}

$confirm = Read-Host "`nProceed with deployment? (y/n)"

if ($confirm -ne "y") {
    Write-Host "Deployment cancelled." -ForegroundColor Yellow
    exit 0
}

# Deploy
Write-Host ""
Write-Host "Deploying to Cloud Run..." -ForegroundColor Green

# Execute gcloud command as a string to avoid argument parsing issues
Invoke-Expression $deployCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Deployment failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# Get service URL
$serviceUrl = gcloud run services describe $ServiceName --region $Region --format "value(status.url)"

Write-Host ""
Write-Host "Deployment successful!" -ForegroundColor Green
Write-Host "Service URL: $serviceUrl" -ForegroundColor Green
Write-Host ""
Write-Host "Test the health endpoint:" -ForegroundColor Yellow
$healthUrl = $serviceUrl + "/api/health"
Write-Host ('  curl ' + $healthUrl) -ForegroundColor Yellow

