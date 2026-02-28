# Google Cloud Run Deployment Script for PowerShell
# This script automates the deployment process on Windows

param(
    [string]$ServiceName = "calculator-mcp-server",
    [string]$Region = "europe-west1",
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

# Set the project
Write-Host "Setting project to 'risolutorematematico'..." -ForegroundColor Yellow
gcloud config set project risolutorematematico --quiet

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

# Function to read .env file
function Read-EnvFile {
    param([string]$FilePath)
    
    $envVars = @{}
    if (Test-Path $FilePath) {
        Write-Host "Reading environment variables from .env file..." -ForegroundColor Green
        $lines = Get-Content $FilePath
        
        foreach ($line in $lines) {
            # Skip empty lines and comments
            $trimmedLine = $line.Trim()
            if ($trimmedLine -eq "" -or $trimmedLine.StartsWith("#")) {
                continue
            }
            
            # Parse KEY=VALUE format
            if ($trimmedLine -match '^([^=]+)=(.*)$') {
                $key = $matches[1].Trim()
                $value = $matches[2].Trim()
                
                # Strip inline comments (anything after a space and #)
                if ($value -match '^(.*?)\s+#') {
                    $value = $matches[1].Trim()
                }
                
                # Remove quotes if present
                if ($value.StartsWith('"') -and $value.EndsWith('"')) {
                    $value = $value.Substring(1, $value.Length - 2)
                } elseif ($value.StartsWith("'") -and $value.EndsWith("'")) {
                    $value = $value.Substring(1, $value.Length - 2)
                }
                
                $envVars[$key] = $value
            }
        }
    } else {
        Write-Host ".env file not found. Skipping environment variables." -ForegroundColor Yellow
    }
    
    return $envVars
}

# Read environment variables from .env file
Write-Host ""
Write-Host "Environment Variables Configuration:" -ForegroundColor Yellow
$envFile = Join-Path (Get-Location) ".env"
$envDict = Read-EnvFile -FilePath $envFile

$envVars = @()
$envVarNames = @("SUPABASE_URL", "SUPABASE_KEY", "ADMIN_USERNAME", "ADMIN_PASSWORD", "OPENAI_API_KEY", "OPENAI_MODEL", "CLOUD_RUN_URL")

foreach ($varName in $envVarNames) {
    if ($envDict.ContainsKey($varName) -and $envDict[$varName]) {
        $envVars += "$varName=$($envDict[$varName])"
        Write-Host "  Found: $varName" -ForegroundColor Green
    }
}

if ($envVars.Count -eq 0) {
    Write-Host "  No environment variables found in .env file." -ForegroundColor Yellow
} else {
    Write-Host "  Total environment variables: $($envVars.Count)" -ForegroundColor Green
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

