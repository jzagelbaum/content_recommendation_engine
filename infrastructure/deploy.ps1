# Content Recommendation Engine - Infrastructure Deployment Script (PowerShell)
# This script deploys the Azure infrastructure using Azure Verified Modules

param(
    [Parameter(Mandatory=$false)]
    [string]$SubscriptionId = "",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "East US",
    
    [Parameter(Mandatory=$false)]
    [string]$Environment = "dev",
    
    [Parameter(Mandatory=$false)]
    [string]$ResourcePrefix = "contentrec",
    
    [Parameter(Mandatory=$false)]
    [string]$PrincipalId = ""
)

Write-Host "Content Recommendation Engine - Infrastructure Deployment" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green

# Check if Azure CLI is installed
if (!(Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Host "Azure CLI is not installed. Please install it first." -ForegroundColor Red
    exit 1
}

# Check if user is logged in
try {
    az account show | Out-Null
} catch {
    Write-Host "You are not logged in to Azure. Please login first." -ForegroundColor Yellow
    az login
}

# Prompt for required parameters if not set
if ([string]::IsNullOrEmpty($SubscriptionId)) {
    $SubscriptionId = Read-Host "Please enter your Azure Subscription ID"
}

if ([string]::IsNullOrEmpty($PrincipalId)) {
    Write-Host "Please enter your Azure AD User Object ID:" -ForegroundColor Yellow
    Write-Host "You can find this by running: az ad signed-in-user show --query id -o tsv" -ForegroundColor Yellow
    $PrincipalId = Read-Host
}

# Set the subscription
Write-Host "Setting Azure subscription..." -ForegroundColor Green
az account set --subscription $SubscriptionId

# Validate the template
Write-Host "Validating Bicep template..." -ForegroundColor Green
$validationResult = az deployment sub validate `
    --location $Location `
    --template-file main.bicep `
    --parameters location=$Location `
                environment=$Environment `
                resourcePrefix=$ResourcePrefix `
                principalId=$PrincipalId `
                principalType="User" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "Template validation successful!" -ForegroundColor Green
} else {
    Write-Host "Template validation failed. Please fix the errors and try again." -ForegroundColor Red
    Write-Host $validationResult -ForegroundColor Red
    exit 1
}

# Deploy the infrastructure
Write-Host "Deploying infrastructure..." -ForegroundColor Green
$deploymentName = "contentrec-$Environment-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

$deploymentResult = az deployment sub create `
    --name $deploymentName `
    --location $Location `
    --template-file main.bicep `
    --parameters location=$Location `
                environment=$Environment `
                resourcePrefix=$ResourcePrefix `
                principalId=$PrincipalId `
                principalType="User" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment completed successfully!" -ForegroundColor Green
    
    # Get deployment outputs
    Write-Host "Deployment Outputs:" -ForegroundColor Green
    az deployment sub show `
        --name $deploymentName `
        --query "properties.outputs" `
        --output table
    
    Write-Host "Next Steps:" -ForegroundColor Green
    Write-Host "1. Configure your ML workspace and set up compute clusters"
    Write-Host "2. Upload your training data to the Data Lake Storage"
    Write-Host "3. Set up Synapse Analytics pipelines for data processing"
    Write-Host "4. Configure Azure Cognitive Search with your content data"
    Write-Host "5. Deploy your recommendation models to Azure ML endpoints"
    
} else {
    Write-Host "Deployment failed. Please check the error messages above." -ForegroundColor Red
    Write-Host $deploymentResult -ForegroundColor Red
    exit 1
}