#!/bin/bash

# Content Recommendation Engine - Infrastructure Deployment Script
# This script deploys the Azure infrastructure using Azure Verified Modules

set -e

# Configuration
SUBSCRIPTION_ID=""
LOCATION="East US"
ENVIRONMENT="dev"
RESOURCE_PREFIX="contentrec"
PRINCIPAL_ID=""  # Your Azure AD User Object ID

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Content Recommendation Engine - Infrastructure Deployment${NC}"
echo "========================================================"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}Azure CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if user is logged in
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}You are not logged in to Azure. Please login first.${NC}"
    az login
fi

# Prompt for required parameters if not set
if [ -z "$SUBSCRIPTION_ID" ]; then
    echo -e "${YELLOW}Please enter your Azure Subscription ID:${NC}"
    read -r SUBSCRIPTION_ID
fi

if [ -z "$PRINCIPAL_ID" ]; then
    echo -e "${YELLOW}Please enter your Azure AD User Object ID:${NC}"
    echo "You can find this by running: az ad signed-in-user show --query id -o tsv"
    read -r PRINCIPAL_ID
fi

# Set the subscription
echo -e "${GREEN}Setting Azure subscription...${NC}"
az account set --subscription "$SUBSCRIPTION_ID"

# Validate the template
echo -e "${GREEN}Validating Bicep template...${NC}"
az deployment sub validate \
    --location "$LOCATION" \
    --template-file main.bicep \
    --parameters location="$LOCATION" \
               environment="$ENVIRONMENT" \
               resourcePrefix="$RESOURCE_PREFIX" \
               principalId="$PRINCIPAL_ID" \
               principalType="User"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Template validation successful!${NC}"
else
    echo -e "${RED}Template validation failed. Please fix the errors and try again.${NC}"
    exit 1
fi

# Deploy the infrastructure
echo -e "${GREEN}Deploying infrastructure...${NC}"
DEPLOYMENT_NAME="contentrec-$ENVIRONMENT-$(date +%Y%m%d-%H%M%S)"

az deployment sub create \
    --name "$DEPLOYMENT_NAME" \
    --location "$LOCATION" \
    --template-file main.bicep \
    --parameters location="$LOCATION" \
               environment="$ENVIRONMENT" \
               resourcePrefix="$RESOURCE_PREFIX" \
               principalId="$PRINCIPAL_ID" \
               principalType="User"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Deployment completed successfully!${NC}"
    
    # Get deployment outputs
    echo -e "${GREEN}Deployment Outputs:${NC}"
    az deployment sub show \
        --name "$DEPLOYMENT_NAME" \
        --query "properties.outputs" \
        --output table
    
    echo -e "${GREEN}Next Steps:${NC}"
    echo "1. Configure your ML workspace and set up compute clusters"
    echo "2. Upload your training data to the Data Lake Storage"
    echo "3. Set up Synapse Analytics pipelines for data processing"
    echo "4. Configure Azure Cognitive Search with your content data"
    echo "5. Deploy your recommendation models to Azure ML endpoints"
    
else
    echo -e "${RED}Deployment failed. Please check the error messages above.${NC}"
    exit 1
fi