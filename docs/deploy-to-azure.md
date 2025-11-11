# Deploy to Azure - Quick Start Guide

This guide walks you through deploying the complete Content Recommendation Engine infrastructure to Azure with a single click.

## 🚀 One-Click Deployment

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fjzagelbaum_microsoft%2Fcapstone%2Fmain%2Finfrastructure%2Fazuredeploy.json/createUIDefinitionUri/https%3A%2F%2Fraw.githubusercontent.com%2Fjzagelbaum_microsoft%2Fcapstone%2Fmain%2Finfrastructure%2FcreateUiDefinition.json)

## Prerequisites

### Required

1. **Azure Subscription**
   - Active Azure subscription with sufficient quota
   - [Create a free account](https://azure.microsoft.com/free/) if you don't have one
   - Free tier includes $200 credit for 30 days

2. **Azure AD Object ID**
   - Your user's Object ID for role assignments
   - Find it in Azure Portal:
     1. Go to **Azure Active Directory**
     2. Click **Users** in left menu
     3. Find and click your username
     4. Copy the **Object ID** (GUID format)

### Recommended

- **Subscription quota** for Azure OpenAI Service
  - Request access: https://aka.ms/oai/access
  - Typically approved within 24-48 hours
  - Required for OpenAI features (can deploy without and add later)

## What Gets Deployed

The one-click deployment provisions a complete, production-ready infrastructure:

### AI & Machine Learning Services

| Service | SKU | Purpose |
|---------|-----|---------|
| **Azure OpenAI Service** | Standard | GPT-4 and text-embedding-ada-002 models for AI recommendations |
| **Azure AI Search** | Basic/Standard | Vector search with semantic capabilities |
| **Azure Machine Learning** | Basic | Traditional ML model training and deployment |

### Data & Storage Services

| Service | SKU | Purpose |
|---------|-----|---------|
| **Azure Cosmos DB** | Serverless | User profiles, preferences, and interactions |
| **Azure Synapse Analytics** | Pay-as-you-go | Large-scale data processing and analytics |
| **Storage Accounts** | Standard LRS | Data lake for raw data, blob storage for artifacts |

### Compute & API Services

| Service | SKU | Purpose |
|---------|-----|---------|
| **Azure Functions (x2)** | Consumption | Serverless API hosting for recommendations |
| **App Service Plans (x2)** | Consumption | Function app hosting |

### Supporting Services

| Service | SKU | Purpose |
|---------|-----|---------|
| **Azure Key Vault** | Standard | Secure storage for secrets and connection strings |
| **Application Insights** | Pay-as-you-go | Application monitoring, logging, and analytics |
| **Log Analytics Workspace** | Pay-as-you-go | Centralized logging and querying |
| **Virtual Network** | Standard | Secure networking with private endpoints |

### Estimated Costs

| Environment | Monthly Cost (Estimate) | Notes |
|-------------|------------------------|-------|
| **Development** | $50-150 | Minimal usage, serverless scaling |
| **Staging** | $200-500 | Medium usage, limited scale |
| **Production** | $500-2000+ | Full scale, high availability |

*Costs vary based on usage, data volume, and API calls. Use Azure Cost Management for accurate tracking.*

## Deployment Steps

### Step 1: Click Deploy to Azure

Click the blue **Deploy to Azure** button at the top of this document. This will:
- Open Azure Portal
- Prompt you to sign in (if not already signed in)
- Load the deployment template with UI wizard

### Step 2: Configure Basic Settings

Fill in the **Basics** section:

#### Subscription
- Select your Azure subscription from the dropdown

#### Resource Group
- **Option A**: Create new (recommended)
  - Click **Create new**
  - Name it: `rg-contentrec-[environment]` (e.g., `rg-contentrec-dev`)
- **Option B**: Use existing
  - Select from dropdown
  - Ensure it's empty or contains compatible resources

#### Region
- Select Azure region closest to your users
- **Recommended regions** for OpenAI availability:
  - East US 2
  - West Europe
  - UK South
  - Australia East
  - Japan East

#### Resource Prefix
- Enter a short, unique prefix (2-10 characters)
- **Must be**: lowercase letters and numbers only
- **Examples**: `contentrec`, `recengine`, `myapp`
- Used to name all resources: `{prefix}-{service}-{uniqueid}`

#### Environment
- Select deployment environment:
  - **Development**: Minimal SKUs, lower costs, for testing
  - **Staging**: Medium SKUs, for pre-production validation
  - **Production**: Higher SKUs, high availability, for live traffic

### Step 3: Configure Identity & Access

Fill in the **Identity & Access** section:

#### Your Azure AD Object ID
- Paste the Object ID you found in prerequisites
- **Format**: `12345678-1234-1234-1234-123456789012`
- This grants you access to deployed Key Vault and other resources

#### Principal Type
- **User** (default): For individual developers
- **Service Principal**: For automation/CI-CD (requires SP Object ID)

### Step 4: Review Configuration Summary

The **Configuration** tab shows what will be deployed:
- ✓ Azure OpenAI Service (GPT-4 + Embeddings)
- ✓ Azure AI Search (Vector Search)
- ✓ Azure Cosmos DB (NoSQL Database)
- ✓ Azure Storage Accounts (Data Lake + Blob)
- ✓ Azure Machine Learning Workspace
- ✓ Azure Synapse Analytics
- ✓ Azure Key Vault (Secrets Management)
- ✓ Application Insights (Monitoring)
- ✓ Azure Functions (2 Function Apps)
- ✓ Virtual Network (Private Endpoints)
- ✓ Log Analytics Workspace

**Estimated deployment time: 15-20 minutes**

### Step 5: Review + Create

1. Click **Review + Create**
2. Azure validates your template
3. Review the summary:
   - All settings
   - Estimated costs
   - Terms and conditions
4. Click **Create** to start deployment

### Step 6: Monitor Deployment

Watch the deployment progress:
- **Status**: Shows current deployment stage
- **Resources**: Lists resources being created
- **Outputs**: Available after completion

**What's happening:**
1. Resource group creation (30 seconds)
2. Networking infrastructure (2-3 minutes)
3. Storage and databases (5-7 minutes)
4. AI/ML services (8-12 minutes)
5. Function apps and monitoring (2-3 minutes)
6. Role assignments and configuration (1-2 minutes)

### Step 7: Deployment Complete! 🎉

When deployment finishes:
1. Click **Outputs** tab to see:
   - Resource Group Name
   - OpenAI Endpoint URL
   - Function App URLs
   - Key Vault Name
   - All service endpoints

2. Save these values - you'll need them for application deployment

## Post-Deployment Steps

### 1. Configure Azure OpenAI Models

Deploy the required models to your OpenAI service:

```bash
# Login to Azure
az login

# Set variables (replace with your values from Outputs)
RESOURCE_GROUP="rg-contentrec-dev"
OPENAI_NAME="contentrec-openai-abc123"

# Deploy GPT-4 model
az cognitiveservices account deployment create \
  --resource-group $RESOURCE_GROUP \
  --name $OPENAI_NAME \
  --deployment-name gpt-4 \
  --model-name gpt-4 \
  --model-version "0613" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name "Standard"

# Deploy embeddings model
az cognitiveservices account deployment create \
  --resource-group $RESOURCE_GROUP \
  --name $OPENAI_NAME \
  --deployment-name text-embedding-ada-002 \
  --model-name text-embedding-ada-002 \
  --model-version "2" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name "Standard"
```

### 2. Clone the Repository

```bash
git clone https://github.com/jzagelbaum_microsoft/capstone.git
cd capstone
```

### 3. Configure Application Settings

Update function app settings with deployed resource information:

```bash
# Get values from deployment outputs
RESOURCE_GROUP="[Your Resource Group]"
FUNCTION_APP_NAME="[Your Function App Name]"
OPENAI_ENDPOINT="[Your OpenAI Endpoint]"
COSMOS_ENDPOINT="[Your Cosmos DB Endpoint]"
SEARCH_ENDPOINT="[Your AI Search Endpoint]"

# Configure main API function app
az functionapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $FUNCTION_APP_NAME \
  --settings \
    "AZURE_OPENAI_ENDPOINT=$OPENAI_ENDPOINT" \
    "AZURE_COSMOS_ENDPOINT=$COSMOS_ENDPOINT" \
    "AZURE_SEARCH_ENDPOINT=$SEARCH_ENDPOINT"
```

### 4. Deploy Application Code

#### Option A: GitHub Actions (Recommended)

1. Fork the repository to your GitHub account
2. Configure GitHub secrets for authentication
3. Push code to trigger deployment:

```bash
git push origin main
```

#### Option B: Manual Deployment

```bash
# Install Azure Functions Core Tools
# https://docs.microsoft.com/en-us/azure/azure-functions/functions-run-local

# Deploy main API
cd src/functions/api
func azure functionapp publish $FUNCTION_APP_NAME --python

# Deploy OpenAI API
cd ../openai_api
func azure functionapp publish $OPENAI_FUNCTION_APP_NAME --python
```

### 5. Initialize Data

Load sample data or configure data ingestion:

```bash
# Upload sample datasets
az storage blob upload-batch \
  --account-name [storage-account] \
  --destination data \
  --source ./sample-data/

# Or use the data generation endpoint
curl -X POST https://[openai-function-app].azurewebsites.net/api/generate-data
```

### 6. Verify Deployment

Test the deployed APIs:

```bash
# Health check
curl https://[function-app].azurewebsites.net/api/health

# Get recommendations
curl -X POST https://[function-app].azurewebsites.net/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "content_type": "movie"}'
```

## Troubleshooting

### Deployment Failed

**Check deployment logs:**
1. Go to Resource Group in Azure Portal
2. Click **Deployments** in left menu
3. Click the failed deployment
4. Review error messages in **Operation details**

**Common issues:**

| Error | Solution |
|-------|----------|
| Quota exceeded | Request quota increase in Azure Portal |
| Resource name conflict | Use different resource prefix |
| Insufficient permissions | Ensure you have Contributor role on subscription |
| Region unavailable | Select different region |

### OpenAI Not Available

If you don't have Azure OpenAI access yet:

1. Deploy infrastructure without OpenAI (it will skip)
2. Request access: https://aka.ms/oai/access
3. After approval, manually add OpenAI service:

```bash
az cognitiveservices account create \
  --name contentrec-openai-[suffix] \
  --resource-group $RESOURCE_GROUP \
  --kind OpenAI \
  --sku S0 \
  --location eastus2
```

### Function App Not Starting

**Check application logs:**
```bash
az functionapp logs tail \
  --resource-group $RESOURCE_GROUP \
  --name $FUNCTION_APP_NAME
```

**Common fixes:**
- Ensure all app settings are configured
- Check that Python runtime is 3.11
- Verify requirements.txt dependencies installed
- Check Application Insights for errors

### Cost Concerns

**Monitor costs:**
1. Go to **Cost Management + Billing** in Azure Portal
2. View **Cost analysis** for your resource group
3. Set up **budgets** and **alerts**

**Reduce costs:**
- Use **Development** environment SKUs
- Delete resources when not in use
- Scale down Cosmos DB throughput
- Use consumption plans for Functions

## Architecture Overview

The deployed infrastructure follows this architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Azure Subscription                          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Resource Group                              │  │
│  │                                                          │  │
│  │  ┌─────────────────────────────────────────────────┐    │  │
│  │  │         Virtual Network (Private)               │    │  │
│  │  │                                                 │    │  │
│  │  │  ┌──────────────┐  ┌──────────────┐            │    │  │
│  │  │  │ Function App │  │ Function App │            │    │  │
│  │  │  │ (Main API)   │  │ (OpenAI API) │            │    │  │
│  │  │  └──────────────┘  └──────────────┘            │    │  │
│  │  │         │                  │                   │    │  │
│  │  └─────────┼──────────────────┼───────────────────┘    │  │
│  │            │                  │                        │  │
│  │  ┌─────────┴──────────────────┴───────────────────┐    │  │
│  │  │              Service Bus                       │    │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │  │
│  │  │  │ OpenAI   │  │ AI Search│  │ Cosmos DB│     │    │  │
│  │  │  └──────────┘  └──────────┘  └──────────┘     │    │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │  │
│  │  │  │ Azure ML │  │ Synapse  │  │ Storage  │     │    │  │
│  │  │  └──────────┘  └──────────┘  └──────────┘     │    │  │
│  │  └────────────────────────────────────────────────┘    │  │
│  │                                                          │  │
│  │  ┌──────────────┐  ┌───────────────────┐               │  │
│  │  │  Key Vault   │  │ App Insights      │               │  │
│  │  └──────────────┘  └───────────────────┘               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Next Steps

After successful deployment:

1. **Configure CI/CD**: Set up GitHub Actions for automated deployments
2. **Load Data**: Import your content catalog and user data
3. **Train Models**: Use Azure ML to train traditional recommendation models
4. **Configure A/B Testing**: Set traffic splits for comparing engines
5. **Set Up Monitoring**: Configure alerts and dashboards in Application Insights
6. **Deploy to Production**: Replicate deployment for staging/production environments

## Additional Resources

- **[Main README](../README.md)** - Project overview and quick start
- **[OpenAI Integration Guide](openai-integration.md)** - OpenAI features and usage
- **[Infrastructure README](../infrastructure/README.md)** - Infrastructure details
- **[Dev Container Guide](../.devcontainer/README.md)** - Development environment setup

## Getting Help

- **Issues**: Open a GitHub issue for bugs or problems
- **Questions**: See [Main README](../README.md) or [Documentation Index](README.md)
- **Azure Support**: Contact Azure support for platform issues
- **OpenAI Questions**: See [OpenAI Integration Guide](openai-integration.md)

## Security Notes

- **Secrets**: Never commit secrets or connection strings to Git
- **Key Vault**: All secrets are stored in Azure Key Vault
- **Networking**: Private endpoints isolate services
- **Access**: Use managed identities where possible
- **Monitoring**: Enable audit logging for compliance

## Clean Up

To delete all deployed resources:

```bash
# Delete the resource group (removes all resources)
az group delete --name rg-contentrec-dev --yes --no-wait

# Or use Azure Portal:
# 1. Go to Resource Groups
# 2. Select your resource group
# 3. Click "Delete resource group"
# 4. Type the name to confirm
# 5. Click Delete
```

**Note**: This is permanent and cannot be undone. Ensure you have backups of any important data.
