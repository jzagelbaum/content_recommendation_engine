# Content Recommendation Engine - Azure Infrastructure

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fjzagelbaum_microsoft%2Fcapstone%2Fmain%2Finfrastructure%2Fazuredeploy.json/createUIDefinitionUri/https%3A%2F%2Fraw.githubusercontent.com%2Fjzagelbaum_microsoft%2Fcapstone%2Fmain%2Finfrastructure%2FcreateUiDefinition.json)

This repository contains the Azure infrastructure templates for deploying a content recommendation engine using Azure Verified Modules (AVM). The infrastructure is designed to support a scalable, secure, and cost-effective content recommendation system for media and entertainment platforms.

## 🚀 Quick Deploy

**Fastest way to get started**: Click the **Deploy to Azure** button above for one-click deployment of all infrastructure components. See the [Deploy to Azure Guide](../docs/deploy-to-azure.md) for detailed instructions.

**Alternative methods**: Manual deployment using Azure CLI, Bicep, or CI/CD pipelines (see below).

## Architecture Overview

The solution deploys the following Azure services:

### Core Infrastructure
- **Azure Key Vault**: Secure storage for secrets, keys, and certificates
- **Log Analytics Workspace**: Centralized logging and monitoring
- **Application Insights**: Application performance monitoring
- **Container Registry**: Docker image storage for ML models

### Data Platform
- **Azure Data Lake Storage Gen2**: Scalable data lake for raw and processed content data
- **Azure Synapse Analytics**: Data warehouse and big data analytics
- **Azure Data Factory**: Data orchestration and ETL pipelines

### AI/ML Platform
- **Azure Machine Learning**: Model training, deployment, and management
- **Azure Cognitive Search**: Intelligent search and content indexing
- **Azure Cognitive Services**: Additional AI capabilities (text analytics, vision, etc.)

### Networking & Security
- **Virtual Network**: Network isolation and security
- **Private Endpoints**: Secure connectivity to Azure services
- **Private DNS Zones**: Name resolution for private endpoints
- **Network Security Groups**: Network-level security controls

## Prerequisites

1. **Azure CLI**: Install the latest version of Azure CLI
2. **Azure Subscription**: Active Azure subscription with appropriate permissions
3. **User Permissions**: 
   - Subscription Contributor or Owner role
   - Ability to create role assignments
   - Ability to create resource groups

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd capstone/infrastructure
```

### 2. Get Your User Object ID
```bash
az ad signed-in-user show --query id -o tsv
```

### 3. Deploy Using PowerShell (Windows)
```powershell
.\deploy.ps1 -SubscriptionId "your-subscription-id" -PrincipalId "your-user-object-id"
```

### 4. Deploy Using Bash (Linux/macOS)
```bash
chmod +x deploy.sh
./deploy.sh
```

## Manual Deployment

If you prefer to deploy manually using Azure CLI:

### 1. Login to Azure
```bash
az login
az account set --subscription "your-subscription-id"
```

### 2. Validate the Template
```bash
az deployment sub validate \
    --location "East US" \
    --template-file main.bicep \
    --parameters location="East US" \
                environment="dev" \
                resourcePrefix="contentrec" \
                principalId="your-user-object-id" \
                principalType="User"
```

### 3. Deploy the Infrastructure
```bash
az deployment sub create \
    --name "contentrec-deployment" \
    --location "East US" \
    --template-file main.bicep \
    --parameters location="East US" \
                environment="dev" \
                resourcePrefix="contentrec" \
                principalId="your-user-object-id" \
                principalType="User"
```

## Configuration

### Parameters

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `location` | Azure region for deployment | East US | Yes |
| `environment` | Environment name (dev/staging/prod) | dev | Yes |
| `resourcePrefix` | Prefix for resource names | contentrec | Yes |
| `principalId` | Azure AD User/Service Principal Object ID | - | Yes |
| `principalType` | Type of principal (User/ServicePrincipal) | User | Yes |

### Customization

To customize the deployment:

1. **Modify Parameters**: Edit `main.parameters.json` with your specific values
2. **Adjust SKUs**: Modify the SKU settings in the module files for your performance requirements
3. **Add Resources**: Extend the modules to include additional Azure services
4. **Security Settings**: Adjust network access controls and private endpoint configurations

## Azure Verified Modules (AVM)

This infrastructure uses Azure Verified Modules, which are:
- **Microsoft-maintained**: Officially supported by Microsoft
- **Production-ready**: Tested and validated for enterprise use
- **Best practices**: Implement Azure deployment best practices
- **Consistent**: Standardized parameter patterns and outputs

### Used AVM Modules

- `avm/res/network/virtual-network`: Virtual networking infrastructure
- `avm/res/storage/storage-account`: Data Lake Storage Gen2
- `avm/res/synapse/workspace`: Azure Synapse Analytics
- `avm/res/machine-learning-services/workspace`: Azure ML workspace
- `avm/res/search/search-service`: Azure Cognitive Search
- `avm/res/key-vault/vault`: Azure Key Vault
- `avm/res/operational-insights/workspace`: Log Analytics
- `avm/res/insights/component`: Application Insights

## Post-Deployment Configuration

After successful deployment:

### 1. Azure Machine Learning Setup
- Configure compute clusters for training
- Set up datastores pointing to your Data Lake
- Upload training datasets
- Create training pipelines

### 2. Azure Synapse Analytics Setup
- Create linked services to your data sources
- Set up data integration pipelines
- Configure data flows for feature engineering
- Schedule data processing jobs

### 3. Azure Cognitive Search Setup
- Create search indexes for your content
- Configure indexers to automatically process new content
- Set up semantic search capabilities
- Configure search suggestions and facets

### 4. Data Lake Organization
```
/contentrec
├── raw-data/
│   ├── content-metadata/
│   ├── user-interactions/
│   └── external-data/
├── processed-data/
│   ├── features/
│   ├── aggregated/
│   └── cleaned/
└── models/
    ├── training-data/
    ├── model-artifacts/
    └── evaluation-results/
```

## Security Considerations

- **Network Isolation**: All services are deployed with private endpoints where supported
- **Identity & Access**: Uses Azure AD authentication and RBAC
- **Data Encryption**: Encryption at rest and in transit
- **Key Management**: Azure Key Vault for secure secret storage
- **Monitoring**: Comprehensive logging and monitoring with Application Insights

## Cost Optimization

The deployment is configured for development environments. For production:

1. **Adjust SKUs**: Scale up compute and storage SKUs based on requirements
2. **Auto-scaling**: Enable auto-scaling for Synapse and ML compute
3. **Reserved Instances**: Consider reserved instances for predictable workloads
4. **Storage Tiers**: Use appropriate storage tiers (Hot/Cool/Archive)

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure your user account has Contributor access to the subscription
2. **Resource Name Conflicts**: The template uses unique suffixes, but you may need to adjust the resource prefix
3. **Quota Limits**: Check your subscription quotas for VMs and other resources
4. **Region Availability**: Ensure all services are available in your chosen region

### Getting Help

- Check the deployment logs in the Azure portal
- Review the Bicep template validation output
- Ensure all prerequisites are met
- Verify your Azure CLI version is up to date

## Clean Up

To remove all deployed resources:

```bash
# Delete the resource group (this will delete all resources)
az group delete --name "contentrec-dev-rg" --yes --no-wait
```

## Contributing

When contributing to this infrastructure:

1. Follow Azure Verified Module patterns
2. Test deployments in a development environment
3. Update documentation for any new parameters or resources
4. Ensure security best practices are maintained

## License

This project is licensed under the MIT License. See the LICENSE file for details.