# Content Recommendation Engine - Complete Azure Solution

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fjzagelbaum%2Fcontent_recommendation_engine%2Fmain%2Finfrastructure%2Fazuredeploy.json/createUIDefinitionUri/https%3A%2F%2Fraw.githubusercontent.com%2Fjzagelbaum%2Fcontent_recommendation_engine%2Fmain%2Finfrastructure%2FcreateUiDefinition.json)

## Overview

This is a comprehensive Azure-native content recommendation engine designed for streaming platforms. The solution leverages modern Azure services to provide scalable, intelligent content recommendations using both traditional machine learning and cutting-edge OpenAI integration, with A/B testing capabilities for comparative analysis.

### 🚀 Quick Deploy

Click the **Deploy to Azure** button above to automatically provision all required infrastructure:
- **Azure OpenAI Service** - GPT-4 and text-embedding-ada-002 models
- **Azure AI Search** - Vector search with semantic capabilities  
- **Azure Cosmos DB** - NoSQL database for user data and interactions
- **Azure Machine Learning** - Traditional ML model training and deployment
- **Azure Synapse Analytics** - Large-scale data processing
- **Storage Accounts** - Data lake and blob storage
- **Azure Functions** - Serverless API hosting (2 function apps)
- **Key Vault** - Secure secrets management
- **Application Insights** - Comprehensive monitoring and logging
- **Virtual Network** - Secure networking with private endpoints

**Deployment takes approximately 15-20 minutes.** After deployment, you'll need to deploy the application code using the provided CI/CD pipelines.

## Architecture

### Core Components

1. **Infrastructure (Bicep/ARM)**
   - Azure Verified Modules for consistent deployment
   - Multi-environment support (dev, staging, prod)
   - Auto-scaling and high availability

2. **Modern Recommendation Engines**
   - **Traditional ML**: Collaborative filtering (SVD), content-based (TF-IDF), hybrid
   - **Deep Learning**: Neural Collaborative Filtering (NCF), Wide & Deep, Two-Tower models
   - **OpenAI-Powered**: Vector search, semantic similarity, GPT-4 insights
   - **Sequential Models**: LSTM/Transformer-based for session & temporal patterns
   - **A/B Testing Framework**: Compare approaches with real-time metrics

3. **API Layer**
   - Azure Functions for serverless APIs
   - Real-time recommendation serving
   - A/B testing router for traffic splitting
   - Interaction tracking and analytics

4. **Search & Discovery**
   - Azure Cognitive Search integration
   - Semantic search capabilities with OpenAI embeddings
   - Faceted navigation and filtering

5. **Monitoring & Analytics**
   - Application Insights telemetry
   - Real-time dashboards with Streamlit
   - A/B test metrics and performance monitoring
   - Power BI business intelligence

## Technology Stack

### Azure Services
- **Azure Functions** - Serverless API hosting for recommendations, search, and monitoring
- **Azure OpenAI Service** - GPT-4 and embedding models for AI-powered recommendations
- **Azure AI Search** - Enhanced search with vector capabilities for semantic similarity
- **Azure Machine Learning** - Traditional ML model training, versioning, and serving
- **Azure Synapse Analytics** - Large-scale data processing and analytics
- **Azure Storage Account** - Data lake and blob storage for datasets
- **Azure Application Insights** - Application performance monitoring and telemetry
- **Azure Key Vault** - Secure secrets and configuration management
- **Azure Log Analytics Workspace** - Centralized logging and monitoring
- **Azure Monitor** - Comprehensive monitoring and alerting
- **Azure Service Bus** - Reliable messaging for event-driven architecture

### Development Stack
- **Python 3.9+** - Primary development language
- **OpenAI Python SDK** - Azure OpenAI integration
- **Pydantic** - Data validation and settings management
- **Bicep** - Infrastructure as Code with Azure Verified Modules
- **MLflow** - ML experiment tracking and model management
- **Streamlit** - Real-time monitoring dashboards
- **Azure CLI & PowerShell** - Deployment automation
- **pytest** - Testing framework with coverage reporting
- **Black, isort, flake8** - Code quality and formatting
- **Bandit, Safety** - Security scanning and vulnerability detection

## Project Structure

The project follows Python best practices with clear separation between library code, function apps, infrastructure, and documentation:

```
content_recommendation_engine/
├── .devcontainer/              # Development container configuration
├── .github/workflows/          # GitHub Actions CI/CD pipelines
├── config/                     # Environment-specific configurations
├── docs/                       # Comprehensive project documentation
│   ├── README.md              # Documentation index
│   ├── deploy-to-azure.md     # One-click Azure deployment guide
│   └── openai-integration.md  # OpenAI features and usage
├── infrastructure/             # Bicep infrastructure templates
│   ├── main.bicep             # Main orchestration template
│   └── modules/               # Reusable Bicep modules
├── scripts/                    # Automation and utility scripts
├── src/                        # Source code (Python package)
│   ├── core/                  # Core business logic and utilities
│   ├── models/                # Data models and schemas
│   ├── openai/                # OpenAI integration services
│   ├── ab_testing/            # A/B testing framework
│   └── functions/             # Azure Functions (entry points)
│       ├── api/               # Main recommendation API
│       └── openai_api/        # OpenAI-specific API
├── tests/                      # Test suite
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── README.md                   # Project overview (this file)
├── CONTRIBUTING.md             # Contribution guidelines
└── requirements.txt            # Python dependencies
```

For a complete description of the directory structure and organization, see the [Documentation Index](docs/README.md).

## Getting Started

### Prerequisites

1. **GitHub Account** - [Create Account](https://github.com) for CI/CD pipeline
2. **Azure CLI** - [Installation Guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
3. **Python 3.9+** - [Download Python](https://www.python.org/downloads/)
4. **Azure Subscription** - [Create Free Account](https://azure.microsoft.com/en-us/free/)
5. **Git** - [Download Git](https://git-scm.com/downloads)

### Local Development Setup

> **Quick Start**: The easiest way to get started is using the dev container. See [.devcontainer/README.md](.devcontainer/README.md) for details.

#### Option 1: Automated Setup (Recommended) ⚡

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/content_recommendation_engine.git
cd content_recommendation_engine

# Run automated setup script
./scripts/setup_local.sh

# Authenticate with Azure
az login
az account set --subscription <your-subscription-id>

# Generate sample data for testing
python scripts/generate_sample_data.py

# Verify setup
./scripts/run_tests.sh -t unit
```

See [scripts/README.md](scripts/README.md) for detailed script documentation.

#### Option 2: Use Dev Container

```bash
# Open in VS Code
code .
# VS Code will prompt to reopen in container
# All dependencies will be automatically installed
```

#### Option 3: Manual Setup

<details>
<summary>Click to expand manual setup instructions</summary>

1. **Fork and clone the repository:**
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/content_recommendation_engine.git
cd content_recommendation_engine
```

2. **Create virtual environment and install dependencies:**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy configuration templates
cp .env.example .env
# Edit .env with your Azure credentials

# Set up Azure Functions local settings
cp src/api/local.settings.json.template src/api/local.settings.json
cp src/openai-functions/local.settings.json.template src/openai-functions/local.settings.json
# Edit with your connection strings
```

3. **Run tests:**
```bash
pytest tests/ -v
```

</details>

---

## 💻 Development

### Available Scripts

The repository includes comprehensive automation scripts for common development tasks:

| Script | Purpose | Usage |
|--------|---------|-------|
| [`setup_local.sh`](scripts/README.md#setup_localsh) | Initial environment setup | `./scripts/setup_local.sh` |
| [`run_tests.sh`](scripts/README.md#run_testssh) | Run tests with options | `./scripts/run_tests.sh -t unit -c` |
| [`check_code_quality.sh`](scripts/README.md#check_code_qualitysh) | Code quality validation | `./scripts/check_code_quality.sh` |
| [`format_code.sh`](scripts/README.md#format_codesh) | Auto-format code | `./scripts/format_code.sh` |
| [`generate_sample_data.py`](scripts/README.md#generate_sample_datapy) | Generate test data | `python scripts/generate_sample_data.py` |
| [`deploy.sh`](scripts/README.md#deploysh) | Deploy to Azure | `./scripts/deploy.sh -e dev` |
| [`clean.sh`](scripts/README.md#cleansh) | Clean workspace | `./scripts/clean.sh` |

See [scripts/README.md](scripts/README.md) for detailed documentation and usage examples.

### Daily Workflow

```bash
# Format code before committing
./scripts/format_code.sh

# Run tests with coverage
./scripts/run_tests.sh -t unit -c

# Check code quality
./scripts/check_code_quality.sh

# Clean workspace when needed
./scripts/clean.sh
```

### Testing

```bash
# Run all tests
./scripts/run_tests.sh

# Run unit tests with coverage
./scripts/run_tests.sh -t unit -c

# Run integration tests (requires Azure)
./scripts/run_tests.sh -t integration

# Run with specific markers
./scripts/run_tests.sh -m slow -v
```

Coverage reports are generated in `htmlcov/index.html`.

### CI/CD Pipeline

4. **Set up GitHub repository secrets:**
```bash
# In your GitHub repository, go to Settings > Secrets and variables > Actions
# Add the following secrets:
# AZURE_CREDENTIALS - Service principal JSON
# AZURE_SUBSCRIPTION_ID - Your Azure subscription ID
# AZURE_CLIENT_ID - Service principal client ID
# AZURE_CLIENT_SECRET - Service principal client secret
# AZURE_TENANT_ID - Your Azure tenant ID
```

5. **Enable GitHub Actions:**
```bash
# The GitHub Actions workflow will automatically trigger on:
# - Push to main branch (production deployment)
# - Push to develop branch (staging deployment)  
# - Push to feature branches (development deployment)
# - Pull requests to main/develop branches (validation only)
```

---

## 🚀 Deployment

### Using Deployment Script (Recommended)

```bash
# Deploy to development
./scripts/deploy.sh -e dev

# Deploy to production (requires confirmation)
./scripts/deploy.sh -e prod -l westus2

# Validate without deploying
./scripts/deploy.sh --validate-only
```

### One-Click Deploy to Azure

#### Option 1: One-Click Deploy to Azure (Fastest) ⚡

The easiest way to get started - deploy all infrastructure with a single click:

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fjzagelbaum%2Fcontent_recommendation_engine%2Fmain%2Finfrastructure%2Fazuredeploy.json/createUIDefinitionUri/https%3A%2F%2Fraw.githubusercontent.com%2Fjzagelbaum%2Fcontent_recommendation_engine%2Fmain%2Finfrastructure%2FcreateUiDefinition.json)

**What you'll need:**
1. **Azure Subscription** - [Create free account](https://azure.microsoft.com/free/) if you don't have one
2. **Your Azure AD Object ID** - Find it in Azure Portal:
   - Go to **Azure Active Directory** → **Users** → Click your name → Copy **Object ID**

**Deployment steps:**
1. Click the **Deploy to Azure** button above
2. Sign in to Azure Portal
3. Fill in the form:
   - **Resource Prefix**: Short name for your resources (e.g., "contentrec")
   - **Environment**: Choose dev, staging, or prod
   - **Your Azure AD Object ID**: Paste the Object ID you copied
   - **Region**: Select closest Azure region
4. Click **Review + Create**, then **Create**
5. Wait 15-20 minutes for deployment to complete

**After infrastructure deployment:**
```bash
# Clone the repository
git clone https://github.com/jzagelbaum/content_recommendation_engine.git
cd content_recommendation_engine

# Deploy application code using GitHub Actions
# (See Option 2 below for GitHub Actions setup)
```

**What gets deployed:**
- ✅ Azure OpenAI Service (GPT-4 + Embeddings)
- ✅ Azure AI Search (Vector search)
- ✅ Azure Cosmos DB (User data & interactions)
- ✅ Azure ML Workspace (Traditional ML models)
- ✅ Azure Synapse Analytics (Data processing)
- ✅ Storage Accounts (Data lake + Blob)
- ✅ Azure Functions (2 serverless APIs)
- ✅ Key Vault (Secrets management)
- ✅ Application Insights (Monitoring)
- ✅ Virtual Network (Private endpoints)

#### Option 2: GitHub Actions Pipeline (Recommended for CI/CD)

For ongoing development and automated deployments, use GitHub Actions:

```bash
# Push code to trigger the pipeline
git add .
git commit -m "Deploy recommendation engine"
git push origin main  # Deploys to production
git push origin develop  # Deploys to staging
git push origin feature/your-feature  # Deploys to development
```

The GitHub Actions workflow automatically:
- Builds and tests the Python application with matrix testing
- Validates Bicep infrastructure templates
- Deploys infrastructure using Azure Verified Modules
- Packages and deploys Azure Function Apps in parallel
- Runs comprehensive smoke tests and health checks
- Provides deployment summaries and notifications
- Supports advanced features like dependency caching and artifact management

See [GitHub Actions Setup](#github-actions-setup) below for configuration details.

#### Option 3: Azure DevOps Pipeline

For enterprise environments using Azure DevOps, the project includes a compatible pipeline:

```bash
# Configure Azure DevOps pipeline using azure-pipelines.yml
# The pipeline triggers on push to main/develop branches
git push origin main
```

#### Option 4: Manual Deployment

For manual deployment or development testing:

#### Option 4: Infrastructure-Only Deployment

To deploy only the infrastructure components:

1. **Deploy Infrastructure:**
```bash
# Login to Azure
az login

# Create resource group
az group create --name rg-recommendation-engine-dev --location "East US 2"

# Deploy Bicep template
az deployment group create \
  --resource-group rg-recommendation-engine-dev \
  --template-file infrastructure/main.bicep \
  --parameters environmentName=dev location=eastus2 projectName=recengine
```

2. **Deploy Function Apps:**
```bash
# Package and deploy API functions
cd src/api
zip -r api.zip .
az functionapp deployment source config-zip \
  --resource-group rg-recommendation-engine-dev \
  --name recengine-api-dev \
  --src api.zip

# Package and deploy search functions
cd ../search
zip -r search.zip .
az functionapp deployment source config-zip \
  --resource-group rg-recommendation-engine-dev \
  --name recengine-search-dev \
  --src search.zip

# Package and deploy monitoring functions
cd ../monitoring
zip -r monitoring.zip .
az functionapp deployment source config-zip \
  --resource-group rg-recommendation-engine-dev \
  --name recengine-monitoring-dev \
  --src monitoring.zip
```

#### CI/CD Pipeline Configuration

The project includes two CI/CD options with GitHub Actions as the recommended approach:

- **GitHub Actions** (Recommended): Uses `.github/workflows/ci-cd.yml`
  - Modern GitHub-native CI/CD integration
  - Advanced parallel job execution and matrix testing
  - Built-in dependency caching and artifact management
  - Supports dev/staging/production environments
  - Automated branch-based deployments with environment protection
  - Comprehensive testing, security scanning, and validation
  - Built-in smoke tests and health checks

- **Azure DevOps**: Uses `azure-pipelines.yml` for enterprise environments
  - Multi-stage deployment pipeline
  - Azure DevOps-specific features and integrations
  - Support for Azure DevOps work item tracking
  - Enterprise governance and compliance features

## GitHub Actions Setup

### Service Principal Configuration

To enable automated Azure deployments via GitHub Actions, you'll need to create an Azure service principal:

```bash
# Login to Azure CLI
az login

# Create a service principal for GitHub Actions
az ad sp create-for-rbac \
  --name "github-actions-content-rec" \
  --role contributor \
  --scopes /subscriptions/YOUR_SUBSCRIPTION_ID \
  --sdk-auth

# Copy the JSON output to GitHub repository secrets as AZURE_CREDENTIALS
```

### GitHub Repository Secrets

Configure the following secrets in your GitHub repository (Settings > Secrets and variables > Actions):

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `AZURE_CREDENTIALS` | Service principal JSON from az ad sp create-for-rbac | `{"clientId": "...", "clientSecret": "...", ...}` |
| `AZURE_SUBSCRIPTION_ID` | Your Azure subscription ID | `12345678-1234-1234-1234-123456789012` |
| `AZURE_CLIENT_ID` | Service principal client ID | `87654321-4321-4321-4321-210987654321` |
| `AZURE_CLIENT_SECRET` | Service principal client secret | `your-client-secret` |
| `AZURE_TENANT_ID` | Your Azure tenant ID | `11111111-1111-1111-1111-111111111111` |

### Workflow Features

The GitHub Actions workflow includes:
- **Matrix testing** across multiple Python versions
- **Parallel job execution** for faster builds
- **Dependency caching** to speed up builds
- **Security scanning** with CodeQL and dependency checks
- **Environment-specific deployments** with approval gates
- **Artifact management** for build outputs
- **Integration with GitHub security features**

### Environment Protection Rules

Configure environment protection rules in GitHub:
1. Go to repository Settings > Environments
2. Create environments: `development`, `staging`, `production`
3. Set protection rules for production (require reviews, restrict branches)
4. Configure environment secrets for each environment if needed

## Configuration

### Environment Variables

```bash
# Azure Authentication
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_TENANT_ID=your-tenant-id

# Azure Services Configuration
AZURE_STORAGE_CONNECTION_STRING=your-storage-connection
AZURE_ML_WORKSPACE_NAME=your-ml-workspace
AZURE_SEARCH_SERVICE_NAME=your-search-service
AZURE_SEARCH_API_KEY=your-search-key
AZURE_KEY_VAULT_URL=https://your-keyvault.vault.azure.net/
AZURE_APPLICATION_INSIGHTS_CONNECTION_STRING=your-app-insights-connection

# Function App Settings
FUNCTIONS_WORKER_RUNTIME=python
FUNCTIONS_EXTENSION_VERSION=~4
WEBSITE_RUN_FROM_PACKAGE=1

# Application Configuration
ENVIRONMENT=dev  # dev, staging, prod
LOG_LEVEL=INFO   # DEBUG, INFO, WARNING, ERROR
CACHE_TTL_MINUTES=30
API_RATE_LIMIT_PER_MINUTE=1000
ML_MODEL_REFRESH_INTERVAL_HOURS=24

# Feature Flags
ENABLE_REAL_TIME_RECOMMENDATIONS=true
ENABLE_CONTENT_BASED_FILTERING=true
ENABLE_COLLABORATIVE_FILTERING=true
ENABLE_HYBRID_RECOMMENDATIONS=true
ENABLE_SEARCH_ANALYTICS=true
```

### Deployment Configuration

The project uses parameter files for environment-specific deployments:

**infrastructure/main.parameters.json**
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "environmentName": {
      "value": "dev"
    },
    "location": {
      "value": "eastus2"
    },
    "projectName": {
      "value": "recengine"
    },
    "enableMonitoring": {
      "value": true
    },
    "enableAdvancedSecurity": {
      "value": false
    }
  }
}
```

**Environment-Specific Configuration**
- **Development**: Basic services, minimal SKUs, debug logging
- **Staging**: Production-like services, standard SKUs, info logging  
- **Production**: Premium services, high availability, minimal logging

### API Documentation

### Recommendation API Endpoints

**Get User Recommendations**
```http
POST /api/recommendations/user
Content-Type: application/json

{
  "user_id": "user123",
  "num_recommendations": 10,
  "categories": ["Action", "Comedy"],
  "exclude_watched": true,
  "algorithm": "hybrid"
}
```

**Get Similar Items**
```http
POST /api/recommendations/similar
Content-Type: application/json

{
  "item_id": "movie456",
  "num_recommendations": 5,
  "similarity_threshold": 0.7
}
```

**Record User Interaction**
```http
POST /api/interactions
Content-Type: application/json

{
  "user_id": "user123",
  "item_id": "movie456",
  "interaction_type": "view",
  "rating": 4.5,
  "timestamp": "2025-10-28T12:00:00Z",
  "context": {
    "device": "mobile",
    "session_id": "session789"
  }
}
```

**Get Trending Content**
```http
GET /api/recommendations/trending?category=Action&time_window=24h&limit=20
```

**Health Check**
```http
GET /api/health
```
**Response Format:**
```json
{
  "status": "success",
  "recommendations": [
    {
      "item_id": "movie456",
      "title": "Epic Action Movie",
      "score": 0.95,
      "confidence": 0.88,
      "metadata": {
        "genre": ["Action", "Adventure"],
        "duration": 120,
        "release_year": 2023,
        "rating": 4.5,
        "popularity": 0.87
      },
      "reasoning": "Based on your viewing history and similar users"
    }
  ],
  "total": 10,
  "algorithm_used": "hybrid",
  "timestamp": "2025-10-28T12:00:00Z",
  "user_context": {
    "personalization_strength": 0.9,
    "exploration_factor": 0.1
  }
}
```

### Search API Endpoints

**Semantic Search**
```http
GET /api/search/semantic?q=action%20movies%20with%20superheroes&size=20&from=0&filters=genre:Action,rating:>4.0
```

**Vector Search**
```http
POST /api/search/vector
Content-Type: application/json

{
  "query_vector": [0.1, 0.2, 0.3, ...],
  "size": 10,
  "similarity_threshold": 0.8,
  "filters": {
    "genre": ["Action", "Sci-Fi"],
    "release_year": {"gte": 2020}
  }
}
```

**Content Discovery**
```http
GET /api/search/discover?user_id=user123&category=Action&mood=adventurous&diversity=0.3
```

**Faceted Search**
```http
GET /api/search/facets?q=movie&facets=genre,year,rating,duration&aggregations=true
```

### Monitoring API Endpoints

**System Performance Metrics**
```http
GET /api/monitoring/performance?start_time=2025-10-27T00:00:00Z&end_time=2025-10-28T00:00:00Z&metrics=latency,throughput,errors
```

**Recommendation Quality Metrics**
```http
GET /api/monitoring/quality?algorithm=hybrid&time_period=7d&metrics=precision,recall,diversity,novelty
```

**User Engagement Analytics**
```http
GET /api/monitoring/engagement?user_segment=premium&time_window=30d&breakdown=daily
```

## Machine Learning

### Recommendation Algorithms

#### Traditional ML (Netflix Prize Era)
1. **Collaborative Filtering**
   - Matrix factorization using SVD
   - User-item interaction analysis
   - Cold start handling

2. **Content-Based Filtering**
   - TF-IDF feature extraction
   - Cosine similarity computation
   - Metadata-driven recommendations

3. **Hybrid Approach**
   - Weighted combination of algorithms
   - Dynamic weight adjustment
   - Performance optimization

#### Modern Deep Learning (Post-2015) ⭐ NEW
4. **Neural Collaborative Filtering (NCF)**
   - Non-linear user-item interaction modeling
   - Deep neural networks for implicit feedback
   - Better cold-start handling with learned embeddings
   - 10-30% improvement over traditional SVD

5. **Sequential & Session-Based Models** (Coming Soon)
   - LSTM/GRU for temporal patterns
   - Transformer-based (BERT4Rec, SASRec)
   - Next-episode prediction
   - Binge-watching pattern detection

6. **Multi-Armed Bandits** (Coming Soon)
   - Thompson Sampling for exploration
   - Contextual bandits (LinUCB)
   - Real-time online learning
   - Balances exploitation vs. exploration

7. **Graph Neural Networks** (Coming Soon)
   - User-item-context graph modeling
   - Multi-hop reasoning
   - Social influence capture
   - Enhanced cold-start performance

**📖 For detailed information on modern techniques, see [Modern Recommendation Techniques Guide](docs/MODERN_RECOMMENDATION_TECHNIQUES.md)**

### Model Training

#### Traditional ML
```python
from ml.recommendation_engine import ContentRecommendationEngine

# Initialize engine
engine = ContentRecommendationEngine()

# Load data
users = pd.read_csv('data/users.csv')
items = pd.read_csv('data/items.csv')
interactions = pd.read_csv('data/interactions.csv')

# Train models
engine.fit_collaborative_model(interactions, users, items)
engine.fit_content_model(items)

# Generate recommendations
recommendations = engine.get_hybrid_recommendations(
    user_id='user123',
    num_recommendations=10
)
```

#### Deep Learning (NCF)
```python
from models.deep_learning import NeuralCollaborativeFiltering

# Initialize NCF model
ncf = NeuralCollaborativeFiltering(
    num_users=10000,
    num_items=5000,
    embedding_dim=64,
    mlp_layers=[128, 64, 32]
)

# Build and train
ncf.build_model()
train_data, val_data = ncf.prepare_data(interactions_df)
metrics = ncf.train(train_data, val_data, epochs=50)

# Get recommendations
recs = ncf.get_recommendations('user123', num_recommendations=10)
print(f"Validation AUC: {metrics['final_val_metrics']['auc']:.4f}")
```

## Monitoring & Analytics

### Real-time Dashboard

Access the Streamlit dashboard:
```bash
streamlit run src/monitoring/dashboard.py
```

Features:
- Real-time recommendation performance
- User engagement metrics
- System health monitoring
- A/B test results comparing traditional vs. OpenAI recommendations

### Power BI Integration

The solution includes automated Power BI integration:
- Dataset refresh automation
- Report generation
- Dashboard embedding
- Business intelligence insights
- A/B test performance analytics

### Application Insights

Monitor application performance:
- Request/response times
- Error rates and exceptions
- Custom telemetry events
- Dependency tracking
- OpenAI service metrics and costs

## OpenAI Integration & A/B Testing

### Key Features

#### Dual Recommendation Engines
- **Traditional ML**: Collaborative filtering, content-based filtering, hybrid models
- **OpenAI-Powered**: Vector search, semantic similarity, AI-generated insights
- **A/B Testing Framework**: Compare approaches with real-time metrics

#### OpenAI Capabilities
- **Azure OpenAI Service**: GPT-4 for content analysis and explanations
- **Vector Embeddings**: Text-embedding-ada-002 for semantic similarity
- **AI Search Integration**: Enhanced search with vector capabilities
- **Synthetic Data Generation**: AI-powered test data creation

#### A/B Testing Features
- **Traffic Splitting**: Configurable percentage routing (default: 30% OpenAI, 70% traditional)
- **Consistent Assignment**: Users see the same variant across sessions
- **Real-time Metrics**: Performance, quality, and engagement tracking
- **Statistical Analysis**: Confidence intervals and significance testing

### OpenAI API Examples

#### Get OpenAI Recommendations
```bash
curl -X POST "https://your-openai-func.azurewebsites.net/api/openai/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "user_profile": {
      "age": 25,
      "preferences": ["action", "comedy"],
      "viewing_history": ["movie1", "movie2"]
    },
    "num_recommendations": 10,
    "context": {
      "device": "mobile",
      "time_of_day": "evening"
    }
  }'
```

#### A/B Test Recommendations
```bash
curl -X POST "https://your-main-func.azurewebsites.net/api/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "user_profile": { ... },
    "enable_ab_test": true,
    "test_name": "openai_vs_traditional"
  }'
```

#### Generate Synthetic Data
```bash
curl -X POST "https://your-openai-func.azurewebsites.net/api/openai/generate-data" \
  -H "Content-Type: application/json" \
  -d '{
    "data_type": "users",
    "count": 50
  }'
```

#### Configure A/B Test
```bash
curl -X POST "https://your-openai-func.azurewebsites.net/api/ab-test/configure" \
  -H "Content-Type: application/json" \
  -d '{
    "test_name": "custom_test",
    "traffic_split": 0.4,
    "enabled": true,
    "control_algorithm": "traditional",
    "treatment_algorithm": "openai",
    "description": "Testing 40% OpenAI traffic"
  }'
```

#### Get A/B Test Results
```bash
curl "https://your-openai-func.azurewebsites.net/api/ab-test/results?test_name=openai_vs_traditional&days_back=7"
```

### Configuration

#### Environment Variables for OpenAI
```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_GPT_DEPLOYMENT=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# A/B Testing Configuration
ENABLE_AB_TESTING=true
DEFAULT_AB_TEST_TRAFFIC_SPLIT=0.3

# AI Search Configuration
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_INDEX_NAME=content-recommendations
```

For detailed OpenAI integration documentation, see [docs/openai-integration.md](docs/openai-integration.md).

### Testing Strategy

### Automated Testing Pipeline

The project includes comprehensive testing integrated into the CI/CD pipeline:

```bash
# Run all tests locally
pytest tests/ -v --cov=src --cov-report=html

# Run specific test categories
pytest tests/ -m unit -v                    # Unit tests only
pytest tests/ -m integration -v             # Integration tests only
pytest tests/ -m "not azure" -v            # Non-Azure dependent tests
pytest tests/ -m azure -v                  # Azure service tests (requires config)
```

### Test Categories

**Unit Tests**
- ML algorithm validation
- Data processing functions
- API endpoint logic
- Configuration management
- Utility functions

**Integration Tests**
- Azure service integration
- End-to-end recommendation flows
- Search functionality
- Monitoring and alerting
- Database operations

**Performance Tests**
- API response times
- ML model inference speed
- Search query performance
- Concurrent user handling
- Memory and CPU usage

### Code Quality Checks

The CI/CD pipeline automatically runs:

```bash
# Code formatting
black --check --diff src/
isort --check-only --diff src/

# Linting
flake8 src/ --max-line-length=88 --extend-ignore=E203,W503

# Security scanning
bandit -r src/ -f json
safety check --json

# Type checking (if configured)
mypy src/ --ignore-missing-imports
```

### Load and Performance Testing

**Azure Load Testing Integration**
```bash
# The CI/CD pipeline can trigger load tests
# Load test configurations are defined in the infrastructure
# Results are automatically captured in Application Insights
```

**Local Performance Testing**
```bash
# Install testing dependencies
pip install locust pytest-benchmark

# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Run load simulation
locust -f tests/load/locustfile.py --host=https://your-api-endpoint
```

## Sample Data and Development

### Development Data Generation

The project includes utilities for generating realistic test data:

```python
from src.data.sample_data_generator import SampleDataGenerator

# Generate development datasets
generator = SampleDataGenerator()
generator.generate_users(num_users=10000, output_file="users.json")
generator.generate_items(num_items=1000, output_file="items.json")
generator.generate_interactions(
    num_interactions=50000, 
    output_file="interactions.json"
)
```

### Data Schema

**Users Dataset**
```json
{
  "user_id": "string",
  "demographics": {
    "age": "integer",
    "gender": "string",
    "location": "string"
  },
  "preferences": {
    "genres": ["string"],
    "viewing_time": "string",
    "device_types": ["string"]
  },
  "subscription": {
    "type": "string",
    "start_date": "datetime",
    "features": ["string"]
  }
}
```

**Items Dataset**
```json
{
  "item_id": "string",
  "title": "string",
  "metadata": {
    "genre": ["string"],
    "duration": "integer",
    "release_year": "integer",
    "rating": "float",
    "description": "string",
    "cast": ["string"],
    "director": "string"
  },
  "content_features": {
    "language": "string",
    "subtitles": ["string"],
    "video_quality": ["string"],
    "content_rating": "string"
  }
}
```

**Interactions Dataset**
```json
{
  "user_id": "string",
  "item_id": "string",
  "interaction_type": "string",
  "timestamp": "datetime",
  "rating": "float",
  "watch_duration": "integer",
  "completion_rate": "float",
  "context": {
    "device": "string",
    "session_id": "string",
    "playlist_context": "string"
  }
}
```

## Performance Optimization

### Caching Strategy

- **In-memory caching** for frequently accessed recommendations
- **Redis caching** for session data and user preferences
- **CDN caching** for static content and metadata

### Scaling Considerations

- **Horizontal scaling** with Azure Functions
- **Auto-scaling** based on demand
- **Load balancing** across multiple instances
- **Database sharding** for large datasets

## Security

### Authentication & Authorization

- **Azure AD integration** for user authentication
- **API key management** with Key Vault
- **Role-based access control** (RBAC)
- **Managed identities** for service-to-service auth

### Data Protection

- **Encryption at rest** for all storage
- **TLS encryption** for data in transit
- **Private endpoints** for internal communication
- **Data anonymization** for analytics

## Troubleshooting

### Common Issues

1. **Function App deployment fails**
   - Check Azure CLI authentication
   - Verify resource group exists
   - Check function app name availability

2. **ML model training errors**
   - Validate data format and schema
   - Check Azure ML workspace permissions
   - Verify compute resources availability

3. **Search service not responding**
   - Check Azure Search service status
   - Verify API keys and endpoints
   - Test network connectivity

### Debugging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
export AZURE_LOG_LEVEL=DEBUG
```

Check Application Insights for detailed telemetry and error tracking.

## Contributing

We welcome contributions! Please see our comprehensive [Contributing Guide](CONTRIBUTING.md) for details on:

- Setting up your development environment
- Code style guidelines and testing requirements
- Commit message format and pull request process
- Architecture guidelines for adding new features
- Documentation standards

Quick start for contributors:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and add tests
4. Commit changes: `git commit -am 'Add new feature'`
5. Push to branch: `git push origin feature/new-feature`
6. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for complete guidelines.

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

### Available Now
- **[Documentation Index](docs/README.md)** - Find everything you need
- **[Deploy to Azure Guide](docs/deploy-to-azure.md)** - One-click deployment guide
- **[OpenAI Integration Guide](docs/openai-integration.md)** - OpenAI features and usage
- **[Dev Container Guide](.devcontainer/README.md)** - Development container usage
- **[Infrastructure Guide](infrastructure/README.md)** - Infrastructure details

### Coming Soon
Additional documentation is being developed for:
- Architecture Overview - System design and components
- API Reference - Complete API documentation
- A/B Testing Guide - A/B testing framework
- Development Guide - Local development setup
- Deployment Guide - Deployment procedures
- Monitoring Guide - Monitoring and logging
- Troubleshooting Guide - Common issues and solutions
- Testing Guide - Testing strategies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the [documentation](docs/)
- Review the [troubleshooting guide](#troubleshooting)

## Acknowledgments

- Azure Verified Modules team for infrastructure templates
- MLflow community for experiment tracking
- Streamlit team for dashboard framework
- Azure documentation and samples