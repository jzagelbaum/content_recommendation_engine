#!/bin/bash
# scripts/setup_local.sh
# Local development environment setup script

set -e

echo "🚀 Setting up local development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "ℹ $1"
}

# Check Python version
print_info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    REQUIRED_VERSION="3.9"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
        print_success "Python $PYTHON_VERSION detected"
    else
        print_error "Python 3.9+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Create virtual environment
print_info "Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source .venv/bin/activate || source .venv/Scripts/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
python -m pip install --upgrade pip --quiet
print_success "pip upgraded"

# Install requirements
print_info "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    print_success "Production dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Install development requirements
if [ -f "requirements-dev.txt" ]; then
    print_info "Installing development dependencies..."
    pip install -r requirements-dev.txt --quiet
    print_success "Development dependencies installed"
fi

# Install package in editable mode
if [ -f "setup.py" ]; then
    print_info "Installing package in editable mode..."
    pip install -e . --quiet
    print_success "Package installed in editable mode"
fi

# Create necessary directories
print_info "Creating directory structure..."
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p config
print_success "Directories created"

# Copy environment template
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_success ".env file created from template"
        print_warning "Please update .env with your Azure credentials"
    else
        print_info "Creating .env template..."
        cat > .env << 'EOF'
# Azure Subscription
AZURE_SUBSCRIPTION_ID=
AZURE_TENANT_ID=
AZURE_RESOURCE_GROUP=

# Cosmos DB
COSMOS_DB_ENDPOINT=
COSMOS_DB_KEY=
COSMOS_DB_DATABASE=recommendations

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_GPT_DEPLOYMENT=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Azure AI Search
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_API_KEY=
AZURE_SEARCH_INDEX_NAME=content-recommendations

# Application Insights
APPLICATIONINSIGHTS_CONNECTION_STRING=

# Development
ENVIRONMENT=development
LOG_LEVEL=INFO
EOF
        print_success ".env template created"
        print_warning "Please update .env with your Azure credentials"
    fi
else
    print_warning ".env file already exists"
fi

# Setup Azure Functions local settings
print_info "Setting up Azure Functions local settings..."
for func_dir in src/functions/*/; do
    if [ -f "${func_dir}function_app.py" ]; then
        local_settings="${func_dir}local.settings.json"
        if [ ! -f "$local_settings" ]; then
            cat > "$local_settings" << 'EOF'
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "COSMOS_DB_ENDPOINT": "",
    "COSMOS_DB_KEY": "",
    "COSMOS_DB_DATABASE": "recommendations",
    "AZURE_OPENAI_ENDPOINT": "",
    "AZURE_OPENAI_API_KEY": "",
    "AZURE_OPENAI_GPT_DEPLOYMENT": "gpt-4",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-ada-002",
    "AZURE_SEARCH_ENDPOINT": "",
    "AZURE_SEARCH_API_KEY": "",
    "AZURE_SEARCH_INDEX_NAME": "content-recommendations",
    "APPLICATIONINSIGHTS_CONNECTION_STRING": ""
  }
}
EOF
            print_success "Created local.settings.json in ${func_dir}"
        fi
    fi
done

# Also check src/api and src/openai-functions
for func_dir in src/api src/openai-functions; do
    if [ -d "$func_dir" ]; then
        local_settings="${func_dir}/local.settings.json"
        if [ ! -f "$local_settings" ]; then
            cat > "$local_settings" << 'EOF'
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "COSMOS_DB_ENDPOINT": "",
    "COSMOS_DB_KEY": "",
    "COSMOS_DB_DATABASE": "recommendations",
    "AZURE_OPENAI_ENDPOINT": "",
    "AZURE_OPENAI_API_KEY": "",
    "AZURE_OPENAI_GPT_DEPLOYMENT": "gpt-4",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-ada-002",
    "AZURE_SEARCH_ENDPOINT": "",
    "AZURE_SEARCH_API_KEY": "",
    "AZURE_SEARCH_INDEX_NAME": "content-recommendations",
    "APPLICATIONINSIGHTS_CONNECTION_STRING": ""
  }
}
EOF
            print_success "Created local.settings.json in ${func_dir}"
        fi
    fi
done

# Install pre-commit hooks
if [ -f ".pre-commit-config.yaml" ]; then
    print_info "Installing pre-commit hooks..."
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning "pre-commit not installed. Install with: pip install pre-commit"
    fi
fi

# Check Azure CLI
print_info "Checking Azure CLI..."
if command -v az &> /dev/null; then
    AZ_VERSION=$(az version --query '\"azure-cli\"' -o tsv)
    print_success "Azure CLI $AZ_VERSION installed"
    
    # Check if logged in
    if az account show &> /dev/null 2>&1; then
        ACCOUNT_NAME=$(az account show --query name -o tsv)
        print_success "Logged in to Azure as: $ACCOUNT_NAME"
    else
        print_warning "Not logged in to Azure. Run: az login"
    fi
else
    print_warning "Azure CLI not installed. Install from: https://docs.microsoft.com/cli/azure/install-azure-cli"
fi

# Check Azure Functions Core Tools
print_info "Checking Azure Functions Core Tools..."
if command -v func &> /dev/null; then
    FUNC_VERSION=$(func --version)
    print_success "Azure Functions Core Tools $FUNC_VERSION installed"
else
    print_warning "Azure Functions Core Tools not installed. Install from: https://docs.microsoft.com/azure/azure-functions/functions-run-local"
fi

# Display versions
echo ""
print_info "Environment summary:"
echo "  Python: $(python --version 2>&1)"
echo "  pip: $(pip --version | cut -d' ' -f1,2)"
if command -v az &> /dev/null; then
    echo "  Azure CLI: $(az version --query '\"azure-cli\"' -o tsv)"
fi
if command -v func &> /dev/null; then
    echo "  Functions Core Tools: $(func --version)"
fi

echo ""
print_success "✅ Setup complete!"
echo ""
print_info "Next steps:"
echo "  1. Update .env with your Azure credentials"
echo "  2. Update local.settings.json files with connection strings"
echo "  3. Run 'az login' to authenticate with Azure"
echo "  4. Run 'scripts/run_tests.sh' to verify setup"
echo "  5. Run 'cd infrastructure && ./deploy.sh -e dev' to deploy"
echo ""
