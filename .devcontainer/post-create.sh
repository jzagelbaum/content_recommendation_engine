#!/bin/bash
# .devcontainer/post-create.sh
# Run after container is created

set -e

echo "🚀 Running post-create setup..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Install Python dependencies
if [ -f "requirements.txt" ]; then
    print_info "Installing production dependencies..."
    pip install -r requirements.txt --quiet
    print_success "Production dependencies installed"
fi

if [ -f "requirements-dev.txt" ]; then
    print_info "Installing development dependencies..."
    pip install -r requirements-dev.txt --quiet
    print_success "Development dependencies installed"
fi

# Install package in editable mode
if [ -f "setup.py" ]; then
    print_info "Installing package in editable mode..."
    pip install -e . --quiet
    print_success "Package installed"
fi

# Set up pre-commit hooks
if [ -f ".pre-commit-config.yaml" ]; then
    print_info "Installing pre-commit hooks..."
    pre-commit install --install-hooks
    print_success "Pre-commit hooks installed"
fi

# Create necessary directories
print_info "Creating directory structure..."
mkdir -p logs data/raw data/processed models config
print_success "Directories created"

# Set up .env if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    print_info "Creating .env from template..."
    cp .env.example .env
    print_success ".env created - please update with your credentials"
fi

# Set up local.settings.json for Azure Functions
for func_dir in src/api src/openai-functions; do
    if [ -d "$func_dir" ] && [ ! -f "$func_dir/local.settings.json" ]; then
        print_info "Creating local.settings.json for $func_dir..."
        cat > "$func_dir/local.settings.json" << 'EOF'
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
        print_success "Created local.settings.json for $func_dir"
    fi
done

# Verify Azure CLI
if command -v az &> /dev/null; then
    print_success "Azure CLI $(az version --query '\"azure-cli\"' -o tsv) available"
else
    print_info "Azure CLI not found"
fi

# Verify Azure Functions Core Tools
if command -v func &> /dev/null; then
    print_success "Azure Functions Core Tools $(func --version) available"
else
    print_info "Azure Functions Core Tools not found"
fi

echo ""
print_success "✅ Post-create setup complete!"
echo ""
print_info "Next steps:"
echo "  1. Update .env with your Azure credentials"
echo "  2. Update local.settings.json files in function directories"
echo "  3. Run: az login (Azure credentials are mounted from host)"
echo "  4. Run: ./scripts/run_tests.sh to verify setup"
echo ""
