#!/bin/bash
# .devcontainer/post-start.sh
# Run every time the container starts

set -e

echo "🔄 Running post-start tasks..."

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check Azure login status
if az account show &> /dev/null 2>&1; then
    ACCOUNT=$(az account show --query name -o tsv 2>/dev/null)
    print_success "Azure: Logged in as $ACCOUNT"
else
    print_info "Azure: Not logged in (credentials mounted from host)"
fi

# Display environment info
print_info "Environment ready!"
echo "  Python: $(python --version)"
echo "  pip: $(pip --version | cut -d' ' -f1,2)"
echo "  Azure CLI: $(az version --query '\"azure-cli\"' -o tsv 2>/dev/null || echo 'Not available')"
echo "  Functions Core Tools: $(func --version 2>/dev/null || echo 'Not available')"
echo ""
