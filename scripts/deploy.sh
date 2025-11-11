#!/bin/bash
# scripts/deploy.sh
# Wrapper script for infrastructure deployment with validation

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

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

# Default values
ENVIRONMENT="dev"
LOCATION="eastus2"
VALIDATE_ONLY=false
SKIP_TESTS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -l|--location)
            LOCATION="$2"
            shift 2
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./deploy.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -e, --environment ENV   Environment: dev, staging, prod (default: dev)"
            echo "  -l, --location LOC      Azure region (default: eastus2)"
            echo "  --validate-only         Only validate, don't deploy"
            echo "  --skip-tests            Skip running tests before deployment"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./deploy.sh                                  # Deploy to dev"
            echo "  ./deploy.sh -e prod -l westus2               # Deploy to prod in westus2"
            echo "  ./deploy.sh --validate-only                  # Validate only"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Run './deploy.sh --help' for usage information"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}🚀 Azure Deployment Helper${NC}\n"
print_info "Environment: $ENVIRONMENT"
print_info "Location: $LOCATION"
print_info "Validate only: $VALIDATE_ONLY"
echo ""

# Check Azure CLI
if ! command -v az &> /dev/null; then
    print_error "Azure CLI not found. Please install: https://docs.microsoft.com/cli/azure/install-azure-cli"
    exit 1
fi

# Check login status
if ! az account show &> /dev/null; then
    print_error "Not logged in to Azure. Run: az login"
    exit 1
fi

ACCOUNT_NAME=$(az account show --query name -o tsv)
print_success "Logged in to Azure: $ACCOUNT_NAME"

# Run tests unless skipped
if [ "$SKIP_TESTS" = false ] && [ "$VALIDATE_ONLY" = false ]; then
    print_info "Running tests before deployment..."
    if [ -f "scripts/run_tests.sh" ]; then
        if bash scripts/run_tests.sh -t unit; then
            print_success "Tests passed"
        else
            print_error "Tests failed. Aborting deployment."
            exit 1
        fi
    else
        print_warning "Test script not found, skipping tests"
    fi
fi

# Validate Bicep templates
print_info "Validating Bicep templates..."
cd infrastructure

if [ -f "main.bicep" ]; then
    if az bicep build --file main.bicep; then
        print_success "Bicep validation passed"
    else
        print_error "Bicep validation failed"
        exit 1
    fi
else
    print_error "main.bicep not found"
    exit 1
fi

# Exit if validate-only
if [ "$VALIDATE_ONLY" = true ]; then
    print_success "Validation complete!"
    exit 0
fi

# Confirm deployment
if [ "$ENVIRONMENT" = "prod" ]; then
    print_warning "You are about to deploy to PRODUCTION!"
    read -p "Are you sure you want to continue? (yes/no): " -r
    echo
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        print_info "Deployment cancelled"
        exit 0
    fi
fi

# Run deployment
print_info "Starting deployment..."

if [ -f "deploy.sh" ]; then
    bash deploy.sh -e "$ENVIRONMENT" -l "$LOCATION"
elif [ -f "deploy.ps1" ]; then
    pwsh deploy.ps1 -EnvironmentName "$ENVIRONMENT" -Location "$LOCATION"
else
    print_error "No deployment script found (deploy.sh or deploy.ps1)"
    exit 1
fi

print_success "Deployment initiated!"
