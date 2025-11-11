#!/bin/bash
# scripts/check_code_quality.sh
# Run all code quality checks

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate || source .venv/Scripts/activate
fi

ERRORS=0

# Black formatting check
print_header "Checking code formatting (Black)"
if black --check src/ tests/ 2>&1; then
    print_success "Code formatting passed"
else
    print_error "Code formatting issues found. Run: black src/ tests/"
    ERRORS=$((ERRORS + 1))
fi

# isort import sorting check
print_header "Checking import sorting (isort)"
if isort --check-only src/ tests/ 2>&1; then
    print_success "Import sorting passed"
else
    print_error "Import sorting issues found. Run: isort src/ tests/"
    ERRORS=$((ERRORS + 1))
fi

# Flake8 linting
print_header "Running flake8 linting"
if flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203,W503 2>&1; then
    print_success "Flake8 linting passed"
else
    print_error "Flake8 linting issues found"
    ERRORS=$((ERRORS + 1))
fi

# Pylint
print_header "Running pylint"
if pylint src/ --max-line-length=120 --disable=C0103,C0114,C0115,C0116,R0913,R0914,W0511 2>&1 || [ $? -eq 0 ]; then
    print_success "Pylint passed"
else
    print_error "Pylint issues found"
    ERRORS=$((ERRORS + 1))
fi

# MyPy type checking (if mypy is installed)
if command -v mypy &> /dev/null; then
    print_header "Running type checking (MyPy)"
    if mypy src/ --ignore-missing-imports 2>&1; then
        print_success "Type checking passed"
    else
        print_error "Type checking issues found"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Bandit security check
print_header "Running security scan (Bandit)"
if bandit -r src/ -ll 2>&1; then
    print_success "Security scan passed"
else
    print_error "Security issues found"
    ERRORS=$((ERRORS + 1))
fi

# Safety dependency check
print_header "Checking dependencies for vulnerabilities (Safety)"
if safety check --json 2>&1 | grep -q '"vulnerabilities": \[\]' || safety check 2>&1; then
    print_success "Dependency check passed"
else
    print_error "Vulnerable dependencies found"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo -e "\n${YELLOW}========================================${NC}"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ All code quality checks passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ $ERRORS check(s) failed${NC}"
    exit 1
fi
