#!/bin/bash
# scripts/clean.sh
# Clean up build artifacts, cache files, and temporary files

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🧹 Cleaning up workspace...${NC}\n"

# Python cache files
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true
echo -e "${GREEN}✓ Python cache cleaned${NC}"

# Test and coverage files
echo "Removing test and coverage files..."
rm -rf .pytest_cache 2>/dev/null || true
rm -rf htmlcov 2>/dev/null || true
rm -f .coverage 2>/dev/null || true
rm -f coverage.xml 2>/dev/null || true
echo -e "${GREEN}✓ Test artifacts cleaned${NC}"

# Build directories
echo "Removing build directories..."
rm -rf build 2>/dev/null || true
rm -rf dist 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
rm -rf .eggs 2>/dev/null || true
echo -e "${GREEN}✓ Build artifacts cleaned${NC}"

# MyPy cache
echo "Removing MyPy cache..."
rm -rf .mypy_cache 2>/dev/null || true
echo -e "${GREEN}✓ MyPy cache cleaned${NC}"

# Logs
echo "Removing log files..."
rm -rf logs/*.log 2>/dev/null || true
echo -e "${GREEN}✓ Log files cleaned${NC}"

# Azure Functions artifacts
echo "Removing Azure Functions artifacts..."
find . -type d -name ".python_packages" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".azurefunctions" -exec rm -rf {} + 2>/dev/null || true
echo -e "${GREEN}✓ Azure Functions artifacts cleaned${NC}"

echo -e "\n${GREEN}✅ Cleanup complete!${NC}"
