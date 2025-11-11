#!/bin/bash
# scripts/format_code.sh
# Auto-format code with Black and isort

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🎨 Formatting code...${NC}\n"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate || source .venv/Scripts/activate
fi

# Format with Black
echo "Running Black formatter..."
black src/ tests/
echo -e "${GREEN}✓ Black formatting complete${NC}\n"

# Sort imports with isort
echo "Running isort..."
isort src/ tests/
echo -e "${GREEN}✓ Import sorting complete${NC}\n"

echo -e "${GREEN}✅ Code formatting complete!${NC}"
