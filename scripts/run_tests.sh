#!/bin/bash
# scripts/run_tests.sh
# Comprehensive test runner with multiple modes

set -e

# Default values
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false
MARKERS=""
PARALLEL=false

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -m|--markers)
            MARKERS="$2"
            shift 2
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --type TYPE       Test type: all, unit, integration, performance (default: all)"
            echo "  -c, --coverage        Run with coverage report"
            echo "  -v, --verbose         Verbose output"
            echo "  -m, --markers MARKERS Run tests with specific markers"
            echo "  -p, --parallel        Run tests in parallel"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                          # Run all tests"
            echo "  ./run_tests.sh -t unit -c               # Run unit tests with coverage"
            echo "  ./run_tests.sh -m slow -v               # Run tests marked as 'slow' verbosely"
            echo "  ./run_tests.sh -t integration -p        # Run integration tests in parallel"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run './run_tests.sh --help' for usage information"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}🧪 Running tests...${NC}"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate || source .venv/Scripts/activate
fi

# Build pytest command
PYTEST_CMD="pytest"

# Add test directory based on type
case $TEST_TYPE in
    unit)
        PYTEST_CMD="$PYTEST_CMD tests/unit/"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD tests/integration/"
        ;;
    performance)
        PYTEST_CMD="$PYTEST_CMD tests/performance/"
        ;;
    all)
        PYTEST_CMD="$PYTEST_CMD tests/"
        ;;
    *)
        echo "Invalid test type: $TEST_TYPE"
        exit 1
        ;;
esac

# Add verbosity
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add coverage
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml"
fi

# Add markers
if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD -m $MARKERS"
fi

# Add parallel execution
if [ "$PARALLEL" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

# Add other useful flags
PYTEST_CMD="$PYTEST_CMD --tb=short --strict-markers"

echo "Executing: $PYTEST_CMD"
echo ""

# Run tests
eval $PYTEST_CMD

# Display coverage report location if generated
if [ "$COVERAGE" = true ]; then
    echo ""
    echo -e "${YELLOW}📊 Coverage report generated:${NC}"
    echo "  HTML: htmlcov/index.html"
    echo "  XML: coverage.xml"
fi

echo ""
echo -e "${GREEN}✅ Tests complete!${NC}"
