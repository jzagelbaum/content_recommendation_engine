# Scripts Directory

Utility scripts for development, testing, and deployment of the Azure Content Recommendation Engine.

## Available Scripts

### 🔧 Setup & Configuration

#### `setup_local.sh`
Set up local development environment with all dependencies and configuration.

```bash
./scripts/setup_local.sh
```

**What it does:**
- ✅ Validates Python 3.9+ installation
- ✅ Creates and activates virtual environment
- ✅ Installs all Python dependencies
- ✅ Creates required directory structure
- ✅ Generates `.env` and `local.settings.json` templates
- ✅ Installs pre-commit hooks
- ✅ Validates Azure CLI and Functions Core Tools

**First-time setup:** This is your starting point for local development.

---

### 🧪 Testing

#### `run_tests.sh`
Flexible test runner supporting multiple test types and configurations.

```bash
# Run all tests
./scripts/run_tests.sh

# Run unit tests with coverage
./scripts/run_tests.sh -t unit -c

# Run integration tests in parallel
./scripts/run_tests.sh -t integration -p

# Run tests with specific markers, verbose output
./scripts/run_tests.sh -m slow -v
```

**Options:**
- `-t, --type TYPE` - Test type: `all`, `unit`, `integration`, `performance` (default: all)
- `-c, --coverage` - Generate HTML and XML coverage reports
- `-v, --verbose` - Verbose output with detailed test information
- `-m, --markers MARKERS` - Run only tests with specific pytest markers
- `-p, --parallel` - Execute tests in parallel for faster completion
- `-h, --help` - Display help message

**Coverage reports:** Generated in `htmlcov/index.html` and `coverage.xml`

---

### 🎨 Code Quality

#### `check_code_quality.sh`
Comprehensive code quality validation suite.

```bash
./scripts/check_code_quality.sh
```

**Checks performed:**
- ✅ **Black** - Code formatting (PEP 8 compliant)
- ✅ **isort** - Import statement sorting
- ✅ **Flake8** - Linting and style guide enforcement
- ✅ **Pylint** - Static code analysis
- ✅ **MyPy** - Type checking (optional)
- ✅ **Bandit** - Security vulnerability scanning
- ✅ **Safety** - Dependency vulnerability checking

**Exit codes:** Returns non-zero if any check fails, useful for CI/CD pipelines.

#### `format_code.sh`
Auto-format code to meet project standards.

```bash
./scripts/format_code.sh
```

**Actions:**
- 🎨 Formats all Python code with Black (120 char line length)
- 📦 Sorts imports with isort (Black-compatible profile)

**Best practice:** Run before committing code.

---

### 📊 Data Generation

#### `download_datasets.py` ⭐ NEW
Download real-world recommendation datasets (similar to Netflix Prize).

```bash
# List available datasets
python scripts/download_datasets.py --list

# Download MovieLens 1M (recommended - 1M ratings)
python scripts/download_datasets.py movielens-1m

# Download MovieLens 100K (small, fast for testing)
python scripts/download_datasets.py movielens-small

# Download MovieLens 25M (large, production-scale)
python scripts/download_datasets.py movielens-25m

# Download Book-Crossing dataset
python scripts/download_datasets.py book-crossing

# Custom output directory
python scripts/download_datasets.py movielens-1m -o data/real
```

**Available Datasets:**

| Dataset | Records | Size | Description |
|---------|---------|------|-------------|
| `movielens-small` | 100K ratings | ~1 MB | 600 users, 9K movies - Perfect for testing |
| `movielens-1m` | 1M ratings | ~6 MB | 6K users, 4K movies - **Recommended** |
| `movielens-25m` | 25M ratings | ~250 MB | 162K users, 62K movies - Production scale |
| `book-crossing` | 1.1M ratings | ~25 MB | Book ratings from Book-Crossing community |

**Why use real datasets?**
- ✅ **Industry-standard**: MovieLens is the de facto standard for recommendation research
- ✅ **Similar to Netflix Prize**: Same collaborative filtering problem
- ✅ **Real user behavior**: Actual rating patterns and distributions
- ✅ **Validated**: Used in thousands of research papers
- ✅ **Multiple scales**: From testing (100K) to production (25M)

**Output format:**
- `{dataset}_users.json` - User profiles
- `{dataset}_products.json` - Movies/books with metadata
- `{dataset}_interactions.json` - Ratings and interactions
- `{dataset}_summary.json` - Dataset statistics

#### `generate_sample_data.py`
Generate synthetic sample data for development and testing.

```bash
# Generate default dataset
python scripts/generate_sample_data.py

# Generate larger dataset
python scripts/generate_sample_data.py -u 5000 -p 2000 -i 20000

# Custom output directory
python scripts/generate_sample_data.py -o data/test
```

**Options:**
- `-u, --users NUM` - Number of users (default: 1000)
- `-p, --products NUM` - Number of products (default: 500)
- `-i, --interactions NUM` - Number of interactions (default: 5000)
- `-o, --output DIR` - Output directory (default: data/raw)

**Generates:**
- `users.json` - User profiles with preferences
- `products.json` - Product catalog with metadata
- `interactions.json` - User-product interactions (views, clicks, purchases)
- `summary.json` - Dataset statistics

**Categories:** Electronics, books, clothing, home, sports, beauty, toys, food, automotive, health

---

### 🚀 Deployment

#### `deploy.sh`
Safe deployment wrapper with validation and testing.

```bash
# Deploy to development
./scripts/deploy.sh

# Deploy to production (with confirmation)
./scripts/deploy.sh -e prod -l westus2

# Validate templates without deploying
./scripts/deploy.sh --validate-only

# Deploy without running tests
./scripts/deploy.sh --skip-tests
```

**Options:**
- `-e, --environment ENV` - Target environment: `dev`, `staging`, `prod` (default: dev)
- `-l, --location LOC` - Azure region (default: eastus2)
- `--validate-only` - Validate Bicep templates without deploying
- `--skip-tests` - Skip pre-deployment test execution
- `-h, --help` - Show help message

**Safety features:**
- ✅ Validates Azure CLI authentication
- ✅ Runs unit tests before deployment (unless skipped)
- ✅ Validates Bicep templates
- ✅ Requires confirmation for production deployments

---

### 🧹 Maintenance

#### `clean.sh`
Clean workspace of build artifacts and cache files.

```bash
./scripts/clean.sh
```

**Removes:**
- Python cache files (`__pycache__`, `*.pyc`, `*.pyo`)
- Test artifacts (`.pytest_cache`, `htmlcov`, `.coverage`)
- Build directories (`build`, `dist`, `*.egg-info`)
- Type checking cache (`.mypy_cache`)
- Log files (`logs/*.log`)
- Azure Functions artifacts (`.python_packages`, `.azurefunctions`)

**When to use:**
- Before major refactoring
- After dependency updates
- To free disk space
- When troubleshooting import issues

---

## 📋 Common Workflows

### Initial Project Setup
```bash
# 1. Set up local environment
./scripts/setup_local.sh

# 2. Update configuration
# Edit .env with your Azure credentials
# Edit src/*/local.settings.json with connection strings

# 3. Authenticate with Azure
az login
az account set --subscription <your-subscription-id>

# 4. Download real dataset (RECOMMENDED)
python scripts/download_datasets.py movielens-1m

# OR generate synthetic data for quick testing
python scripts/generate_sample_data.py

# 5. Verify setup
./scripts/run_tests.sh -t unit
```

### Daily Development Workflow
```bash
# 1. Pull latest changes
git pull

# 2. Make code changes
# ... edit files ...

# 3. Format code
./scripts/format_code.sh

# 4. Run tests
./scripts/run_tests.sh -t unit -c

# 5. Check code quality
./scripts/check_code_quality.sh

# 6. Commit and push
git add .
git commit -m "Your commit message"
git push
```

### Pre-Deployment Checklist
```bash
# 1. Run all tests with coverage
./scripts/run_tests.sh -c

# 2. Verify code quality
./scripts/check_code_quality.sh

# 3. Validate deployment templates
./scripts/deploy.sh --validate-only

# 4. Deploy to staging
./scripts/deploy.sh -e staging

# 5. Run integration tests against staging
./scripts/run_tests.sh -t integration

# 6. Deploy to production
./scripts/deploy.sh -e prod
```

### Troubleshooting
```bash
# Clean workspace
./scripts/clean.sh

# Rebuild environment
rm -rf .venv
./scripts/setup_local.sh

# Run tests with verbose output
./scripts/run_tests.sh -v

# Check specific test file
pytest tests/unit/test_specific.py -v
```

---

## 🔗 CI/CD Integration

### GitHub Actions Example
```yaml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Environment
        run: ./scripts/setup_local.sh
      
      - name: Run Tests
        run: ./scripts/run_tests.sh -c
      
      - name: Check Code Quality
        run: ./scripts/check_code_quality.sh
      
      - name: Upload Coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Deploy to Azure
        run: ./scripts/deploy.sh -e prod
```

### Azure Pipelines Example
```yaml
trigger:
  - main
  - develop

stages:
  - stage: Test
    jobs:
      - job: TestAndQuality
        steps:
          - script: ./scripts/setup_local.sh
            displayName: 'Setup Environment'
          
          - script: ./scripts/run_tests.sh -c
            displayName: 'Run Tests with Coverage'
          
          - script: ./scripts/check_code_quality.sh
            displayName: 'Code Quality Checks'
          
          - task: PublishCodeCoverageResults@1
            inputs:
              codeCoverageTool: 'Cobertura'
              summaryFileLocation: '$(System.DefaultWorkingDirectory)/coverage.xml'

  - stage: Deploy
    dependsOn: Test
    condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
    jobs:
      - job: DeployProduction
        steps:
          - task: AzureCLI@2
            inputs:
              azureSubscription: 'Azure-Connection'
              scriptType: 'bash'
              scriptLocation: 'scriptPath'
              scriptPath: './scripts/deploy.sh'
              arguments: '-e prod -l eastus2'
```

---

## 🛠️ Script Maintenance

### Making Scripts Executable
```bash
chmod +x scripts/*.sh
```

### Windows Users
Use Git Bash or WSL to run bash scripts, or use PowerShell equivalents:
```powershell
# PowerShell alternative for setup
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Adding New Scripts
When creating new scripts:

1. **Follow naming convention:** `verb_noun.sh` or `verb_noun.py`
2. **Add shebang:** `#!/bin/bash` or `#!/usr/bin/env python3`
3. **Include help text:** Support `-h` or `--help`
4. **Use color output:** Follow existing color scheme
5. **Add error handling:** Use `set -e` for bash
6. **Update this README:** Document usage and examples
7. **Make executable:** `chmod +x scripts/new_script.sh`

---

## 📚 Additional Resources

- [Azure Functions Python Developer Guide](https://docs.microsoft.com/azure/azure-functions/functions-reference-python)
- [Azure CLI Documentation](https://docs.microsoft.com/cli/azure/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Pre-commit Framework](https://pre-commit.com/)

---

## 🤝 Contributing

Improvements to scripts are welcome! Please:
1. Test scripts on both Linux/macOS and Windows (Git Bash/WSL)
2. Update documentation
3. Follow existing code style
4. Add error handling

---

**Questions or issues?** Open an issue or contact the development team.


## Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup_local.ps1` | Set up local development environment | Initial setup |
| `run_tests.ps1` | Execute test suite with coverage | Testing |
| `deploy.ps1` | Deploy to Azure environments | Deployment |

## Setup Scripts

### setup_local.ps1

Sets up your local development environment:

```powershell
# Run local setup
.\scripts\setup_local.ps1
```

**What it does**:
- Creates virtual environment
- Installs Python dependencies
- Configures local settings
- Sets up development tools

## Testing Scripts

### run_tests.ps1

Runs the complete test suite:

### run_tests.ps1

Runs the complete test suite:

```powershell
# Run all tests
.\scripts\run_tests.ps1

# Run with coverage report
.\scripts\run_tests.ps1 -Coverage

# Run specific test category
.\scripts\run_tests.ps1 -Category unit
```

**Options**:
- `-Coverage` - Generate coverage report
- `-Category` - Run specific category (unit, integration, all)
- `-Verbose` - Show detailed output

## Deployment Scripts

### deploy.ps1

Deploys application to Azure:

```powershell
# Deploy to development
.\scripts\deploy.ps1 -Environment dev

# Deploy to production
.\scripts\deploy.ps1 -Environment prod
```

**Parameters**:
- `-Environment` - Target environment (dev, staging, prod)
- `-ResourceGroup` - Azure resource group name
- `-SkipTests` - Skip running tests before deployment

## Common Tasks

### First-Time Setup

```powershell
# 1. Set up local environment
.\scripts\setup_local.ps1

# 2. Run tests to verify
.\scripts\run_tests.ps1

# 3. Start local development
cd src\functions\api
func start
```

### Before Committing

```powershell
# Run tests and linting
.\scripts\run_tests.ps1 -Coverage
```

### Deploying Changes

```powershell
# Deploy to dev environment
.\scripts\deploy.ps1 -Environment dev

# After validation, deploy to prod
.\scripts\deploy.ps1 -Environment prod
```

## Prerequisites

### PowerShell Version
- PowerShell 5.1+ (Windows)
- PowerShell Core 7+ (cross-platform)

### Python
- Python 3.11+
- pip for package management

### Azure
- Azure CLI installed and configured
- Azure Functions Core Tools v4
- Appropriate Azure permissions

## Troubleshooting

### Script Execution Policy

If you get an execution policy error:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Python Not Found

Ensure Python is in your PATH:

```powershell
python --version
# Should show Python 3.11 or higher
```

### Azure CLI Not Authenticated

Login to Azure:

```powershell
az login
az account set --subscription YOUR_SUBSCRIPTION_ID
```

## Best Practices

- **Run tests before deploying** - Always verify code works
- **Use virtual environments** - Keep dependencies isolated
- **Review script output** - Check for warnings or errors
- **Keep scripts updated** - Pull latest changes regularly

## Related Documentation

- **[Main README](../README.md)** - Project overview
- **[Documentation Index](../docs/README.md)** - All documentation
- **[Deploy to Azure Guide](../docs/deploy-to-azure.md)** - Azure deployment
- **[OpenAI Integration](../docs/openai-integration.md)** - OpenAI features

## Contributing

If you improve these scripts:
1. Test thoroughly with `-DryRun` first
2. Add comments explaining changes
3. Update this README
4. Submit pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
