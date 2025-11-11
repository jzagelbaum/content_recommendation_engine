# Dev Container Configuration

This directory contains the development container configuration for the Azure Content Recommendation Engine.

## What is a Dev Container?

A development container (dev container) is a running Docker container with a well-defined tool/runtime stack and its prerequisites. This project includes a dev container configuration that:

- ✅ Provides a consistent development environment
- ✅ Includes all required tools and dependencies
- ✅ Works on Windows, macOS, and Linux
- ✅ Integrates with VS Code
- ✅ Mounts Azure credentials from your host machine

## Prerequisites

1. **[Visual Studio Code](https://code.visualstudio.com/)**
2. **[Docker Desktop](https://www.docker.com/products/docker-desktop)**
   - Windows: WSL 2 backend recommended
   - macOS: Latest version
   - Linux: Docker Engine + Docker Compose
3. **[Dev Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)**

## Quick Start

### Option 1: Open in Container (VS Code)

1. **Open VS Code** in the project root directory
2. **Command Palette** (`Ctrl+Shift+P` or `Cmd+Shift+P`)
3. Select: **"Dev Containers: Reopen in Container"**
4. Wait for container to build and start (first time takes ~5-10 minutes)
5. **Done!** You're in the dev container

### Option 2: Clone in Container

1. **Command Palette** (`Ctrl+Shift+P` or `Cmd+Shift+P`)
2. Select: **"Dev Containers: Clone Repository in Container Volume"**
3. Enter repository URL
4. Wait for clone and container build
5. **Done!**

## What's Included

### Development Tools

- **Python 3.11** with pip, setuptools, wheel
- **Azure CLI** with Bicep support
- **Azure Functions Core Tools v4**
- **Docker-in-Docker** for container builds
- **Git** and **GitHub CLI**
- **Node.js LTS** (for additional tooling)

### Python Development Tools

- **black** - Code formatting
- **isort** - Import sorting
- **pylint** - Linting
- **flake8** - Style guide enforcement
- **mypy** - Type checking
- **pytest** - Testing framework
- **bandit** - Security scanning
- **safety** - Dependency vulnerability checking
- **pre-commit** - Git hooks

### VS Code Extensions

Automatically installed in the container:

- Python
- Pylance
- Black Formatter
- Azure Functions
- Bicep
- Docker
- GitHub Copilot
- Jupyter
- YAML
- REST Client

### Port Forwarding

The following ports are automatically forwarded:

| Port | Service | Description |
|------|---------|-------------|
| 7071 | Main API | Traditional recommendation API |
| 7072 | OpenAI API | OpenAI-powered recommendation API |
| 8000 | Web App | Demo web application |
| 8501 | Dashboard | Streamlit monitoring dashboard |

## Container Features

### Azure Credentials

Your Azure credentials are **automatically mounted** from your host machine:

- `~/.azure` → `/home/vscode/.azure`
- `~/.ssh` → `/home/vscode/.ssh` (read-only)

This means:
- ✅ No need to run `az login` in the container
- ✅ Your existing Azure sessions work immediately
- ✅ Credentials stay on your host machine (secure)

### Automatic Setup

When the container is created, it automatically:

1. ✅ Installs all Python dependencies
2. ✅ Sets up pre-commit hooks
3. ✅ Creates necessary directories
4. ✅ Generates configuration templates
5. ✅ Verifies Azure tooling

### Persistent Storage

The following are mounted from your host:

- **Source code**: Bind mount (changes persist)
- **Azure credentials**: Bind mount (from host)
- **SSH keys**: Read-only bind mount
- **Extensions**: Container volume (persists between rebuilds)

## Usage

### Working with Azure Functions

```bash
# Start main API
cd src/api
func start

# Start OpenAI API (in new terminal)
cd src/openai-functions
func start --port 7072
```

### Running Tests

```bash
# All tests
./scripts/run_tests.sh

# Unit tests with coverage
./scripts/run_tests.sh -t unit -c

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Code Quality

```bash
# Format code
./scripts/format_code.sh

# Check quality
./scripts/check_code_quality.sh
```

### Deployment

```bash
# Deploy to development
./scripts/deploy.sh -e dev

# Deploy to production
./scripts/deploy.sh -e prod
```

## Troubleshooting

### Container Won't Start

**Problem**: Container fails to build or start

**Solutions**:
```bash
# 1. Check Docker is running
docker ps

# 2. Rebuild without cache
# Command Palette → "Dev Containers: Rebuild Container Without Cache"

# 3. Check Docker logs
docker logs azure-recommendation-engine-devcontainer
```

### Azure CLI Not Working

**Problem**: `az` command not found or not logged in

**Solutions**:
```bash
# 1. Verify Azure CLI installation
az --version

# 2. Check mounted credentials
ls -la ~/.azure

# 3. Login manually if needed
az login
```

### Port Already in Use

**Problem**: Port 7071 or 7072 already in use

**Solutions**:
```bash
# 1. Find process using port
lsof -i :7071  # macOS/Linux
netstat -ano | findstr :7071  # Windows

# 2. Kill process or use different port
func start --port 7073
```

### Python Imports Not Working

**Problem**: Import errors or modules not found

**Solutions**:
```bash
# 1. Verify Python path
which python
# Should output: /usr/local/bin/python

# 2. Reinstall dependencies
pip install -r requirements.txt

# 3. Reinstall package in editable mode
pip install -e .
```

## Resources

- [VS Code Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [Dev Container Specification](https://containers.dev/)
- [Azure Dev Containers](https://github.com/Azure-Samples/azure-dev-container-samples)
- [Project Documentation](../docs/README.md)

---

**Happy Coding! 🚀**
