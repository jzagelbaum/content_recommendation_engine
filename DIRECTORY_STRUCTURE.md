# Directory Structure

This document describes the organization of the Content Recommendation Engine codebase.

## Overview

The project follows Python best practices with a clear separation between:
- **Library code** (`src/`) - Reusable, testable business logic
- **Function apps** (`src/functions/`) - Thin Azure Functions entry points
- **Infrastructure** (`infrastructure/`) - Bicep templates and IaC
- **Configuration** (`config/`) - Environment-specific settings
- **Documentation** (`docs/`) - Comprehensive project documentation

## Complete Directory Tree

```
capstone/
├── .devcontainer/                    # Development container configuration
│   ├── Dockerfile                    # Container image definition
│   ├── devcontainer.json            # VS Code dev container config
│   ├── postCreateCommand.sh         # Post-creation setup script
│   └── README.md                     # Dev container documentation
│
├── .github/                          # GitHub workflows and configuration
│   └── workflows/
│       └── deploy.yml                # CI/CD pipeline for Azure deployment
│
├── config/                           # Environment-specific configuration
│   ├── dev.json                      # Development environment settings
│   ├── staging.json                  # Staging environment settings
│   └── prod.json                     # Production environment settings
│
├── docs/                             # Project documentation
│   ├── README.md                     # Documentation index
│   ├── deploy-to-azure.md            # One-click Azure deployment guide
│   └── openai-integration.md         # OpenAI features and usage
│
├── infrastructure/                   # Infrastructure as Code (Bicep)
│   ├── main.bicep                    # Main orchestration template
│   ├── modules/                      # Reusable Bicep modules
│   │   ├── functionApp.bicep         # Function App resources
│   │   ├── storage.bicep             # Storage Account resources
│   │   ├── cosmosdb.bicep            # Cosmos DB resources
│   │   ├── openai.bicep              # Azure OpenAI resources
│   │   ├── search.bicep              # Azure AI Search resources
│   │   ├── monitoring.bicep          # Application Insights resources
│   │   └── network.bicep             # Networking resources
│   ├── parameters/                   # Environment-specific parameters
│   │   ├── dev.bicepparam            # Development parameters
│   │   ├── staging.bicepparam        # Staging parameters
│   │   └── prod.bicepparam           # Production parameters
│   └── README.md                     # Infrastructure documentation
│
├── scripts/                          # Automation and utility scripts
│   ├── setup_local.ps1               # Local environment setup
│   ├── run_tests.ps1                 # Test execution script
│   ├── deploy.ps1                    # Deployment automation
│   └── README.md                     # Scripts documentation
│
├── src/                              # Source code (Python package)
│   ├── __init__.py                   # Package marker
│   │
│   ├── core/                         # Core business logic and utilities
│   │   ├── __init__.py
│   │   ├── config.py                 # Configuration management
│   │   ├── logging_config.py         # Logging setup
│   │   ├── data_loader.py            # Data loading utilities
│   │   └── recommendations.py        # Traditional recommendation algorithms
│   │
│   ├── models/                       # Data models and schemas
│   │   ├── __init__.py
│   │   ├── user_models.py            # User-related models
│   │   ├── content_models.py         # Content-related models
│   │   ├── recommendation_models.py  # Recommendation response models
│   │   └── openai_models.py          # OpenAI-specific models
│   │
│   ├── openai/                       # OpenAI integration services
│   │   ├── __init__.py
│   │   ├── service.py                # OpenAI service orchestration
│   │   ├── embeddings.py             # Embedding generation
│   │   ├── completion.py             # GPT completion service
│   │   ├── search_service.py         # Azure AI Search integration
│   │   └── data_generator.py         # Synthetic data generation
│   │
│   ├── ab_testing/                   # A/B testing framework
│   │   ├── __init__.py
│   │   ├── router.py                 # Traffic routing logic
│   │   ├── metrics.py                # Metrics collection
│   │   └── analysis.py               # Statistical analysis
│   │
│   └── functions/                    # Azure Functions (entry points)
│       ├── api/                      # Main recommendation API
│       │   ├── function_app.py       # Function app entry point
│       │   ├── host.json             # Function host configuration
│       │   ├── requirements.txt      # Python dependencies
│       │   └── local.settings.json.template  # Local settings template
│       │
│       └── openai_api/               # OpenAI-specific API
│           ├── function_app.py       # Function app entry point
│           ├── host.json             # Function host configuration
│           ├── requirements.txt      # Python dependencies
│           └── local.settings.json.template  # Local settings template
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── unit/                         # Unit tests
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_recommendations.py
│   │   ├── test_openai_service.py
│   │   ├── test_embeddings.py
│   │   ├── test_router.py
│   │   └── test_metrics.py
│   ├── integration/                  # Integration tests
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   ├── test_openai_api.py
│   │   └── test_ab_testing.py
│   └── fixtures/                     # Test data and fixtures
│       ├── sample_users.json
│       ├── sample_content.json
│       └── sample_interactions.json
│
├── .gitignore                        # Git ignore patterns
├── .funcignore                       # Azure Functions ignore patterns
├── README.md                         # Main project README
├── DIRECTORY_STRUCTURE.md            # This file
├── CONTRIBUTING.md                   # Contribution guidelines
├── LICENSE                           # Project license
├── setup.py                          # Python package setup
├── pyproject.toml                    # Python project metadata
└── requirements.txt                  # Top-level dependencies
```

## Directory Purposes

### Root Level

| File/Directory | Purpose |
|----------------|---------|
| `.devcontainer/` | Complete development environment configuration for VS Code |
| `.github/` | GitHub Actions workflows and repository configuration |
| `config/` | Environment-specific JSON configuration files |
| `docs/` | Project documentation |
| `infrastructure/` | Azure infrastructure definitions (Bicep templates) |
| `scripts/` | Automation scripts for setup, testing, and deployment |
| `src/` | All Python source code (application logic) |
| `tests/` | Comprehensive test suite (unit + integration) |
| `README.md` | Main project documentation and quick start |
| `setup.py` | Makes `src/` installable as a package |
| `pyproject.toml` | Modern Python project metadata and tool configuration |

### Source Code (`src/`)

| Directory | Purpose | Import Pattern |
|-----------|---------|----------------|
| `src/core/` | Core business logic, configuration, traditional ML algorithms | `from src.core.config import Config` |
| `src/models/` | Pydantic models for data validation and serialization | `from src.models.user_models import User` |
| `src/openai/` | Azure OpenAI integration services and utilities | `from src.openai.service import OpenAIService` |
| `src/ab_testing/` | A/B testing framework for comparing recommendation engines | `from src.ab_testing.router import ABTestRouter` |
| `src/functions/` | Azure Functions entry points (thin wrappers) | N/A (entry points only) |

### Function Apps (`src/functions/`)

| Directory | Purpose | Deployment |
|-----------|---------|------------|
| `src/functions/api/` | Main recommendation API with A/B testing | Deploy as separate Function App |
| `src/functions/openai_api/` | OpenAI-specific endpoints (recommendations, data generation) | Deploy as separate Function App |

Each function app:
- Has its own `function_app.py` (entry point)
- Has its own `host.json` (runtime configuration)
- Has its own `requirements.txt` (dependencies)
- Imports from `src/` using absolute imports
- Uses `sys.path` manipulation to resolve `src/` package

### Infrastructure (`infrastructure/`)

| Directory | Purpose |
|-----------|---------|
| `modules/` | Reusable Bicep modules for Azure resources |
| `parameters/` | Environment-specific parameter files (.bicepparam) |

The infrastructure follows Azure best practices:
- Uses Azure Verified Modules where possible
- Separates resources by concern (storage, compute, AI services)
- Supports multi-environment deployments (dev, staging, prod)

### Documentation (`docs/`)

| File | Purpose | Audience |
|------|---------|----------|
| `README.md` | Documentation index and navigation | Everyone |
| `deploy-to-azure.md` | One-click Azure deployment guide | DevOps, Developers |
| `openai-integration.md` | OpenAI features and implementation | ML Engineers, Developers |

Additional documentation is planned and will be added as the project evolves.

## Import Patterns

### Correct Import Examples

```python
# Core utilities and configuration
from src.core.config import Config
from src.core.logging_config import setup_logging
from src.core.recommendations import TraditionalRecommendationEngine

# Data models
from src.models.user_models import User, UserPreferences
from src.models.content_models import Content
from src.models.recommendation_models import RecommendationResponse

# OpenAI services
from src.openai.service import OpenAIService
from src.openai.embeddings import EmbeddingService
from src.openai.completion import CompletionService

# A/B testing
from src.ab_testing.router import ABTestRouter
from src.ab_testing.metrics import MetricsCollector
```

### Incorrect Import Examples (DO NOT USE)

```python
# ❌ Relative imports (fragile, breaks when moved)
from ..core.config import Config
from ...models.user_models import User

# ❌ Direct module imports without src prefix
from core.config import Config
from models.user_models import User

# ❌ Imports from functions directory
from src.functions.api.function_app import app
```

## Path Resolution in Function Apps

Function apps use the following pattern to resolve `src/` imports:

```python
import sys
from pathlib import Path

# Add src/ to Python path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

# Now can import from src/
from src.core.config import Config
```

This allows the function app to import from `src/` regardless of where it's deployed.

## Configuration Files

### Environment Configuration
- `config/dev.json` - Development environment
- `config/staging.json` - Staging environment
- `config/prod.json` - Production environment

### Function App Configuration
- `local.settings.json` - Local runtime settings (git-ignored)
- `local.settings.json.template` - Template for local settings
- `host.json` - Azure Functions host configuration

### Python Package Configuration
- `setup.py` - Package installation and metadata
- `pyproject.toml` - Modern Python tooling configuration
- `requirements.txt` - Top-level dependencies

## Key Principles

1. **Separation of Concerns**: Library code (`src/`) separate from entry points (`src/functions/`)
2. **Testability**: All business logic in `src/` can be unit tested independently
3. **Reusability**: Core modules can be imported by multiple function apps
4. **Python Conventions**: Follow PEP 8 and Python packaging best practices
5. **Clear Boundaries**: Each directory has a single, well-defined purpose
6. **Environment Agnostic**: Configuration separated from code
7. **Documentation**: Each major component has associated documentation

## Adding New Code

### Adding a New Service

1. Create module in appropriate directory:
   - Core logic → `src/core/`
   - OpenAI-related → `src/openai/`
   - A/B testing → `src/ab_testing/`

2. Add models to `src/models/`

3. Add tests to `tests/unit/` or `tests/integration/`

4. Update `requirements.txt` if adding dependencies

5. Document in `docs/`

### Adding a New Function App

1. Create directory: `src/functions/<app_name>/`

2. Add required files:
   - `function_app.py` (entry point)
   - `host.json` (configuration)
   - `requirements.txt` (dependencies)
   - `local.settings.json.template` (settings template)

3. Add path resolution to `function_app.py`:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent.parent))
   ```

4. Import from `src/` using absolute imports

5. Add infrastructure to `infrastructure/modules/`

6. Update documentation in `docs/`

## Related Documentation

- [Main README](README.md) - Project overview and quick start
- [Documentation Index](docs/README.md) - All available documentation
- [Deploy to Azure Guide](docs/deploy-to-azure.md) - Azure deployment
- [OpenAI Integration](docs/openai-integration.md) - OpenAI features
- [Infrastructure Guide](infrastructure/README.md) - Infrastructure details
