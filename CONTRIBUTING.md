# Contributing to Content Recommendation Engine

Thank you for your interest in contributing! This guide will help you get started.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.11+
- Azure CLI
- Azure Functions Core Tools v4
- VS Code (recommended)
- Git

### Setting Up Your Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/capstone.git
   cd capstone
   ```

2. **Use the dev container (recommended):**
   ```bash
   # Open in VS Code
   code .
   # VS Code will prompt to reopen in container
   ```

   Or set up manually:
   ```bash
   # Create virtual environment
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows PowerShell
   
   # Install dependencies
   pip install -e .
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Configure local settings:**
   ```bash
   # Copy templates
   Copy-Item src/functions/api/local.settings.json.template src/functions/api/local.settings.json
   Copy-Item src/functions/openai_api/local.settings.json.template src/functions/openai_api/local.settings.json
   
   # Edit with your Azure credentials
   code src/functions/api/local.settings.json
   ```

4. **Run tests to verify setup:**
   ```bash
   pytest tests/
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Use prefixes:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or improvements

### 2. Make Your Changes

Follow these guidelines:

#### Code Style

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting
- Use type hints where appropriate
- Write docstrings for all public functions/classes

**Example:**
```python
from typing import List, Optional
from src.models.user_models import User

def get_recommendations(
    user_id: str,
    content_type: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Get personalized recommendations for a user.
    
    Args:
        user_id: Unique identifier for the user
        content_type: Type of content to recommend
        limit: Maximum number of recommendations to return
        
    Returns:
        List of recommendation dictionaries with scores
        
    Raises:
        ValueError: If user_id is invalid
        ConfigurationError: If recommendation engine not configured
    """
    # Implementation
    pass
```

#### Testing

- Write tests for all new code
- Maintain test coverage above 80%
- Include unit tests and integration tests
- Use meaningful test names

**Example:**
```python
import pytest
from src.core.recommendations import TraditionalRecommendationEngine

class TestTraditionalRecommendationEngine:
    def test_collaborative_filtering_returns_recommendations(self):
        """Test that collaborative filtering returns expected recommendations."""
        engine = TraditionalRecommendationEngine()
        recommendations = engine.collaborative_filter("user123", limit=10)
        
        assert len(recommendations) <= 10
        assert all("content_id" in rec for rec in recommendations)
        assert all("score" in rec for rec in recommendations)
    
    def test_collaborative_filtering_raises_on_invalid_user(self):
        """Test that invalid user_id raises ValueError."""
        engine = TraditionalRecommendationEngine()
        
        with pytest.raises(ValueError, match="Invalid user_id"):
            engine.collaborative_filter("", limit=10)
```

#### Documentation

- Update documentation for any changes
- Add docstrings to all new functions/classes
- Update README.md if changing features
- Add examples for new functionality

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_recommendations.py

# Run with coverage
pytest --cov=src tests/

# Run linting
black src/ tests/
flake8 src/ tests/
mypy src/
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add OpenAI embedding caching

- Implement Redis caching for embeddings
- Add cache configuration options
- Reduce API calls by 70%
- Add tests for cache behavior"
```

**Commit message format:**
```
<type>: <short summary>

<detailed description>

<footer>
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting, etc.)
- `refactor` - Code refactoring
- `test` - Test changes
- `chore` - Build process, dependencies, etc.

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title describing the change
- Detailed description of what and why
- Reference to any related issues
- Screenshots if applicable (for UI changes)

## What to Work On

### Good First Issues

Look for issues labeled `good-first-issue`:
- Documentation improvements
- Adding tests
- Small bug fixes
- Code cleanup

### Areas Needing Help

- **Testing**: Increase test coverage
- **Documentation**: Expand guides and examples
- **Performance**: Optimize recommendation algorithms
- **Features**: Implement items from the roadmap
- **Infrastructure**: Improve deployment templates

## Project Structure

Understand the codebase organization:

```
src/
├── core/           # Core business logic
├── models/         # Data models
├── openai/         # OpenAI integration
├── ab_testing/     # A/B testing framework
└── functions/      # Azure Functions entry points
```

See [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md) for details.

## Architecture Guidelines

### Adding New Recommendation Algorithms

1. Implement in `src/core/recommendations.py`
2. Add tests in `tests/unit/test_recommendations.py`
3. Update documentation in `docs/`
4. Update API to expose new algorithm

### Adding New OpenAI Features

1. Implement service in `src/openai/`
2. Add models in `src/models/openai_models.py`
3. Add tests in `tests/unit/test_openai_*.py`
4. Update function app in `src/functions/openai_api/`
5. Document in `docs/openai-integration.md`

### Adding New API Endpoints

1. Add route to `src/functions/*/function_app.py`
2. Implement logic in appropriate `src/` module
3. Add integration tests
4. Update documentation in `docs/`
5. Update OpenAPI/Swagger specs

### Modifying Infrastructure

1. Update Bicep templates in `infrastructure/`
2. Test deployment to dev environment
3. Document changes in `infrastructure/README.md`
4. Update parameter files for all environments

## Testing Guidelines

### Unit Tests

- Test individual functions/classes in isolation
- Mock external dependencies
- Fast execution (< 1 second each)
- Location: `tests/unit/`

### Integration Tests

- Test component interactions
- Use test Azure resources
- Can be slower (< 30 seconds each)
- Location: `tests/integration/`

### Test Fixtures

- Reusable test data in `tests/fixtures/`
- Use pytest fixtures for setup/teardown
- Keep test data small and focused

### Running Tests

```bash
# All tests
pytest

# Specific category
pytest tests/unit/
pytest tests/integration/

# Specific file
pytest tests/unit/test_recommendations.py

# Specific test
pytest tests/unit/test_recommendations.py::TestClass::test_method

# With coverage
pytest --cov=src --cov-report=html tests/

# Parallel execution (faster)
pytest -n auto tests/
```

## Code Review Process

### For Contributors

1. Ensure all tests pass
2. Update documentation
3. Follow code style guidelines
4. Write clear commit messages
5. Respond to review feedback promptly

### For Reviewers

- Be constructive and respectful
- Check for:
  - Correctness
  - Test coverage
  - Documentation
  - Code style
  - Performance implications
  - Security concerns
- Approve when satisfied

## Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

1. **MAJOR**: Breaking changes
2. **MINOR**: New features (backward compatible)
3. **PATCH**: Bug fixes

### Creating a Release

1. Update version in `setup.py` and `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release branch: `git checkout -b release/v1.2.0`
4. Test thoroughly
5. Merge to main
6. Tag release: `git tag v1.2.0`
7. Push tags: `git push --tags`

## Documentation

### Adding Documentation

- Main docs in `docs/` directory
- Use Markdown format
- Include code examples
- Add to `docs/README.md` index
- Cross-reference related docs

### Documentation Standards

- Clear, concise writing
- Code examples that work
- Screenshots where helpful
- Keep up-to-date with code changes

## Community

### Getting Help

- Read the [Documentation](docs/README.md)
- Open an issue for bugs
- Start a discussion for questions

### Reporting Bugs

Include:
1. Description of the bug
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details (OS, Python version, etc.)
6. Error messages/logs

**Bug report template:**
```markdown
## Description
Clear description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: Windows 11
- Python: 3.11.5
- Azure Functions Core Tools: 4.0.5455

## Error Messages
```
Paste error messages here
```

## Additional Context
Any other relevant information
```

### Suggesting Features

Include:
1. Description of the feature
2. Use case / motivation
3. Proposed implementation (optional)
4. Alternatives considered

## Security

### Reporting Security Issues

**Do not open public issues for security vulnerabilities.**

Email security concerns to: [security@example.com]

Include:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Security Best Practices

- Never commit credentials
- Use Azure Key Vault for secrets
- Follow least privilege principle
- Keep dependencies updated
- Use managed identities where possible

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).

## Questions?

- Open an issue for bugs
- Start a discussion for questions
- Check existing issues and discussions first

Thank you for contributing! 🎉
