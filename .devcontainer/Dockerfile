# .devcontainer/Dockerfile
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/devcontainers/python:${VARIANT}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The base image already includes most tools we need
# Only add what's absolutely necessary

# Switch to vscode user for pip installs
USER vscode

# Upgrade pip and install Python development tools
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    black \
    isort \
    pylint \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-mock \
    bandit \
    safety \
    pre-commit \
    ipython \
    ipykernel \
    debugpy

# Set working directory
WORKDIR /workspaces

# Keep container running
CMD ["sleep", "infinity"]
