"""
Test Configuration and Setup
===========================

PyTest configuration and test utilities for the Content Recommendation Engine.

Author: Content Recommendation Engine Team
Date: October 2025
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_users_data():
    """Create sample user data for testing"""
    return pd.DataFrame({
        'user_id': range(100),
        'age': np.random.randint(18, 65, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'location': np.random.choice(['US', 'CA', 'UK', 'DE', 'FR'], 100),
        'subscription_type': np.random.choice(['free', 'premium', 'family'], 100),
        'signup_date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'last_activity': pd.date_range('2023-01-01', periods=100, freq='H')
    })


@pytest.fixture
def sample_items_data():
    """Create sample item data for testing"""
    return pd.DataFrame({
        'item_id': range(50),
        'title': [f'Movie {i}' for i in range(50)],
        'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance'], 50),
        'duration': np.random.randint(80, 180, 50),
        'release_year': np.random.randint(2000, 2024, 50),
        'rating': np.random.uniform(1.0, 5.0, 50),
        'director': [f'Director {i}' for i in range(50)],
        'language': np.random.choice(['English', 'Spanish', 'French', 'German'], 50)
    })


@pytest.fixture
def sample_interactions_data():
    """Create sample interaction data for testing"""
    return pd.DataFrame({
        'user_id': np.random.randint(0, 100, 1000),
        'item_id': np.random.randint(0, 50, 1000),
        'rating': np.random.randint(1, 6, 1000),
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),
        'interaction_type': np.random.choice(['view', 'like', 'share', 'comment'], 1000),
        'watch_duration': np.random.randint(0, 120, 1000)
    })


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_azure_storage():
    """Mock Azure Storage Blob client"""
    with patch('azure.storage.blob.BlobServiceClient') as mock_client:
        mock_container = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.get_container_client.return_value = mock_container
        mock_container.get_blob_client.return_value = mock_blob
        yield mock_client


@pytest.fixture
def mock_azure_ml():
    """Mock Azure ML client"""
    with patch('azure.ai.ml.MLClient') as mock_client:
        mock_workspace = MagicMock()
        mock_client.return_value = mock_workspace
        yield mock_client


@pytest.fixture
def mock_cognitive_search():
    """Mock Azure Cognitive Search client"""
    with patch('azure.search.documents.SearchClient') as mock_client:
        mock_search = MagicMock()
        mock_client.return_value = mock_search
        yield mock_client


@pytest.fixture
def mock_application_insights():
    """Mock Application Insights telemetry"""
    with patch('azure.monitor.opentelemetry.configure_azure_monitor') as mock_ai:
        yield mock_ai


@pytest.fixture
def sample_recommendation_request():
    """Sample recommendation request data"""
    return {
        'user_id': 'user_123',
        'num_recommendations': 10,
        'categories': ['Action', 'Comedy'],
        'exclude_watched': True,
        'include_metadata': True
    }


@pytest.fixture
def sample_search_request():
    """Sample search request data"""
    return {
        'query': 'action movies',
        'filters': {
            'genre': ['Action'],
            'release_year': {'min': 2020, 'max': 2024}
        },
        'sort': 'rating_desc',
        'size': 20,
        'from': 0
    }


@pytest.fixture
def mock_function_context():
    """Mock Azure Functions context"""
    context = MagicMock()
    context.function_name = 'test_function'
    context.function_directory = '/tmp/test'
    context.invocation_id = 'test_invocation_123'
    context.thread_local_storage = {}
    return context


@pytest.fixture
def mock_http_request():
    """Mock HTTP request for Azure Functions"""
    request = MagicMock()
    request.method = 'GET'
    request.url = 'https://test.azurewebsites.net/api/test'
    request.headers = {'Content-Type': 'application/json'}
    request.params = {}
    return request


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables"""
    test_env_vars = {
        'AZURE_STORAGE_CONNECTION_STRING': 'DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test;EndpointSuffix=core.windows.net',
        'AZURE_CLIENT_ID': 'test-client-id',
        'AZURE_CLIENT_SECRET': 'test-client-secret',
        'AZURE_TENANT_ID': 'test-tenant-id',
        'AZURE_SUBSCRIPTION_ID': 'test-subscription-id',
        'AZURE_RESOURCE_GROUP': 'test-rg',
        'AZURE_ML_WORKSPACE': 'test-ml-workspace',
        'AZURE_SEARCH_SERVICE': 'test-search-service',
        'AZURE_SEARCH_KEY': 'test-search-key',
        'APPLICATIONINSIGHTS_CONNECTION_STRING': 'InstrumentationKey=test-key',
        'ENVIRONMENT': 'test',
        'LOG_LEVEL': 'DEBUG'
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def disable_logging():
    """Disable logging during tests"""
    import logging
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "azure: mark test as requiring Azure services"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ['integration', 'slow', 'azure'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# Custom assertions
def assert_dataframe_equal(df1, df2, check_dtype=True, check_names=True):
    """Assert that two DataFrames are equal"""
    pd.testing.assert_frame_equal(
        df1, df2, 
        check_dtype=check_dtype, 
        check_names=check_names
    )


def assert_recommendations_valid(recommendations, user_id, num_recommendations):
    """Assert that recommendations are valid"""
    assert len(recommendations) <= num_recommendations
    assert all('item_id' in rec for rec in recommendations)
    assert all('score' in rec for rec in recommendations)
    assert all(isinstance(rec['score'], (int, float)) for rec in recommendations)
    
    # Check scores are in descending order
    scores = [rec['score'] for rec in recommendations]
    assert scores == sorted(scores, reverse=True)


def assert_search_results_valid(results, query, max_results=None):
    """Assert that search results are valid"""
    assert 'items' in results
    assert 'total' in results
    assert 'facets' in results
    
    if max_results:
        assert len(results['items']) <= max_results
    
    for item in results['items']:
        assert 'item_id' in item
        assert 'title' in item
        assert 'score' in item