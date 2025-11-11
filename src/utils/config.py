"""
Configuration Management for Content Recommendation Engine
=========================================================

This module handles configuration loading and management for the recommendation engine,
supporting different environments and Azure services integration.

Author: Content Recommendation Engine Team
Date: October 2025
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, field
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AzureConfig:
    """Azure service configuration"""
    subscription_id: str = ""
    resource_group: str = ""
    ml_workspace_name: str = ""
    data_lake_account_name: str = ""
    data_lake_container: str = "contentrec"
    key_vault_name: str = ""
    cognitive_search_service: str = ""
    cognitive_search_index: str = "content-index"
    storage_connection_string: str = ""

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    collaborative: Dict[str, Any] = field(default_factory=lambda: {
        'n_components': 50,
        'random_state': 42,
        'algorithm': 'randomized',
        'n_iter': 10
    })
    content: Dict[str, Any] = field(default_factory=lambda: {
        'max_features': 5000,
        'min_df': 5,
        'max_df': 0.8,
        'ngram_range': (1, 2),
        'stop_words': 'english'
    })
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {
        'collaborative': 0.7,
        'content': 0.3
    })
    evaluation: Dict[str, Any] = field(default_factory=lambda: {
        'test_size': 0.2,
        'random_state': 42
    })

@dataclass
class DataConfig:
    """Data processing configuration"""
    min_interactions: int = 5
    max_users: int = 100000
    max_items: int = 50000
    batch_size: int = 1000
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        'remove_duplicates': True,
        'filter_cold_start': True,
        'normalize_ratings': False
    })

@dataclass
class APIConfig:
    """API service configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    max_recommendations: int = 50
    cache_ttl: int = 3600  # 1 hour
    rate_limiting: Dict[str, Any] = field(default_factory=lambda: {
        'requests_per_minute': 100,
        'requests_per_hour': 1000
    })

@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    log_level: str = "INFO"
    application_insights_key: str = ""
    mlflow_tracking_uri: str = ""
    enable_telemetry: bool = True
    metrics_collection_interval: int = 60  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'error_rate': 0.05,
        'response_time_p95': 2.0,
        'recommendation_accuracy': 0.8
    })

class Config:
    """
    Main configuration class that loads and manages all configuration settings
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (dev, staging, prod)
        """
        self.environment = environment or os.getenv('ENVIRONMENT', 'dev')
        self.config_path = config_path
        
        # Initialize configuration sections
        self.azure = AzureConfig()
        self.model_config = ModelConfig()
        self.data = DataConfig()
        self.api = APIConfig()
        self.monitoring = MonitoringConfig()
        
        # Load configuration
        self._load_configuration()
        self._load_secrets()
        
        # Set derived properties
        self.experiment_name = f"content-recommendation-{self.environment}"
        self.model_version = "1.0.0"
        
    def _load_configuration(self):
        """Load configuration from files and environment variables"""
        # Load from configuration file if provided
        if self.config_path and os.path.exists(self.config_path):
            self._load_from_file(self.config_path)
        else:
            # Try to load default configuration files
            config_files = [
                f"config/{self.environment}.yaml",
                f"config/{self.environment}.json",
                "config/default.yaml",
                "config/default.json"
            ]
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    self._load_from_file(config_file)
                    break
        
        # Override with environment variables
        self._load_from_environment()
    
    def _load_from_file(self, file_path: str):
        """Load configuration from a file"""
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                elif file_path.endswith('.json'):
                    config_data = json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {file_path}")
                    return
            
            self._update_config_from_dict(config_data)
            logger.info(f"Loaded configuration from {file_path}")
            
        except Exception as e:
            logger.warning(f"Could not load configuration from {file_path}: {e}")
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # Azure configuration
            'AZURE_SUBSCRIPTION_ID': ('azure', 'subscription_id'),
            'AZURE_RESOURCE_GROUP': ('azure', 'resource_group'),
            'AZURE_ML_WORKSPACE': ('azure', 'ml_workspace_name'),
            'AZURE_DATA_LAKE_ACCOUNT': ('azure', 'data_lake_account_name'),
            'AZURE_DATA_LAKE_CONTAINER': ('azure', 'data_lake_container'),
            'AZURE_KEY_VAULT_NAME': ('azure', 'key_vault_name'),
            'AZURE_SEARCH_SERVICE': ('azure', 'cognitive_search_service'),
            'AZURE_SEARCH_INDEX': ('azure', 'cognitive_search_index'),
            'AZURE_STORAGE_CONNECTION_STRING': ('azure', 'storage_connection_string'),
            
            # API configuration
            'API_HOST': ('api', 'host'),
            'API_PORT': ('api', 'port'),
            'API_WORKERS': ('api', 'workers'),
            
            # Monitoring configuration
            'LOG_LEVEL': ('monitoring', 'log_level'),
            'APPLICATION_INSIGHTS_KEY': ('monitoring', 'application_insights_key'),
            'MLFLOW_TRACKING_URI': ('monitoring', 'mlflow_tracking_uri'),
            
            # Data configuration
            'MIN_INTERACTIONS': ('data', 'min_interactions'),
            'BATCH_SIZE': ('data', 'batch_size'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_config_value(section, key, value)
    
    def _load_secrets(self):
        """Load secrets from Azure Key Vault"""
        if not self.azure.key_vault_name:
            logger.info("No Key Vault configured, skipping secret loading")
            return
        
        try:
            credential = DefaultAzureCredential()
            vault_url = f"https://{self.azure.key_vault_name}.vault.azure.net/"
            client = SecretClient(vault_url=vault_url, credential=credential)
            
            # Load common secrets
            secret_mappings = {
                'storage-connection-string': ('azure', 'storage_connection_string'),
                'application-insights-key': ('monitoring', 'application_insights_key'),
                'mlflow-tracking-uri': ('monitoring', 'mlflow_tracking_uri'),
            }
            
            for secret_name, (section, key) in secret_mappings.items():
                try:
                    secret = client.get_secret(secret_name)
                    self._set_config_value(section, key, secret.value)
                except Exception as e:
                    logger.warning(f"Could not load secret {secret_name}: {e}")
            
            logger.info("Loaded secrets from Key Vault")
            
        except Exception as e:
            logger.warning(f"Could not access Key Vault: {e}")
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def _set_config_value(self, section_name: str, key: str, value: str):
        """Set a configuration value with proper type conversion"""
        if not hasattr(self, section_name):
            return
        
        section = getattr(self, section_name)
        if not hasattr(section, key):
            return
        
        # Get the current value to determine the expected type
        current_value = getattr(section, key)
        
        # Convert value to the appropriate type
        try:
            if isinstance(current_value, bool):
                converted_value = value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(current_value, int):
                converted_value = int(value)
            elif isinstance(current_value, float):
                converted_value = float(value)
            else:
                converted_value = value
            
            setattr(section, key, converted_value)
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert {value} for {section_name}.{key}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': self.environment,
            'azure': self.azure.__dict__,
            'model_config': {
                'collaborative': self.model_config.collaborative,
                'content': self.model_config.content,
                'hybrid_weights': self.model_config.hybrid_weights,
                'evaluation': self.model_config.evaluation
            },
            'data': self.data.__dict__,
            'api': self.api.__dict__,
            'monitoring': self.monitoring.__dict__
        }
    
    def save_to_file(self, file_path: str):
        """Save configuration to file"""
        config_dict = self.to_dict()
        
        with open(file_path, 'w') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                yaml.dump(config_dict, f, default_flow_style=False)
            elif file_path.endswith('.json'):
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError("Unsupported file format")
        
        logger.info(f"Configuration saved to {file_path}")
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate Azure configuration
        if self.environment == 'prod':
            if not self.azure.subscription_id:
                errors.append("Azure subscription ID is required for production")
            if not self.azure.resource_group:
                errors.append("Azure resource group is required for production")
            if not self.azure.ml_workspace_name:
                errors.append("Azure ML workspace name is required for production")
        
        # Validate model configuration
        if self.model_config.collaborative['n_components'] <= 0:
            errors.append("SVD components must be positive")
        
        if sum(self.model_config.hybrid_weights.values()) != 1.0:
            errors.append("Hybrid weights must sum to 1.0")
        
        # Validate data configuration
        if self.data.min_interactions < 1:
            errors.append("Minimum interactions must be at least 1")
        
        # Validate API configuration
        if self.api.port < 1 or self.api.port > 65535:
            errors.append("API port must be between 1 and 65535")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    @property
    def mlflow_tracking_uri(self) -> str:
        """Get MLflow tracking URI"""
        return self.monitoring.mlflow_tracking_uri or "file:./mlruns"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == 'prod'
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == 'dev'


def load_config(config_path: str = None, environment: str = None) -> Config:
    """
    Factory function to load configuration
    
    Args:
        config_path: Path to configuration file
        environment: Environment name
        
    Returns:
        Configured Config instance
    """
    config = Config(config_path, environment)
    
    if not config.validate():
        logger.warning("Configuration validation failed, but continuing...")
    
    return config


if __name__ == "__main__":
    # Example usage
    config = load_config()
    
    print(f"Environment: {config.environment}")
    print(f"Azure ML Workspace: {config.azure.ml_workspace_name}")
    print(f"Model Components: {config.model_config.collaborative['n_components']}")
    print(f"API Port: {config.api.port}")
    
    # Save example configuration
    config.save_to_file("config/example.yaml")