"""
Deployment Automation Scripts
=============================

Automated deployment scripts for the Azure-native content recommendation engine
using Azure CLI, Bicep templates, and configuration management.

Author: Content Recommendation Engine Team
Date: October 2025
"""

import os
import json
import logging
import subprocess
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """
    Manages the automated deployment of the recommendation engine infrastructure
    """
    
    def __init__(self, config_path: str = "deployment_config.yaml"):
        """Initialize the deployment manager"""
        self.config_path = config_path
        self.config = self._load_config()
        self.deployment_log = []
        
        # Deployment paths
        self.project_root = Path(__file__).parent.parent.parent
        self.infrastructure_path = self.project_root / "infrastructure"
        self.src_path = self.project_root / "src"
        
        logger.info("DeploymentManager initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        return yaml.safe_load(f)
                    else:
                        return json.load(f)
            else:
                # Return default configuration
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default deployment configuration"""
        return {
            "environment": "dev",
            "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
            "resource_group": "rg-recommendation-engine-dev",
            "location": "East US 2",
            "deployment_name": "recommendation-engine-deployment",
            "tags": {
                "project": "content-recommendation-engine",
                "environment": "dev",
                "owner": "dev-team"
            },
            "parameters": {
                "environmentName": "dev",
                "location": "eastus2",
                "projectName": "recengine"
            },
            "function_apps": [
                {
                    "name": "recommendation-api",
                    "source_path": "src/api",
                    "runtime": "python",
                    "version": "3.9"
                },
                {
                    "name": "search-api",
                    "source_path": "src/search",
                    "runtime": "python",
                    "version": "3.9"
                },
                {
                    "name": "monitoring-api",
                    "source_path": "src/monitoring",
                    "runtime": "python",
                    "version": "3.9"
                }
            ],
            "ml_models": {
                "deployment_endpoint": "recommendation-models",
                "model_path": "src/ml/models",
                "compute_instance": "ml-compute-instance"
            },
            "post_deployment": {
                "run_tests": True,
                "upload_sample_data": True,
                "configure_monitoring": True,
                "setup_alerts": True
            }
        }

    def _run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and log the result"""
        logger.info(f"Executing: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=check
            )
            
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Stderr: {result.stderr}")
            
            self.deployment_log.append({
                "timestamp": datetime.now().isoformat(),
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            })
            
            return result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            self.deployment_log.append({
                "timestamp": datetime.now().isoformat(),
                "command": command,
                "returncode": e.returncode,
                "error": str(e)
            })
            raise

    def check_prerequisites(self) -> bool:
        """Check deployment prerequisites"""
        logger.info("Checking deployment prerequisites...")
        
        try:
            # Check Azure CLI
            result = self._run_command("az --version", check=False)
            if result.returncode != 0:
                logger.error("Azure CLI not found. Please install Azure CLI.")
                return False
            
            # Check login status
            result = self._run_command("az account show", check=False)
            if result.returncode != 0:
                logger.error("Not logged into Azure. Please run 'az login'.")
                return False
            
            # Check subscription
            if self.config.get("subscription_id"):
                self._run_command(f"az account set --subscription {self.config['subscription_id']}")
            
            # Check required extensions
            extensions = ["application-insights", "ml"]
            for ext in extensions:
                self._run_command(f"az extension add --name {ext} --only-show-errors", check=False)
            
            logger.info("Prerequisites check completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Prerequisites check failed: {e}")
            return False

    def create_resource_group(self) -> bool:
        """Create Azure resource group"""
        try:
            logger.info(f"Creating resource group: {self.config['resource_group']}")
            
            # Check if resource group exists
            result = self._run_command(
                f"az group show --name {self.config['resource_group']}", 
                check=False
            )
            
            if result.returncode == 0:
                logger.info("Resource group already exists")
                return True
            
            # Create resource group
            tags = " ".join([f"{k}={v}" for k, v in self.config.get("tags", {}).items()])
            
            command = (
                f"az group create "
                f"--name {self.config['resource_group']} "
                f"--location \"{self.config['location']}\""
            )
            
            if tags:
                command += f" --tags {tags}"
            
            self._run_command(command)
            
            logger.info("Resource group created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create resource group: {e}")
            return False

    def deploy_infrastructure(self) -> bool:
        """Deploy infrastructure using Bicep templates"""
        try:
            logger.info("Deploying infrastructure...")
            
            # Main Bicep template path
            main_template = self.infrastructure_path / "main.bicep"
            
            if not main_template.exists():
                logger.error(f"Main Bicep template not found: {main_template}")
                return False
            
            # Prepare parameters
            parameters = self.config.get("parameters", {})
            param_string = " ".join([
                f"{key}={value}" for key, value in parameters.items()
            ])
            
            # Deploy infrastructure
            command = (
                f"az deployment group create "
                f"--resource-group {self.config['resource_group']} "
                f"--name {self.config['deployment_name']} "
                f"--template-file {main_template} "
                f"--parameters {param_string}"
            )
            
            self._run_command(command)
            
            logger.info("Infrastructure deployment completed")
            return True
            
        except Exception as e:
            logger.error(f"Infrastructure deployment failed: {e}")
            return False

    def deploy_function_apps(self) -> bool:
        """Deploy Azure Function Apps"""
        try:
            logger.info("Deploying Function Apps...")
            
            for func_app in self.config.get("function_apps", []):
                logger.info(f"Deploying Function App: {func_app['name']}")
                
                # Create deployment package
                source_path = self.project_root / func_app["source_path"]
                if not source_path.exists():
                    logger.warning(f"Source path not found: {source_path}")
                    continue
                
                # Create zip package
                package_path = f"{func_app['name']}-package.zip"
                self._create_function_package(source_path, package_path)
                
                # Get function app name (may include environment prefix)
                func_app_name = f"{self.config['parameters']['projectName']}-{func_app['name']}-{self.config['environment']}"
                
                # Deploy function app
                deploy_command = (
                    f"az functionapp deployment source config-zip "
                    f"--resource-group {self.config['resource_group']} "
                    f"--name {func_app_name} "
                    f"--src {package_path}"
                )
                
                self._run_command(deploy_command)
                
                # Configure app settings
                self._configure_function_app_settings(func_app_name)
                
                # Clean up package
                if os.path.exists(package_path):
                    os.remove(package_path)
            
            logger.info("Function Apps deployment completed")
            return True
            
        except Exception as e:
            logger.error(f"Function Apps deployment failed: {e}")
            return False

    def _create_function_package(self, source_path: Path, package_path: str):
        """Create deployment package for Function App"""
        import zipfile
        
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_path.rglob('*'):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    arcname = file_path.relative_to(source_path)
                    zipf.write(file_path, arcname)
        
        logger.info(f"Created function package: {package_path}")

    def _configure_function_app_settings(self, func_app_name: str):
        """Configure Function App application settings"""
        try:
            # Common settings
            settings = {
                "FUNCTIONS_WORKER_RUNTIME": "python",
                "FUNCTIONS_EXTENSION_VERSION": "~4",
                "WEBSITE_RUN_FROM_PACKAGE": "1",
                "PYTHON_ENABLE_WORKER_EXTENSIONS": "1",
                "AzureWebJobsFeatureFlags": "EnableWorkerIndexing"
            }
            
            # Environment-specific settings
            env_settings = self._get_environment_settings()
            settings.update(env_settings)
            
            # Apply settings
            for key, value in settings.items():
                command = (
                    f"az functionapp config appsettings set "
                    f"--resource-group {self.config['resource_group']} "
                    f"--name {func_app_name} "
                    f"--settings {key}={value}"
                )
                self._run_command(command, check=False)
            
            logger.info(f"Configured settings for {func_app_name}")
            
        except Exception as e:
            logger.error(f"Failed to configure Function App settings: {e}")

    def _get_environment_settings(self) -> Dict[str, str]:
        """Get environment-specific application settings"""
        return {
            "ENVIRONMENT": self.config["environment"],
            "LOG_LEVEL": "INFO" if self.config["environment"] == "prod" else "DEBUG",
            "CACHE_TTL_MINUTES": "30",
            "MAX_RECOMMENDATIONS": "50",
            "ENABLE_MONITORING": "true",
            "AZURE_FUNCTIONS_ENVIRONMENT": self.config["environment"]
        }

    def deploy_ml_models(self) -> bool:
        """Deploy ML models to Azure ML"""
        try:
            logger.info("Deploying ML models...")
            
            ml_config = self.config.get("ml_models", {})
            
            # Get ML workspace details from deployment output
            workspace_name = f"{self.config['parameters']['projectName']}-ml-{self.config['environment']}"
            
            # Register model
            model_path = self.project_root / ml_config.get("model_path", "src/ml/models")
            if model_path.exists():
                command = (
                    f"az ml model create "
                    f"--resource-group {self.config['resource_group']} "
                    f"--workspace-name {workspace_name} "
                    f"--name recommendation-model "
                    f"--version 1 "
                    f"--path {model_path}"
                )
                self._run_command(command, check=False)
            
            # Create endpoint
            endpoint_name = ml_config.get("deployment_endpoint", "recommendation-models")
            command = (
                f"az ml online-endpoint create "
                f"--resource-group {self.config['resource_group']} "
                f"--workspace-name {workspace_name} "
                f"--name {endpoint_name} "
                f"--auth-mode key"
            )
            self._run_command(command, check=False)
            
            logger.info("ML models deployment completed")
            return True
            
        except Exception as e:
            logger.error(f"ML models deployment failed: {e}")
            return False

    def upload_sample_data(self) -> bool:
        """Upload sample data to storage accounts"""
        try:
            if not self.config.get("post_deployment", {}).get("upload_sample_data", False):
                return True
            
            logger.info("Uploading sample data...")
            
            # Generate sample data if not exists
            sample_data_path = self.project_root / "sample_data"
            if not sample_data_path.exists():
                logger.info("Generating sample data...")
                from data.sample_data_generator import generate_complete_dataset
                generate_complete_dataset(output_dir=str(sample_data_path))
            
            # Get storage account name
            storage_account = f"{self.config['parameters']['projectName']}storage{self.config['environment']}"
            
            # Upload data files
            for data_file in sample_data_path.glob("*.json"):
                command = (
                    f"az storage blob upload "
                    f"--account-name {storage_account} "
                    f"--container-name sample-data "
                    f"--name {data_file.name} "
                    f"--file {data_file} "
                    f"--auth-mode login"
                )
                self._run_command(command, check=False)
            
            logger.info("Sample data upload completed")
            return True
            
        except Exception as e:
            logger.error(f"Sample data upload failed: {e}")
            return False

    def configure_monitoring(self) -> bool:
        """Configure monitoring and alerting"""
        try:
            if not self.config.get("post_deployment", {}).get("configure_monitoring", False):
                return True
            
            logger.info("Configuring monitoring...")
            
            # Get Application Insights details
            app_insights_name = f"{self.config['parameters']['projectName']}-ai-{self.config['environment']}"
            
            # Configure alerts (example)
            alert_rules = [
                {
                    "name": "high-error-rate",
                    "description": "High error rate detected",
                    "condition": "requests/count > 100 and requests/failed > 5",
                    "severity": 2
                },
                {
                    "name": "slow-response-time",
                    "description": "Slow response time detected",
                    "condition": "requests/duration > 2000",
                    "severity": 3
                }
            ]
            
            for rule in alert_rules:
                # Create metric alert (simplified)
                command = (
                    f"az monitor metrics alert create "
                    f"--resource-group {self.config['resource_group']} "
                    f"--name {rule['name']} "
                    f"--description \"{rule['description']}\" "
                    f"--severity {rule['severity']} "
                    f"--scopes /subscriptions/{self.config['subscription_id']}/resourceGroups/{self.config['resource_group']}/providers/Microsoft.Insights/components/{app_insights_name}"
                )
                self._run_command(command, check=False)
            
            logger.info("Monitoring configuration completed")
            return True
            
        except Exception as e:
            logger.error(f"Monitoring configuration failed: {e}")
            return False

    def run_deployment_tests(self) -> bool:
        """Run post-deployment tests"""
        try:
            if not self.config.get("post_deployment", {}).get("run_tests", False):
                return True
            
            logger.info("Running deployment tests...")
            
            # Get function app URLs and test endpoints
            func_apps = self.config.get("function_apps", [])
            
            for func_app in func_apps:
                func_app_name = f"{self.config['parameters']['projectName']}-{func_app['name']}-{self.config['environment']}"
                
                # Get function app URL
                result = self._run_command(
                    f"az functionapp show --resource-group {self.config['resource_group']} --name {func_app_name} --query defaultHostName -o tsv",
                    check=False
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    url = f"https://{result.stdout.strip()}"
                    logger.info(f"Function app URL: {url}")
                    
                    # Test health endpoint
                    import requests
                    try:
                        response = requests.get(f"{url}/api/health", timeout=30)
                        if response.status_code == 200:
                            logger.info(f"Health check passed for {func_app_name}")
                        else:
                            logger.warning(f"Health check failed for {func_app_name}: {response.status_code}")
                    except Exception as e:
                        logger.warning(f"Failed to test {func_app_name}: {e}")
            
            logger.info("Deployment tests completed")
            return True
            
        except Exception as e:
            logger.error(f"Deployment tests failed: {e}")
            return False

    def save_deployment_output(self) -> bool:
        """Save deployment outputs and configuration"""
        try:
            logger.info("Saving deployment output...")
            
            # Get deployment outputs
            result = self._run_command(
                f"az deployment group show --resource-group {self.config['resource_group']} --name {self.config['deployment_name']} --query properties.outputs",
                check=False
            )
            
            outputs = {}
            if result.returncode == 0 and result.stdout.strip():
                try:
                    outputs = json.loads(result.stdout)
                except json.JSONDecodeError:
                    pass
            
            # Create deployment summary
            deployment_summary = {
                "deployment_timestamp": datetime.now().isoformat(),
                "environment": self.config["environment"],
                "resource_group": self.config["resource_group"],
                "location": self.config["location"],
                "deployment_name": self.config["deployment_name"],
                "outputs": outputs,
                "configuration": self.config,
                "deployment_log": self.deployment_log
            }
            
            # Save to file
            output_file = f"deployment_output_{self.config['environment']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(deployment_summary, f, indent=2, default=str)
            
            logger.info(f"Deployment output saved to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save deployment output: {e}")
            return False

    def deploy_complete_solution(self) -> bool:
        """Deploy the complete recommendation engine solution"""
        logger.info("Starting complete solution deployment...")
        
        start_time = time.time()
        
        try:
            # Step 1: Check prerequisites
            if not self.check_prerequisites():
                logger.error("Prerequisites check failed")
                return False
            
            # Step 2: Create resource group
            if not self.create_resource_group():
                logger.error("Resource group creation failed")
                return False
            
            # Step 3: Deploy infrastructure
            if not self.deploy_infrastructure():
                logger.error("Infrastructure deployment failed")
                return False
            
            # Wait for infrastructure deployment to complete
            logger.info("Waiting for infrastructure deployment to stabilize...")
            time.sleep(60)
            
            # Step 4: Deploy Function Apps
            if not self.deploy_function_apps():
                logger.error("Function Apps deployment failed")
                return False
            
            # Step 5: Deploy ML models
            if not self.deploy_ml_models():
                logger.warning("ML models deployment failed (non-critical)")
            
            # Step 6: Upload sample data
            if not self.upload_sample_data():
                logger.warning("Sample data upload failed (non-critical)")
            
            # Step 7: Configure monitoring
            if not self.configure_monitoring():
                logger.warning("Monitoring configuration failed (non-critical)")
            
            # Step 8: Run tests
            if not self.run_deployment_tests():
                logger.warning("Deployment tests failed (non-critical)")
            
            # Step 9: Save deployment output
            self.save_deployment_output()
            
            end_time = time.time()
            duration = round((end_time - start_time) / 60, 2)
            
            logger.info(f"Complete solution deployment completed successfully in {duration} minutes!")
            return True
            
        except Exception as e:
            logger.error(f"Complete solution deployment failed: {e}")
            return False

def create_deployment_config(environment: str = "dev") -> str:
    """Create deployment configuration file"""
    config = {
        "environment": environment,
        "subscription_id": "${AZURE_SUBSCRIPTION_ID}",
        "resource_group": f"rg-recommendation-engine-{environment}",
        "location": "East US 2",
        "deployment_name": f"recommendation-engine-{environment}",
        "tags": {
            "project": "content-recommendation-engine",
            "environment": environment,
            "owner": "dev-team",
            "cost-center": "engineering"
        },
        "parameters": {
            "environmentName": environment,
            "location": "eastus2",
            "projectName": "recengine"
        },
        "function_apps": [
            {
                "name": "recommendation-api",
                "source_path": "src/api",
                "runtime": "python",
                "version": "3.9"
            },
            {
                "name": "search-api",
                "source_path": "src/search",
                "runtime": "python",
                "version": "3.9"
            },
            {
                "name": "monitoring-api",
                "source_path": "src/monitoring",
                "runtime": "python",
                "version": "3.9"
            }
        ],
        "ml_models": {
            "deployment_endpoint": "recommendation-models",
            "model_path": "src/ml/models",
            "compute_instance": "ml-compute-instance"
        },
        "post_deployment": {
            "run_tests": True,
            "upload_sample_data": True,
            "configure_monitoring": True,
            "setup_alerts": True
        }
    }
    
    filename = f"deployment_config_{environment}.yaml"
    with open(filename, 'w') as f:
        yaml.dump(config, f, indent=2, default_flow_style=False)
    
    logger.info(f"Created deployment configuration: {filename}")
    return filename

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Content Recommendation Engine")
    parser.add_argument("--environment", "-e", default="dev", help="Deployment environment")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--create-config", action="store_true", help="Create configuration file")
    
    args = parser.parse_args()
    
    if args.create_config:
        config_file = create_deployment_config(args.environment)
        print(f"Configuration file created: {config_file}")
    else:
        config_path = args.config or f"deployment_config_{args.environment}.yaml"
        
        # Create config if it doesn't exist
        if not os.path.exists(config_path):
            config_path = create_deployment_config(args.environment)
        
        # Deploy solution
        deployment_manager = DeploymentManager(config_path)
        success = deployment_manager.deploy_complete_solution()
        
        if success:
            print("✅ Deployment completed successfully!")
        else:
            print("❌ Deployment failed!")
            exit(1)