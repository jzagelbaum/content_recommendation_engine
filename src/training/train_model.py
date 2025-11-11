"""
Azure ML Training Script for Content Recommendation Engine
==========================================================

This script trains the recommendation models using Azure Machine Learning,
with MLflow for experiment tracking and model registration.

Author: Content Recommendation Engine Team
Date: October 2025
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential
from azureml.core import Run
import joblib
import json
from datetime import datetime
import logging

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from models.recommendation_engine import ContentRecommendationEngine, log_experiment_to_mlflow
from data.data_loader import DataLoader
from utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureMLTrainer:
    """
    Azure ML trainer for the content recommendation engine
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Azure ML trainer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path) if config_path else Config()
        self.ml_client = None
        self.run = None
        self.setup_azure_ml()
    
    def setup_azure_ml(self):
        """Setup Azure ML client and get current run context"""
        try:
            # Try to get the current run context (when running in Azure ML)
            self.run = Run.get_context()
            logger.info("Running in Azure ML context")
        except Exception:
            logger.info("Running outside Azure ML context")
            self.run = None
        
        # Setup MLflow tracking
        if self.config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        
        mlflow.set_experiment(self.config.experiment_name)
    
    def load_data(self, data_path: str = None) -> tuple:
        """
        Load training data from Azure Data Lake or local path
        
        Args:
            data_path: Path to data directory
            
        Returns:
            Tuple of (interactions_df, content_df)
        """
        logger.info("Loading training data...")
        
        data_loader = DataLoader(self.config)
        
        if data_path:
            # Load from local path
            interactions_df = pd.read_parquet(f"{data_path}/interactions.parquet")
            content_df = pd.read_parquet(f"{data_path}/content.parquet")
        else:
            # Load from Azure Data Lake
            interactions_df = data_loader.load_interactions_data()
            content_df = data_loader.load_content_data()
        
        logger.info(f"Loaded {len(interactions_df)} interactions and {len(content_df)} content items")
        
        return interactions_df, content_df
    
    def train_model(self, interactions_df: pd.DataFrame, content_df: pd.DataFrame) -> ContentRecommendationEngine:
        """
        Train the recommendation model
        
        Args:
            interactions_df: User-item interactions data
            content_df: Content metadata
            
        Returns:
            Trained recommendation engine
        """
        logger.info("Training recommendation model...")
        
        # Initialize recommendation engine
        engine = ContentRecommendationEngine(self.config.model_config)
        
        # Prepare data
        user_item_matrix, content_features = engine.prepare_data(interactions_df, content_df)
        
        # Train collaborative filtering
        svd_model = engine.train_collaborative_filtering(user_item_matrix)
        
        # Train content-based filtering
        tfidf_model = engine.train_content_based_filtering(content_features)
        
        logger.info("Model training completed")
        
        return engine
    
    def evaluate_model(self, engine: ContentRecommendationEngine, interactions_df: pd.DataFrame) -> dict:
        """
        Evaluate the trained model
        
        Args:
            engine: Trained recommendation engine
            interactions_df: Evaluation data
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model...")
        
        metrics = engine.evaluate_model(interactions_df)
        
        # Log metrics to Azure ML if running in Azure ML context
        if self.run and hasattr(self.run, 'log'):
            for metric_name, metric_value in metrics.items():
                self.run.log(metric_name, metric_value)
        
        logger.info(f"Evaluation completed: {metrics}")
        
        return metrics
    
    def save_and_register_model(self, engine: ContentRecommendationEngine, metrics: dict) -> str:
        """
        Save model and register it in Azure ML
        
        Args:
            engine: Trained recommendation engine
            metrics: Evaluation metrics
            
        Returns:
            Path to saved model
        """
        logger.info("Saving and registering model...")
        
        # Create outputs directory
        outputs_dir = Path("./outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = outputs_dir / "recommendation_model.joblib"
        metadata = {
            "training_date": datetime.now().isoformat(),
            "metrics": metrics,
            "config": self.config.model_config,
            "model_version": self.config.model_version
        }
        
        engine.save_model(str(model_path), metadata)
        
        # Save model artifacts for Azure ML
        if self.run and hasattr(self.run, 'upload_file'):
            self.run.upload_file("recommendation_model.joblib", str(model_path))
            
            # Upload model config
            config_path = outputs_dir / "model_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config.model_config, f, indent=2)
            self.run.upload_file("model_config.json", str(config_path))
        
        logger.info(f"Model saved to {model_path}")
        
        return str(model_path)
    
    def run_training_pipeline(self, data_path: str = None):
        """
        Run the complete training pipeline
        
        Args:
            data_path: Optional path to training data
        """
        try:
            with mlflow.start_run() as run:
                logger.info(f"Starting training pipeline - MLflow run: {run.info.run_id}")
                
                # Load data
                interactions_df, content_df = self.load_data(data_path)
                
                # Log data statistics
                mlflow.log_metric("num_interactions", len(interactions_df))
                mlflow.log_metric("num_users", interactions_df['user_id'].nunique())
                mlflow.log_metric("num_items", interactions_df['item_id'].nunique())
                mlflow.log_metric("num_content_items", len(content_df))
                
                # Train model
                engine = self.train_model(interactions_df, content_df)
                
                # Evaluate model
                metrics = self.evaluate_model(engine, interactions_df)
                
                # Save and register model
                model_path = self.save_and_register_model(engine, metrics)
                
                # Log to MLflow
                log_experiment_to_mlflow(engine, metrics, self.config.model_config, model_path)
                
                logger.info("Training pipeline completed successfully")
                
                return {
                    "model_path": model_path,
                    "metrics": metrics,
                    "mlflow_run_id": run.info.run_id
                }
                
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Content Recommendation Engine")
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to training data directory"
    )
    
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="content-recommendation-experiment",
        help="MLflow experiment name"
    )
    
    parser.add_argument(
        "--model-version",
        type=str,
        default="1.0.0",
        help="Model version"
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_arguments()
    
    logger.info("Starting Content Recommendation Engine training")
    logger.info(f"Arguments: {args}")
    
    # Initialize trainer
    trainer = AzureMLTrainer(args.config_path)
    
    # Override config with command line arguments
    if args.experiment_name:
        trainer.config.experiment_name = args.experiment_name
    if args.model_version:
        trainer.config.model_version = args.model_version
    
    # Run training pipeline
    results = trainer.run_training_pipeline(args.data_path)
    
    logger.info("Training completed successfully")
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()