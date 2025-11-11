"""
Content Recommendation Engine - Core Models
============================================

This module contains the main recommendation algorithms for the content recommendation engine:
- Collaborative Filtering (Matrix Factorization with SVD)
- Content-Based Filtering (TF-IDF + Cosine Similarity)
- Hybrid Model (Weighted combination)
- Deep Learning Model (Neural Collaborative Filtering)

Author: Content Recommendation Engine Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn
import mlflow.keras
from scipy.sparse import csr_matrix
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class ContentRecommendationEngine:
    """
    Main recommendation engine that combines multiple algorithms
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the recommendation engine
        
        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = config or self._get_default_config()
        self.collaborative_model = None
        self.content_model = None
        self.hybrid_weights = self.config.get('hybrid_weights', {'collaborative': 0.7, 'content': 0.3})
        self.user_item_matrix = None
        self.content_features = None
        self.item_features = None
        self.model_metadata = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the recommendation engine"""
        return {
            'collaborative': {
                'n_components': 50,
                'random_state': 42,
                'algorithm': 'randomized',
                'n_iter': 10
            },
            'content': {
                'max_features': 5000,
                'min_df': 5,
                'max_df': 0.8,
                'ngram_range': (1, 2),
                'stop_words': 'english'
            },
            'hybrid_weights': {
                'collaborative': 0.7,
                'content': 0.3
            },
            'evaluation': {
                'test_size': 0.2,
                'random_state': 42
            }
        }
    
    def prepare_data(self, interactions_df: pd.DataFrame, content_df: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame]:
        """
        Prepare data for training recommendation models
        
        Args:
            interactions_df: DataFrame with user-item interactions (user_id, item_id, rating, timestamp)
            content_df: DataFrame with item content features (item_id, title, description, genre, etc.)
            
        Returns:
            Tuple of user-item matrix and processed content features
        """
        logger.info("Preparing data for recommendation models...")
        
        # Create user-item interaction matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating',
            fill_value=0
        )
        
        # Convert to sparse matrix for memory efficiency
        sparse_matrix = csr_matrix(self.user_item_matrix.values)
        
        # Prepare content features
        self.content_features = content_df.copy()
        
        # Combine text features for content-based filtering
        self.content_features['combined_features'] = (
            self.content_features.get('title', '').fillna('') + ' ' +
            self.content_features.get('description', '').fillna('') + ' ' +
            self.content_features.get('genre', '').fillna('') + ' ' +
            self.content_features.get('tags', '').fillna('')
        )
        
        logger.info(f"Prepared data: {self.user_item_matrix.shape[0]} users, {self.user_item_matrix.shape[1]} items")
        
        return sparse_matrix, self.content_features
    
    def train_collaborative_filtering(self, user_item_matrix: csr_matrix) -> TruncatedSVD:
        """
        Train collaborative filtering model using SVD
        
        Args:
            user_item_matrix: Sparse user-item interaction matrix
            
        Returns:
            Trained SVD model
        """
        logger.info("Training collaborative filtering model...")
        
        config = self.config['collaborative']
        
        # Initialize and train SVD model
        svd_model = TruncatedSVD(
            n_components=config['n_components'],
            random_state=config['random_state'],
            algorithm=config['algorithm'],
            n_iter=config['n_iter']
        )
        
        # Fit the model
        user_factors = svd_model.fit_transform(user_item_matrix)
        item_factors = svd_model.components_
        
        # Store model components
        self.collaborative_model = {
            'svd': svd_model,
            'user_factors': user_factors,
            'item_factors': item_factors,
            'explained_variance_ratio': svd_model.explained_variance_ratio_.sum()
        }
        
        logger.info(f"Collaborative filtering trained - Explained variance: {self.collaborative_model['explained_variance_ratio']:.3f}")
        
        return svd_model
    
    def train_content_based_filtering(self, content_features: pd.DataFrame) -> TfidfVectorizer:
        """
        Train content-based filtering model using TF-IDF
        
        Args:
            content_features: DataFrame with item content features
            
        Returns:
            Trained TF-IDF vectorizer
        """
        logger.info("Training content-based filtering model...")
        
        config = self.config['content']
        
        # Initialize TF-IDF vectorizer
        tfidf = TfidfVectorizer(
            max_features=config['max_features'],
            min_df=config['min_df'],
            max_df=config['max_df'],
            ngram_range=config['ngram_range'],
            stop_words=config['stop_words']
        )
        
        # Fit TF-IDF on combined features
        tfidf_matrix = tfidf.fit_transform(content_features['combined_features'])
        
        # Calculate item similarity matrix
        item_similarity = cosine_similarity(tfidf_matrix)
        
        # Store model components
        self.content_model = {
            'tfidf': tfidf,
            'tfidf_matrix': tfidf_matrix,
            'item_similarity': item_similarity,
            'feature_names': tfidf.get_feature_names_out()
        }
        
        logger.info(f"Content-based filtering trained - Feature matrix shape: {tfidf_matrix.shape}")
        
        return tfidf
    
    def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Get recommendations using collaborative filtering
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if self.collaborative_model is None:
            raise ValueError("Collaborative filtering model not trained")
        
        try:
            # Get user index in the matrix
            user_idx = self.user_item_matrix.index.get_loc(user_id)
        except KeyError:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        # Get user factors and predict ratings for all items
        user_vector = self.collaborative_model['user_factors'][user_idx:user_idx+1]
        predicted_ratings = user_vector.dot(self.collaborative_model['item_factors'])
        
        # Get items the user hasn't interacted with
        user_interactions = self.user_item_matrix.iloc[user_idx]
        unrated_items = user_interactions[user_interactions == 0].index
        
        # Get predictions for unrated items
        item_predictions = []
        for item_id in unrated_items:
            try:
                item_idx = self.user_item_matrix.columns.get_loc(item_id)
                predicted_rating = predicted_ratings[0, item_idx]
                item_predictions.append((item_id, predicted_rating))
            except KeyError:
                continue
        
        # Sort by predicted rating and return top N
        item_predictions.sort(key=lambda x: x[1], reverse=True)
        return item_predictions[:n_recommendations]
    
    def get_content_based_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Get recommendations using content-based filtering
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if self.content_model is None:
            raise ValueError("Content-based filtering model not trained")
        
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
        except KeyError:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        # Get user's interaction history
        user_interactions = self.user_item_matrix.iloc[user_idx]
        liked_items = user_interactions[user_interactions > 3].index  # Items rated > 3
        
        if len(liked_items) == 0:
            return []
        
        # Calculate user profile based on liked items
        item_similarities = self.content_model['item_similarity']
        user_profile_scores = np.zeros(len(self.content_features))
        
        for item_id in liked_items:
            try:
                item_idx = self.content_features[self.content_features.index == item_id].index[0]
                item_idx_pos = self.content_features.index.get_loc(item_idx)
                user_profile_scores += item_similarities[item_idx_pos] * user_interactions[item_id]
            except (KeyError, IndexError):
                continue
        
        # Get recommendations for unrated items
        unrated_items = user_interactions[user_interactions == 0].index
        recommendations = []
        
        for item_id in unrated_items:
            try:
                item_idx = self.content_features[self.content_features.index == item_id].index[0]
                item_idx_pos = self.content_features.index.get_loc(item_idx)
                similarity_score = user_profile_scores[item_idx_pos]
                recommendations.append((item_id, similarity_score))
            except (KeyError, IndexError):
                continue
        
        # Sort by similarity score and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Get recommendations using hybrid approach (combining collaborative and content-based)
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (item_id, combined_score) tuples
        """
        # Get recommendations from both models
        collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
        content_recs = self.get_content_based_recommendations(user_id, n_recommendations * 2)
        
        # Combine recommendations with weighted scores
        combined_scores = {}
        
        # Normalize and weight collaborative filtering scores
        if collab_recs:
            collab_scores = [score for _, score in collab_recs]
            min_collab, max_collab = min(collab_scores), max(collab_scores)
            
            for item_id, score in collab_recs:
                normalized_score = (score - min_collab) / (max_collab - min_collab) if max_collab > min_collab else 0
                combined_scores[item_id] = normalized_score * self.hybrid_weights['collaborative']
        
        # Normalize and weight content-based scores
        if content_recs:
            content_scores = [score for _, score in content_recs]
            min_content, max_content = min(content_scores), max(content_scores)
            
            for item_id, score in content_recs:
                normalized_score = (score - min_content) / (max_content - min_content) if max_content > min_content else 0
                if item_id in combined_scores:
                    combined_scores[item_id] += normalized_score * self.hybrid_weights['content']
                else:
                    combined_scores[item_id] = normalized_score * self.hybrid_weights['content']
        
        # Sort by combined score and return top N
        hybrid_recommendations = [(item_id, score) for item_id, score in combined_scores.items()]
        hybrid_recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_recommendations[:n_recommendations]
    
    def evaluate_model(self, interactions_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the recommendation models using standard metrics
        
        Args:
            interactions_df: DataFrame with user-item interactions for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating recommendation models...")
        
        config = self.config['evaluation']
        
        # Split data for evaluation
        train_df, test_df = train_test_split(
            interactions_df, 
            test_size=config['test_size'], 
            random_state=config['random_state']
        )
        
        # Calculate RMSE and MAE for collaborative filtering
        metrics = {}
        
        if self.collaborative_model is not None:
            predictions = []
            actuals = []
            
            for _, row in test_df.iterrows():
                user_id, item_id, actual_rating = row['user_id'], row['item_id'], row['rating']
                
                try:
                    user_idx = self.user_item_matrix.index.get_loc(user_id)
                    item_idx = self.user_item_matrix.columns.get_loc(item_id)
                    
                    user_vector = self.collaborative_model['user_factors'][user_idx:user_idx+1]
                    predicted_rating = user_vector.dot(self.collaborative_model['item_factors'][:, item_idx])[0]
                    
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
                    
                except KeyError:
                    continue
            
            if predictions:
                metrics['collaborative_rmse'] = np.sqrt(mean_squared_error(actuals, predictions))
                metrics['collaborative_mae'] = mean_absolute_error(actuals, predictions)
        
        # Calculate coverage and diversity metrics
        total_items = len(self.user_item_matrix.columns)
        sample_users = self.user_item_matrix.index[:100]  # Sample for efficiency
        
        recommended_items = set()
        for user_id in sample_users:
            user_recs = self.get_hybrid_recommendations(user_id, 10)
            recommended_items.update([item_id for item_id, _ in user_recs])
        
        metrics['catalog_coverage'] = len(recommended_items) / total_items
        metrics['total_recommended_items'] = len(recommended_items)
        metrics['total_items'] = total_items
        
        logger.info(f"Evaluation completed - Coverage: {metrics.get('catalog_coverage', 0):.3f}")
        
        return metrics
    
    def save_model(self, model_path: str, metadata: Dict[str, Any] = None):
        """
        Save the trained recommendation models
        
        Args:
            model_path: Path to save the model
            metadata: Additional metadata to save with the model
        """
        logger.info(f"Saving model to {model_path}")
        
        model_data = {
            'collaborative_model': self.collaborative_model,
            'content_model': self.content_model,
            'user_item_matrix': self.user_item_matrix,
            'content_features': self.content_features,
            'hybrid_weights': self.hybrid_weights,
            'config': self.config,
            'metadata': metadata or {},
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        logger.info("Model saved successfully")
    
    def load_model(self, model_path: str):
        """
        Load a previously trained recommendation model
        
        Args:
            model_path: Path to the saved model
        """
        logger.info(f"Loading model from {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.collaborative_model = model_data['collaborative_model']
        self.content_model = model_data['content_model']
        self.user_item_matrix = model_data['user_item_matrix']
        self.content_features = model_data['content_features']
        self.hybrid_weights = model_data['hybrid_weights']
        self.config = model_data['config']
        self.model_metadata = model_data.get('metadata', {})
        
        logger.info("Model loaded successfully")


def log_experiment_to_mlflow(engine: ContentRecommendationEngine, metrics: Dict[str, float], 
                           config: Dict[str, Any], model_path: str):
    """
    Log experiment results to MLflow for tracking
    
    Args:
        engine: Trained recommendation engine
        metrics: Evaluation metrics
        config: Model configuration
        model_path: Path to saved model
    """
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model artifacts
        if engine.collaborative_model:
            mlflow.sklearn.log_model(
                engine.collaborative_model['svd'], 
                "collaborative_filtering_model"
            )
        
        if engine.content_model:
            mlflow.sklearn.log_model(
                engine.content_model['tfidf'], 
                "content_based_model"
            )
        
        # Log the complete model
        mlflow.log_artifact(model_path, "complete_model")
        
        # Log configuration
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact("config.json")


if __name__ == "__main__":
    # Example usage
    print("Content Recommendation Engine - Core Models")
    print("This module provides the main recommendation algorithms.")
    print("Import this module to use the ContentRecommendationEngine class.")