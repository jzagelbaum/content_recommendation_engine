"""
Unit Tests for ML Recommendation Engine
======================================

Test suite for the core recommendation engine functionality.

Author: Content Recommendation Engine Team
Date: October 2025
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import tempfile
import os

# Import the recommendation engine
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'ml'))

from recommendation_engine import ContentRecommendationEngine


class TestContentRecommendationEngine:
    """Test cases for ContentRecommendationEngine"""
    
    def test_initialization(self):
        """Test engine initialization"""
        engine = ContentRecommendationEngine()
        
        assert engine is not None
        assert hasattr(engine, 'collaborative_model')
        assert hasattr(engine, 'content_model')
        assert hasattr(engine, 'hybrid_weights')
        
    def test_fit_collaborative_model(self, sample_users_data, sample_items_data, sample_interactions_data):
        """Test collaborative filtering model training"""
        engine = ContentRecommendationEngine()
        
        # Train the model
        engine.fit_collaborative_model(
            interactions=sample_interactions_data,
            users=sample_users_data,
            items=sample_items_data
        )
        
        # Check that model was trained
        assert engine.collaborative_model is not None
        assert hasattr(engine, 'user_encoder')
        assert hasattr(engine, 'item_encoder')
        
    def test_fit_content_model(self, sample_items_data):
        """Test content-based filtering model training"""
        engine = ContentRecommendationEngine()
        
        # Train the model
        engine.fit_content_model(items=sample_items_data)
        
        # Check that model was trained
        assert engine.content_model is not None
        assert hasattr(engine, 'content_features')
        
    def test_get_collaborative_recommendations(self, sample_users_data, sample_items_data, sample_interactions_data):
        """Test collaborative filtering recommendations"""
        engine = ContentRecommendationEngine()
        
        # Train the model
        engine.fit_collaborative_model(
            interactions=sample_interactions_data,
            users=sample_users_data,
            items=sample_items_data
        )
        
        # Get recommendations
        user_id = sample_users_data['user_id'].iloc[0]
        recommendations = engine.get_collaborative_recommendations(
            user_id=user_id,
            num_recommendations=5
        )
        
        # Validate recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        for rec in recommendations:
            assert 'item_id' in rec
            assert 'score' in rec
            assert isinstance(rec['score'], (int, float))
            
    def test_get_content_recommendations(self, sample_items_data):
        """Test content-based filtering recommendations"""
        engine = ContentRecommendationEngine()
        
        # Train the model
        engine.fit_content_model(items=sample_items_data)
        
        # Get recommendations
        item_id = sample_items_data['item_id'].iloc[0]
        recommendations = engine.get_content_recommendations(
            item_id=item_id,
            num_recommendations=5
        )
        
        # Validate recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        for rec in recommendations:
            assert 'item_id' in rec
            assert 'score' in rec
            assert isinstance(rec['score'], (int, float))
            
    def test_get_hybrid_recommendations(self, sample_users_data, sample_items_data, sample_interactions_data):
        """Test hybrid recommendations"""
        engine = ContentRecommendationEngine()
        
        # Train both models
        engine.fit_collaborative_model(
            interactions=sample_interactions_data,
            users=sample_users_data,
            items=sample_items_data
        )
        engine.fit_content_model(items=sample_items_data)
        
        # Get hybrid recommendations
        user_id = sample_users_data['user_id'].iloc[0]
        recommendations = engine.get_hybrid_recommendations(
            user_id=user_id,
            num_recommendations=10
        )
        
        # Validate recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 10
        for rec in recommendations:
            assert 'item_id' in rec
            assert 'score' in rec
            assert isinstance(rec['score'], (int, float))
            
    def test_evaluate_model(self, sample_users_data, sample_items_data, sample_interactions_data):
        """Test model evaluation"""
        engine = ContentRecommendationEngine()
        
        # Train the model
        engine.fit_collaborative_model(
            interactions=sample_interactions_data,
            users=sample_users_data,
            items=sample_items_data
        )
        
        # Evaluate the model
        metrics = engine.evaluate_model(
            test_interactions=sample_interactions_data.sample(100),
            k=5
        )
        
        # Validate metrics
        assert isinstance(metrics, dict)
        assert 'precision_at_k' in metrics
        assert 'recall_at_k' in metrics
        assert 'f1_at_k' in metrics
        assert 'ndcg_at_k' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['precision_at_k'] <= 1
        assert 0 <= metrics['recall_at_k'] <= 1
        assert 0 <= metrics['f1_at_k'] <= 1
        assert 0 <= metrics['ndcg_at_k'] <= 1
        
    def test_get_trending_items(self, sample_interactions_data):
        """Test trending items functionality"""
        engine = ContentRecommendationEngine()
        
        trending = engine.get_trending_items(
            interactions=sample_interactions_data,
            time_window_days=7,
            num_items=10
        )
        
        # Validate trending items
        assert isinstance(trending, list)
        assert len(trending) <= 10
        for item in trending:
            assert 'item_id' in item
            assert 'score' in item
            assert isinstance(item['score'], (int, float))
            
    def test_get_similar_items(self, sample_items_data):
        """Test similar items functionality"""
        engine = ContentRecommendationEngine()
        
        # Train content model
        engine.fit_content_model(items=sample_items_data)
        
        # Get similar items
        item_id = sample_items_data['item_id'].iloc[0]
        similar = engine.get_similar_items(
            item_id=item_id,
            num_items=5
        )
        
        # Validate similar items
        assert isinstance(similar, list)
        assert len(similar) <= 5
        for item in similar:
            assert 'item_id' in item
            assert 'score' in item
            assert isinstance(item['score'], (int, float))
            
    def test_save_and_load_model(self, sample_users_data, sample_items_data, sample_interactions_data):
        """Test model persistence"""
        engine = ContentRecommendationEngine()
        
        # Train the model
        engine.fit_collaborative_model(
            interactions=sample_interactions_data,
            users=sample_users_data,
            items=sample_items_data
        )
        
        # Save model to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model')
            engine.save_model(model_path)
            
            # Check that files were created
            assert os.path.exists(f"{model_path}_collaborative.joblib")
            assert os.path.exists(f"{model_path}_encoders.joblib")
            
            # Load model in new engine
            new_engine = ContentRecommendationEngine()
            new_engine.load_model(model_path)
            
            # Test that loaded model works
            user_id = sample_users_data['user_id'].iloc[0]
            recommendations = new_engine.get_collaborative_recommendations(
                user_id=user_id,
                num_recommendations=5
            )
            
            assert isinstance(recommendations, list)
            assert len(recommendations) <= 5
            
    def test_handle_cold_start_user(self, sample_items_data, sample_interactions_data):
        """Test handling of cold start users"""
        engine = ContentRecommendationEngine()
        
        # Train content model only
        engine.fit_content_model(items=sample_items_data)
        
        # Try to get recommendations for non-existent user
        recommendations = engine.get_hybrid_recommendations(
            user_id='new_user_999',
            num_recommendations=5
        )
        
        # Should fallback to popular items or content-based
        assert isinstance(recommendations, list)
        
    def test_handle_cold_start_item(self, sample_users_data, sample_items_data, sample_interactions_data):
        """Test handling of cold start items"""
        engine = ContentRecommendationEngine()
        
        # Train models
        engine.fit_collaborative_model(
            interactions=sample_interactions_data,
            users=sample_users_data,
            items=sample_items_data
        )
        engine.fit_content_model(items=sample_items_data)
        
        # Try to get similar items for non-existent item
        similar = engine.get_similar_items(
            item_id='new_item_999',
            num_items=5
        )
        
        # Should handle gracefully
        assert isinstance(similar, list)
        
    def test_filter_recommendations(self, sample_users_data, sample_items_data, sample_interactions_data):
        """Test recommendation filtering"""
        engine = ContentRecommendationEngine()
        
        # Train models
        engine.fit_collaborative_model(
            interactions=sample_interactions_data,
            users=sample_users_data,
            items=sample_items_data
        )
        
        # Get recommendations with filters
        user_id = sample_users_data['user_id'].iloc[0]
        
        # Get user's watched items
        user_interactions = sample_interactions_data[
            sample_interactions_data['user_id'] == user_id
        ]
        watched_items = set(user_interactions['item_id'].tolist())
        
        recommendations = engine.get_collaborative_recommendations(
            user_id=user_id,
            num_recommendations=10,
            exclude_items=watched_items
        )
        
        # Validate that watched items are excluded
        recommended_items = {rec['item_id'] for rec in recommendations}
        assert len(recommended_items.intersection(watched_items)) == 0
        
    def test_diversify_recommendations(self, sample_users_data, sample_items_data, sample_interactions_data):
        """Test recommendation diversification"""
        engine = ContentRecommendationEngine()
        
        # Train models
        engine.fit_collaborative_model(
            interactions=sample_interactions_data,
            users=sample_users_data,
            items=sample_items_data
        )
        engine.fit_content_model(items=sample_items_data)
        
        # Get diversified recommendations
        user_id = sample_users_data['user_id'].iloc[0]
        recommendations = engine.get_hybrid_recommendations(
            user_id=user_id,
            num_recommendations=10,
            diversify=True,
            diversity_weight=0.3
        )
        
        # Should still return valid recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 10
        
    @patch('mlflow.start_run')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_params')
    def test_mlflow_logging(self, mock_log_params, mock_log_metrics, mock_start_run, 
                           sample_users_data, sample_items_data, sample_interactions_data):
        """Test MLflow experiment logging"""
        engine = ContentRecommendationEngine()
        
        # Train with MLflow logging
        engine.fit_collaborative_model(
            interactions=sample_interactions_data,
            users=sample_users_data,
            items=sample_items_data,
            experiment_name='test_experiment'
        )
        
        # Verify MLflow was called
        mock_start_run.assert_called()
        mock_log_params.assert_called()
        
    def test_batch_recommendations(self, sample_users_data, sample_items_data, sample_interactions_data):
        """Test batch recommendation generation"""
        engine = ContentRecommendationEngine()
        
        # Train the model
        engine.fit_collaborative_model(
            interactions=sample_interactions_data,
            users=sample_users_data,
            items=sample_items_data
        )
        
        # Get batch recommendations
        user_ids = sample_users_data['user_id'].head(5).tolist()
        batch_recommendations = engine.get_batch_recommendations(
            user_ids=user_ids,
            num_recommendations=5
        )
        
        # Validate batch results
        assert isinstance(batch_recommendations, dict)
        assert len(batch_recommendations) == len(user_ids)
        
        for user_id in user_ids:
            assert user_id in batch_recommendations
            assert isinstance(batch_recommendations[user_id], list)
            assert len(batch_recommendations[user_id]) <= 5
            
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        engine = ContentRecommendationEngine()
        
        # Test with empty dataframes
        with pytest.raises((ValueError, IndexError)):
            engine.fit_collaborative_model(
                interactions=pd.DataFrame(),
                users=pd.DataFrame(),
                items=pd.DataFrame()
            )
            
        # Test with invalid user_id
        recommendations = engine.get_collaborative_recommendations(
            user_id=None,
            num_recommendations=5
        )
        assert recommendations == []
        
    def test_performance_metrics(self, sample_users_data, sample_items_data, sample_interactions_data):
        """Test recommendation performance tracking"""
        engine = ContentRecommendationEngine()
        
        # Train the model
        start_time = pd.Timestamp.now()
        engine.fit_collaborative_model(
            interactions=sample_interactions_data,
            users=sample_users_data,
            items=sample_items_data
        )
        training_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Test recommendation speed
        user_id = sample_users_data['user_id'].iloc[0]
        start_time = pd.Timestamp.now()
        recommendations = engine.get_collaborative_recommendations(
            user_id=user_id,
            num_recommendations=10
        )
        inference_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Basic performance assertions
        assert training_time < 60  # Should train in under 60 seconds
        assert inference_time < 5  # Should recommend in under 5 seconds
        assert len(recommendations) <= 10