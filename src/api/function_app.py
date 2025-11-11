"""
Azure Functions API for Content Recommendation Engine
====================================================

This module provides REST API endpoints for the content recommendation engine:
- Get recommendations for users
- Update user interactions
- Content search and discovery
- Real-time event processing

Author: Content Recommendation Engine Team
Date: October 2025
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Add the src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.recommendation_engine import ContentRecommendationEngine
from utils.config import Config
from api.models import RecommendationRequest, RecommendationResponse, InteractionEvent
from api.cache import CacheManager
from api.monitoring import APIMonitor
from api.ab_test_router import ABTestRouter
from models.openai_models import OpenAIRecommendationRequest, UserProfile

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and configuration
recommendation_engine = None
config = None
cache_manager = None
api_monitor = None
ab_test_router = None

def init():
    """Initialize the recommendation engine and dependencies"""
    global recommendation_engine, config, cache_manager, api_monitor, ab_test_router
    
    try:
        # Load configuration
        config = Config()
        
        # Initialize cache manager
        cache_manager = CacheManager(config)
        
        # Initialize monitoring
        api_monitor = APIMonitor(config)
        
        # Initialize A/B test router
        ab_test_router = ABTestRouter()
        
        # Load the trained model
        model_path = os.getenv('MODEL_PATH', '/models/recommendation_model.joblib')
        if os.path.exists(model_path):
            recommendation_engine = ContentRecommendationEngine()
            recommendation_engine.load_model(model_path)
            logger.info("Recommendation engine loaded successfully")
        else:
            logger.warning(f"Model not found at {model_path}, using fallback recommendations")
            recommendation_engine = None
            
    except Exception as e:
        logger.error(f"Failed to initialize recommendation engine: {e}")
        recommendation_engine = None

# Initialize on module load
init()

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Main Azure Function entry point for HTTP requests
    """
    try:
        # Parse request
        method = req.method
        url_path = req.url.split('/')[-1] if '/' in req.url else ''
        
        # Route requests
        if method == 'GET' and 'recommendations' in url_path:
            return get_recommendations(req)
        elif method == 'POST' and 'recommendations' in url_path:
            return get_recommendations_post(req)
        elif method == 'POST' and 'interactions' in url_path:
            return record_interaction(req)
        elif method == 'GET' and 'trending' in url_path:
            return get_trending_content(req)
        elif method == 'GET' and 'similar' in url_path:
            return get_similar_items(req)
        elif method == 'GET' and 'health' in url_path:
            return health_check(req)
        elif method == 'GET' and 'ab-test' in url_path:
            return get_ab_test_status(req)
        else:
            return func.HttpResponse(
                json.dumps({"error": "Endpoint not found"}),
                status_code=404,
                mimetype="application/json"
            )
            
    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        if api_monitor:
            api_monitor.log_error(str(e))
        
        return func.HttpResponse(
            json.dumps({"error": "Internal server error"}),
            status_code=500,
            mimetype="application/json"
        )

def get_recommendations(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get personalized recommendations for a user
    
    Expected query parameters:
    - user_id: User identifier
    - num_recommendations: Number of recommendations (default: 10)
    - algorithm: Algorithm type (collaborative, content, hybrid)
    """
    start_time = datetime.now()
    
    try:
        # Parse request parameters
        user_id = req.params.get('user_id')
        if not user_id:
            return func.HttpResponse(
                json.dumps({"error": "user_id parameter is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        user_id = int(user_id)
        num_recommendations = int(req.params.get('num_recommendations', 10))
        algorithm = req.params.get('algorithm', 'hybrid')
        
        # Validate parameters
        if num_recommendations <= 0 or num_recommendations > config.api.max_recommendations:
            return func.HttpResponse(
                json.dumps({
                    "error": f"num_recommendations must be between 1 and {config.api.max_recommendations}"
                }),
                status_code=400,
                mimetype="application/json"
            )
        
        # Check cache first
        cache_key = f"recommendations:{user_id}:{algorithm}:{num_recommendations}"
        cached_result = cache_manager.get(cache_key) if cache_manager else None
        
        if cached_result:
            logger.info(f"Returning cached recommendations for user {user_id}")
            if api_monitor:
                api_monitor.log_request("recommendations", True, datetime.now() - start_time)
            
            return func.HttpResponse(
                json.dumps(cached_result),
                status_code=200,
                mimetype="application/json"
            )
        
        # Generate recommendations
        if recommendation_engine is None:
            # Fallback to random recommendations
            recommendations = generate_fallback_recommendations(user_id, num_recommendations)
        else:
            # Use trained model
            if algorithm == 'collaborative':
                recommendations = recommendation_engine.get_collaborative_recommendations(
                    user_id, num_recommendations
                )
            elif algorithm == 'content':
                recommendations = recommendation_engine.get_content_based_recommendations(
                    user_id, num_recommendations
                )
            else:  # hybrid
                recommendations = recommendation_engine.get_hybrid_recommendations(
                    user_id, num_recommendations
                )
        
        # Format response
        response_data = {
            "user_id": user_id,
            "algorithm": algorithm,
            "recommendations": [
                {
                    "item_id": item_id,
                    "score": float(score),
                    "rank": idx + 1
                }
                for idx, (item_id, score) in enumerate(recommendations)
            ],
            "generated_at": datetime.now().isoformat(),
            "total_count": len(recommendations)
        }
        
        # Cache the result
        if cache_manager:
            cache_manager.set(cache_key, response_data, ttl=config.api.cache_ttl)
        
        # Log metrics
        if api_monitor:
            api_monitor.log_request("recommendations", True, datetime.now() - start_time)
            api_monitor.log_metric("recommendations_generated", len(recommendations))
        
        return func.HttpResponse(
            json.dumps(response_data),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        if api_monitor:
            api_monitor.log_request("recommendations", False, datetime.now() - start_time)
            api_monitor.log_error(str(e))
        
        return func.HttpResponse(
            json.dumps({"error": "Failed to generate recommendations"}),
            status_code=500,
            mimetype="application/json"
        )

def record_interaction(req: func.HttpRequest) -> func.HttpResponse:
    """
    Record a user interaction event
    
    Expected POST body:
    {
        "user_id": int,
        "item_id": int,
        "interaction_type": str (view, rating, purchase, etc.),
        "rating": float (optional),
        "timestamp": str (optional)
    }
    """
    start_time = datetime.now()
    
    try:
        # Parse request body
        req_body = req.get_json()
        if not req_body:
            return func.HttpResponse(
                json.dumps({"error": "Request body is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Validate required fields
        required_fields = ['user_id', 'item_id', 'interaction_type']
        for field in required_fields:
            if field not in req_body:
                return func.HttpResponse(
                    json.dumps({"error": f"{field} is required"}),
                    status_code=400,
                    mimetype="application/json"
                )
        
        # Create interaction event
        interaction = {
            "user_id": int(req_body['user_id']),
            "item_id": int(req_body['item_id']),
            "interaction_type": req_body['interaction_type'],
            "rating": req_body.get('rating'),
            "timestamp": req_body.get('timestamp', datetime.now().isoformat()),
            "session_id": req_body.get('session_id'),
            "metadata": req_body.get('metadata', {})
        }
        
        # Store interaction (in a real implementation, this would go to a database or event hub)
        logger.info(f"Recorded interaction: {interaction}")
        
        # Invalidate cache for this user
        if cache_manager:
            cache_manager.invalidate_user_cache(interaction['user_id'])
        
        # Log metrics
        if api_monitor:
            api_monitor.log_request("interactions", True, datetime.now() - start_time)
            api_monitor.log_metric("interactions_recorded", 1)
            api_monitor.log_metric(f"interaction_type_{interaction['interaction_type']}", 1)
        
        response_data = {
            "status": "success",
            "interaction_id": f"{interaction['user_id']}_{interaction['item_id']}_{int(datetime.now().timestamp())}",
            "recorded_at": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(response_data),
            status_code=201,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error recording interaction: {e}")
        if api_monitor:
            api_monitor.log_request("interactions", False, datetime.now() - start_time)
            api_monitor.log_error(str(e))
        
        return func.HttpResponse(
            json.dumps({"error": "Failed to record interaction"}),
            status_code=500,
            mimetype="application/json"
        )

def get_trending_content(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get trending content based on recent activity
    
    Expected query parameters:
    - time_window: Time window in hours (default: 24)
    - limit: Number of items to return (default: 20)
    - category: Content category filter (optional)
    """
    start_time = datetime.now()
    
    try:
        # Parse parameters
        time_window = int(req.params.get('time_window', 24))
        limit = int(req.params.get('limit', 20))
        category = req.params.get('category')
        
        # Check cache
        cache_key = f"trending:{time_window}:{limit}:{category or 'all'}"
        cached_result = cache_manager.get(cache_key) if cache_manager else None
        
        if cached_result:
            if api_monitor:
                api_monitor.log_request("trending", True, datetime.now() - start_time)
            
            return func.HttpResponse(
                json.dumps(cached_result),
                status_code=200,
                mimetype="application/json"
            )
        
        # Generate trending content (fallback implementation)
        trending_items = generate_fallback_trending(limit, category)
        
        response_data = {
            "time_window_hours": time_window,
            "category": category,
            "trending_items": trending_items,
            "generated_at": datetime.now().isoformat(),
            "total_count": len(trending_items)
        }
        
        # Cache result
        if cache_manager:
            cache_manager.set(cache_key, response_data, ttl=config.api.cache_ttl)
        
        # Log metrics
        if api_monitor:
            api_monitor.log_request("trending", True, datetime.now() - start_time)
        
        return func.HttpResponse(
            json.dumps(response_data),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error getting trending content: {e}")
        if api_monitor:
            api_monitor.log_request("trending", False, datetime.now() - start_time)
            api_monitor.log_error(str(e))
        
        return func.HttpResponse(
            json.dumps({"error": "Failed to get trending content"}),
            status_code=500,
            mimetype="application/json"
        )

def get_similar_items(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get items similar to a given item
    
    Expected query parameters:
    - item_id: Item identifier
    - num_similar: Number of similar items (default: 10)
    """
    start_time = datetime.now()
    
    try:
        # Parse parameters
        item_id = req.params.get('item_id')
        if not item_id:
            return func.HttpResponse(
                json.dumps({"error": "item_id parameter is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        item_id = int(item_id)
        num_similar = int(req.params.get('num_similar', 10))
        
        # Check cache
        cache_key = f"similar:{item_id}:{num_similar}"
        cached_result = cache_manager.get(cache_key) if cache_manager else None
        
        if cached_result:
            if api_monitor:
                api_monitor.log_request("similar", True, datetime.now() - start_time)
            
            return func.HttpResponse(
                json.dumps(cached_result),
                status_code=200,
                mimetype="application/json"
            )
        
        # Generate similar items (fallback implementation)
        similar_items = generate_fallback_similar_items(item_id, num_similar)
        
        response_data = {
            "item_id": item_id,
            "similar_items": similar_items,
            "generated_at": datetime.now().isoformat(),
            "total_count": len(similar_items)
        }
        
        # Cache result
        if cache_manager:
            cache_manager.set(cache_key, response_data, ttl=config.api.cache_ttl)
        
        # Log metrics
        if api_monitor:
            api_monitor.log_request("similar", True, datetime.now() - start_time)
        
        return func.HttpResponse(
            json.dumps(response_data),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error getting similar items: {e}")
        if api_monitor:
            api_monitor.log_request("similar", False, datetime.now() - start_time)
            api_monitor.log_error(str(e))
        
        return func.HttpResponse(
            json.dumps({"error": "Failed to get similar items"}),
            status_code=500,
            mimetype="application/json"
        )

def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "recommendation_engine": "loaded" if recommendation_engine else "not_loaded",
                "cache": "available" if cache_manager else "not_available",
                "monitoring": "available" if api_monitor else "not_available"
            }
        }
        
        return func.HttpResponse(
            json.dumps(health_status),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return func.HttpResponse(
            json.dumps({
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

def generate_fallback_recommendations(user_id: int, num_recommendations: int) -> List[tuple]:
    """
    Generate fallback recommendations when the ML model is not available
    """
    np.random.seed(user_id)  # Ensure consistent recommendations for the same user
    
    # Generate random item IDs and scores
    recommendations = []
    for i in range(num_recommendations):
        item_id = np.random.randint(1, 1000)
        score = np.random.uniform(0.5, 1.0)
        recommendations.append((item_id, score))
    
    # Sort by score descending
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return recommendations

def generate_fallback_trending(limit: int, category: Optional[str] = None) -> List[Dict]:
    """
    Generate fallback trending content
    """
    trending_items = []
    
    for i in range(limit):
        item = {
            "item_id": np.random.randint(1, 1000),
            "title": f"Trending Item {i+1}",
            "category": category or np.random.choice(['Action', 'Comedy', 'Drama', 'Horror']),
            "trend_score": np.random.uniform(0.7, 1.0),
            "rank": i + 1
        }
        trending_items.append(item)
    
    return trending_items

def generate_fallback_similar_items(item_id: int, num_similar: int) -> List[Dict]:
    """
    Generate fallback similar items
    """
    np.random.seed(item_id)  # Ensure consistent similar items for the same item
    
    similar_items = []
    for i in range(num_similar):
        similar_item = {
            "item_id": np.random.randint(1, 1000),
            "similarity_score": np.random.uniform(0.6, 0.95),
            "rank": i + 1
        }
        similar_items.append(similar_item)
    
    # Sort by similarity score descending
    similar_items.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return similar_items

async def get_recommendations_post(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get personalized recommendations with A/B testing support via POST request
    
    Expected POST body:
    {
        "user_id": str,
        "user_profile": dict (optional),
        "num_recommendations": int (default: 10),
        "context": dict (optional),
        "exclude_items": list (optional),
        "enable_ab_test": bool (default: false),
        "test_name": str (optional)
    }
    """
    start_time = datetime.now()
    
    try:
        # Parse request body
        req_body = req.get_json()
        if not req_body:
            return func.HttpResponse(
                json.dumps({"error": "Request body is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Validate required fields
        user_id = req_body.get('user_id')
        if not user_id:
            return func.HttpResponse(
                json.dumps({"error": "user_id is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Extract parameters
        user_profile = req_body.get('user_profile', {})
        num_recommendations = req_body.get('num_recommendations', 10)
        context = req_body.get('context', {})
        exclude_items = req_body.get('exclude_items', [])
        enable_ab_test = req_body.get('enable_ab_test', False)
        test_name = req_body.get('test_name')
        
        # Add session ID to context if provided
        context['session_id'] = req_body.get('session_id', context.get('session_id'))
        
        # Check if A/B testing is enabled and router is available
        if enable_ab_test and ab_test_router is not None:
            try:
                # Initialize A/B test router if needed
                if not hasattr(ab_test_router, 'openai_engine') or ab_test_router.openai_engine is None:
                    await ab_test_router.initialize()
                
                # Create OpenAI recommendation request
                openai_request = OpenAIRecommendationRequest(
                    user_id=user_id,
                    user_profile=UserProfile(**user_profile),
                    num_recommendations=num_recommendations,
                    context=context,
                    exclude_items=exclude_items
                )
                
                # Route through A/B test
                ab_result = await ab_test_router.route_recommendation_request(
                    openai_request, test_name
                )
                
                # Convert recommendations to API format
                recommendations_data = []
                for rec in ab_result.recommendations:
                    rec_dict = {
                        "item_id": rec.id,
                        "title": rec.title,
                        "description": rec.description,
                        "genre": rec.genre,
                        "category": rec.category,
                        "rating": rec.rating,
                        "confidence_score": rec.confidence_score,
                        "relevance_score": rec.relevance_score,
                        "explanation": rec.explanation,
                        "source": rec.source
                    }
                    
                    # Add OpenAI-specific scores if available
                    if hasattr(rec, 'content_score') and rec.content_score is not None:
                        rec_dict["content_score"] = rec.content_score
                    if hasattr(rec, 'ai_score') and rec.ai_score is not None:
                        rec_dict["ai_score"] = rec.ai_score
                    if hasattr(rec, 'personalization_score') and rec.personalization_score is not None:
                        rec_dict["personalization_score"] = rec.personalization_score
                    if hasattr(rec, 'final_score') and rec.final_score is not None:
                        rec_dict["final_score"] = rec.final_score
                    
                    recommendations_data.append(rec_dict)
                
                # Format response
                response_data = {
                    "user_id": user_id,
                    "recommendations": recommendations_data,
                    "generated_at": datetime.now().isoformat(),
                    "total_count": len(recommendations_data),
                    "ab_test_info": {
                        "enabled": True,
                        "test_name": ab_result.test_name,
                        "variant": ab_result.variant,
                        "algorithm_used": ab_result.algorithm_used.value,
                        "response_time_ms": ab_result.response_time_ms,
                        "session_id": ab_result.session_id
                    }
                }
                
                # Log metrics
                if api_monitor:
                    api_monitor.log_request("recommendations_ab", True, datetime.now() - start_time)
                    api_monitor.log_metric("ab_test_requests", 1)
                    api_monitor.log_metric(f"ab_variant_{ab_result.variant}", 1)
                
                return func.HttpResponse(
                    json.dumps(response_data, default=str),
                    status_code=200,
                    mimetype="application/json"
                )
                
            except Exception as e:
                logger.error(f"A/B test failed, falling back to traditional: {e}")
                # Fall through to traditional recommendations
        
        # Traditional recommendations (fallback or when A/B testing is disabled)
        algorithm = req_body.get('algorithm', 'hybrid')
        
        # Check cache first
        cache_key = f"recommendations:{user_id}:{algorithm}:{num_recommendations}"
        cached_result = cache_manager.get(cache_key) if cache_manager else None
        
        if cached_result:
            logger.info(f"Returning cached recommendations for user {user_id}")
            # Add A/B test info to cached result
            cached_result["ab_test_info"] = {
                "enabled": False,
                "reason": "cached_result"
            }
            
            if api_monitor:
                api_monitor.log_request("recommendations", True, datetime.now() - start_time)
            
            return func.HttpResponse(
                json.dumps(cached_result),
                status_code=200,
                mimetype="application/json"
            )
        
        # Generate traditional recommendations
        if recommendation_engine is None:
            # Fallback to random recommendations
            recommendations = generate_fallback_recommendations(user_id, num_recommendations)
        else:
            # Use trained model
            if algorithm == 'collaborative':
                recommendations = recommendation_engine.get_collaborative_recommendations(
                    user_id, num_recommendations
                )
            elif algorithm == 'content':
                recommendations = recommendation_engine.get_content_based_recommendations(
                    user_id, num_recommendations
                )
            else:  # hybrid
                recommendations = recommendation_engine.get_hybrid_recommendations(
                    user_id, num_recommendations
                )
        
        # Format response
        response_data = {
            "user_id": user_id,
            "algorithm": algorithm,
            "recommendations": [
                {
                    "item_id": item_id,
                    "score": float(score),
                    "rank": idx + 1,
                    "source": "traditional"
                }
                for idx, (item_id, score) in enumerate(recommendations)
            ],
            "generated_at": datetime.now().isoformat(),
            "total_count": len(recommendations),
            "ab_test_info": {
                "enabled": False,
                "reason": "traditional_only" if not enable_ab_test else "ab_test_unavailable"
            }
        }
        
        # Cache the result
        if cache_manager:
            cache_manager.set(cache_key, response_data, ttl=config.api.cache_ttl)
        
        # Log metrics
        if api_monitor:
            api_monitor.log_request("recommendations", True, datetime.now() - start_time)
            api_monitor.log_metric("recommendations_generated", len(recommendations))
        
        return func.HttpResponse(
            json.dumps(response_data),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        if api_monitor:
            api_monitor.log_request("recommendations", False, datetime.now() - start_time)
            api_monitor.log_error(str(e))
        
        return func.HttpResponse(
            json.dumps({"error": "Failed to generate recommendations"}),
            status_code=500,
            mimetype="application/json"
        )

def get_ab_test_status(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get A/B test status and metrics
    
    Expected query parameters:
    - test_name: Test name (optional)
    - days_back: Number of days to look back for metrics (default: 7)
    """
    start_time = datetime.now()
    
    try:
        # Check if A/B test router is available
        if ab_test_router is None:
            return func.HttpResponse(
                json.dumps({
                    "error": "A/B testing is not available",
                    "available": False
                }),
                status_code=503,
                mimetype="application/json"
            )
        
        # Get query parameters
        test_name = req.params.get('test_name')
        days_back = int(req.params.get('days_back', 7))
        
        # Get test metrics
        metrics = ab_test_router.calculate_test_metrics(test_name, days_back)
        
        # Get active tests
        active_tests = ab_test_router.get_active_tests()
        
        response_data = {
            "available": True,
            "metrics": metrics,
            "active_tests": active_tests,
            "query_parameters": {
                "test_name": test_name,
                "days_back": days_back
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Log metrics
        if api_monitor:
            api_monitor.log_request("ab_test_status", True, datetime.now() - start_time)
        
        return func.HttpResponse(
            json.dumps(response_data, default=str),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error getting A/B test status: {e}")
        if api_monitor:
            api_monitor.log_request("ab_test_status", False, datetime.now() - start_time)
            api_monitor.log_error(str(e))
        
        return func.HttpResponse(
            json.dumps({"error": "Failed to get A/B test status"}),
            status_code=500,
            mimetype="application/json"
        )