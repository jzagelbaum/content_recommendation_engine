"""
A/B Testing Router for Recommendation Systems
Routes traffic between traditional and OpenAI-powered recommendation engines
"""

import hashlib
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from ..models.openai_models import (
    ABTestConfig, ABTestResult, OpenAIRecommendationRequest, 
    RecommendationItem, AlgorithmType
)
from ..openai.openai_recommendation_engine import OpenAIRecommendationEngine
import logging

logger = logging.getLogger(__name__)

class ABTestRouter:
    """Router for A/B testing between traditional and OpenAI recommendation engines"""
    
    def __init__(self):
        """Initialize the A/B test router"""
        self.openai_engine = OpenAIRecommendationEngine()
        self.test_configs: Dict[str, ABTestConfig] = {}
        self.results_storage: List[ABTestResult] = []  # In production, use proper storage
        self.default_test_name = "openai_vs_traditional"
        
        # Initialize default test configuration
        self._setup_default_test()
    
    def _setup_default_test(self):
        """Setup default A/B test configuration"""
        default_config = ABTestConfig(
            test_name=self.default_test_name,
            traffic_split=0.3,  # 30% to OpenAI, 70% to traditional
            enabled=True,
            control_algorithm=AlgorithmType.TRADITIONAL,
            treatment_algorithm=AlgorithmType.OPENAI,
            description="Default A/B test comparing traditional vs OpenAI recommendations"
        )
        self.test_configs[self.default_test_name] = default_config
        logger.info(f"Initialized default A/B test with 30% OpenAI traffic")
    
    async def initialize(self) -> bool:
        """Initialize the A/B test router and OpenAI engine"""
        try:
            success = await self.openai_engine.initialize()
            if success:
                logger.info("A/B test router initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Error initializing A/B test router: {e}")
            return False
    
    def configure_test(self, config: ABTestConfig) -> bool:
        """Configure A/B test parameters"""
        try:
            # Validate test configuration
            if not self._validate_test_config(config):
                return False
            
            self.test_configs[config.test_name] = config
            logger.info(
                f"Configured A/B test: {config.test_name} with "
                f"{config.traffic_split*100}% {config.treatment_algorithm} traffic"
            )
            return True
        except Exception as e:
            logger.error(f"Error configuring A/B test: {e}")
            return False
    
    async def route_recommendation_request(
        self, 
        request: OpenAIRecommendationRequest,
        test_name: Optional[str] = None
    ) -> ABTestResult:
        """Route recommendation request based on A/B test configuration"""
        start_time = datetime.utcnow()
        test_name = test_name or self.default_test_name
        
        try:
            # Get test configuration
            config = self.test_configs.get(test_name)
            if not config or not config.enabled or not self._is_test_active(config):
                # Default to control algorithm
                variant = "control"
                algorithm_used = AlgorithmType.TRADITIONAL
                recommendations = await self._get_traditional_recommendations(request)
            else:
                # Determine variant based on user ID hash and traffic split
                variant, algorithm_used = self._determine_variant(request.user_id, config)
                
                if algorithm_used == AlgorithmType.OPENAI:
                    recommendations = await self._get_openai_recommendations(request)
                else:
                    recommendations = await self._get_traditional_recommendations(request)
            
            # Calculate response time
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Create result
            result = ABTestResult(
                user_id=request.user_id,
                test_name=test_name,
                variant=variant,
                algorithm_used=algorithm_used,
                recommendations=recommendations,
                session_id=request.context.get("session_id") if request.context else None,
                response_time_ms=response_time,
                error_occurred=False
            )
            
            # Store result for analysis
            self._store_result(result)
            
            return result
        except Exception as e:
            logger.error(f"Error in A/B test routing: {e}")
            
            # Create error result with fallback to traditional
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            fallback_recommendations = await self._get_traditional_recommendations_safe(request)
            
            return ABTestResult(
                user_id=request.user_id,
                test_name=test_name,
                variant="control",
                algorithm_used=AlgorithmType.TRADITIONAL,
                recommendations=fallback_recommendations,
                session_id=request.context.get("session_id") if request.context else None,
                response_time_ms=response_time,
                error_occurred=True,
                error_message=str(e)
            )
    
    def _determine_variant(self, user_id: str, config: ABTestConfig) -> Tuple[str, AlgorithmType]:
        """Determine which variant to show based on user ID hash"""
        # Use hash of user ID for consistent assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0  # More precise than 100
        
        if normalized_hash < config.traffic_split:
            return "treatment", config.treatment_algorithm
        else:
            return "control", config.control_algorithm
    
    async def _get_openai_recommendations(
        self, 
        request: OpenAIRecommendationRequest
    ) -> List[RecommendationItem]:
        """Get recommendations from OpenAI engine"""
        try:
            response = await self.openai_engine.get_recommendations(
                user_id=request.user_id,
                user_profile=request.user_profile,
                num_recommendations=request.num_recommendations,
                context=request.context,
                exclude_items=request.exclude_items
            )
            
            # Convert response to RecommendationItem list
            recommendations = []
            for rec_data in response.get("recommendations", []):
                if isinstance(rec_data, dict):
                    rec_item = RecommendationItem(
                        id=rec_data.get("id", ""),
                        title=rec_data.get("title", ""),
                        description=rec_data.get("description"),
                        genre=rec_data.get("genre", []),
                        category=rec_data.get("category"),
                        rating=rec_data.get("rating"),
                        confidence_score=rec_data.get("final_recommendation_score", 0.5),
                        relevance_score=rec_data.get("content_score", 0.5),
                        explanation=response.get("explanation", ""),
                        source="openai",
                        content_score=rec_data.get("content_score"),
                        ai_score=rec_data.get("ai_score"),
                        personalization_score=rec_data.get("personalization_score"),
                        final_score=rec_data.get("final_recommendation_score")
                    )
                    recommendations.append(rec_item)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting OpenAI recommendations: {e}")
            # Fallback to traditional recommendations
            return await self._get_traditional_recommendations(request)
    
    async def _get_traditional_recommendations(
        self, 
        request: OpenAIRecommendationRequest
    ) -> List[RecommendationItem]:
        """Get recommendations from traditional engine"""
        try:
            # Import here to avoid circular imports
            from ..api.recommendations import get_user_recommendations
            
            # Convert to format expected by traditional engine
            traditional_request = {
                "user_id": request.user_id,
                "user_profile": request.user_profile,
                "num_recommendations": request.num_recommendations,
                "filters": {
                    "exclude_items": request.exclude_items or []
                }
            }
            
            # Call existing recommendation function
            recommendations_data = await get_user_recommendations(traditional_request)
            
            # Convert to RecommendationItem format
            recommendations = []
            for rec_data in recommendations_data.get("recommendations", []):
                rec_item = RecommendationItem(
                    id=rec_data.get("item_id", rec_data.get("id", "")),
                    title=rec_data.get("title", ""),
                    description=rec_data.get("description"),
                    genre=rec_data.get("genre", []),
                    category=rec_data.get("category"),
                    rating=rec_data.get("rating"),
                    confidence_score=rec_data.get("score", 0.5),
                    relevance_score=rec_data.get("relevance", rec_data.get("score", 0.5)),
                    explanation="Based on your preferences and viewing history",
                    source="traditional",
                    final_score=rec_data.get("score", 0.5)
                )
                recommendations.append(rec_item)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting traditional recommendations: {e}")
            return []
    
    async def _get_traditional_recommendations_safe(
        self, 
        request: OpenAIRecommendationRequest
    ) -> List[RecommendationItem]:
        """Safe fallback for traditional recommendations"""
        try:
            return await self._get_traditional_recommendations(request)
        except Exception as e:
            logger.error(f"Safe fallback also failed: {e}")
            # Return minimal fallback recommendations
            return [
                RecommendationItem(
                    id=f"fallback_{i}",
                    title=f"Recommended Content {i+1}",
                    confidence_score=0.3,
                    relevance_score=0.3,
                    explanation="Fallback recommendation",
                    source="fallback"
                )
                for i in range(min(request.num_recommendations, 5))
            ]
    
    def _validate_test_config(self, config: ABTestConfig) -> bool:
        """Validate A/B test configuration"""
        try:
            # Check traffic split
            if not 0.0 <= config.traffic_split <= 1.0:
                logger.error(f"Invalid traffic split: {config.traffic_split}")
                return False
            
            # Check date constraints
            if config.start_date and config.end_date:
                if config.start_date >= config.end_date:
                    logger.error("Start date must be before end date")
                    return False
            
            # Check algorithm types
            if config.control_algorithm == config.treatment_algorithm:
                logger.warning("Control and treatment algorithms are the same")
            
            return True
        except Exception as e:
            logger.error(f"Error validating test config: {e}")
            return False
    
    def _is_test_active(self, config: ABTestConfig) -> bool:
        """Check if test is currently active"""
        now = datetime.utcnow()
        
        if config.start_date and now < config.start_date:
            return False
        
        if config.end_date and now > config.end_date:
            return False
        
        return config.enabled
    
    def _store_result(self, result: ABTestResult):
        """Store A/B test result (in production, use proper storage)"""
        try:
            self.results_storage.append(result)
            
            # Keep only recent results to avoid memory issues
            if len(self.results_storage) > 10000:
                self.results_storage = self.results_storage[-5000:]
        except Exception as e:
            logger.error(f"Error storing A/B test result: {e}")
    
    def get_test_results(
        self, 
        test_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[ABTestResult]:
        """Get A/B test results for analysis"""
        try:
            results = self.results_storage
            
            # Filter by test name
            if test_name:
                results = [r for r in results if r.test_name == test_name]
            
            # Filter by date range
            if start_date:
                results = [r for r in results if r.timestamp >= start_date]
            
            if end_date:
                results = [r for r in results if r.timestamp <= end_date]
            
            return results
        except Exception as e:
            logger.error(f"Error getting test results: {e}")
            return []
    
    def calculate_test_metrics(
        self, 
        test_name: Optional[str] = None,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """Calculate A/B test metrics"""
        try:
            # Get recent results
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            results = self.get_test_results(test_name, start_date, end_date)
            
            if not results:
                return {"error": "No results found for the specified period"}
            
            # Separate by variant
            control_results = [r for r in results if r.variant == "control"]
            treatment_results = [r for r in results if r.variant == "treatment"]
            
            # Calculate basic metrics
            total_users = len(results)
            control_users = len(control_results)
            treatment_users = len(treatment_results)
            
            # Calculate performance metrics
            control_avg_response_time = (
                sum(r.response_time_ms or 0 for r in control_results) / len(control_results)
                if control_results else 0
            )
            
            treatment_avg_response_time = (
                sum(r.response_time_ms or 0 for r in treatment_results) / len(treatment_results)
                if treatment_results else 0
            )
            
            # Error rates
            control_error_rate = (
                sum(1 for r in control_results if r.error_occurred) / len(control_results)
                if control_results else 0
            )
            
            treatment_error_rate = (
                sum(1 for r in treatment_results if r.error_occurred) / len(treatment_results)
                if treatment_results else 0
            )
            
            # Recommendation quality metrics
            control_avg_confidence = (
                sum(
                    sum(rec.confidence_score for rec in r.recommendations) / len(r.recommendations)
                    for r in control_results if r.recommendations
                ) / len(control_results) if control_results else 0
            )
            
            treatment_avg_confidence = (
                sum(
                    sum(rec.confidence_score for rec in r.recommendations) / len(r.recommendations)
                    for r in treatment_results if r.recommendations
                ) / len(treatment_results) if treatment_results else 0
            )
            
            return {
                "test_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days_back
                },
                "traffic_distribution": {
                    "total_users": total_users,
                    "control_users": control_users,
                    "treatment_users": treatment_users,
                    "actual_treatment_split": treatment_users / total_users if total_users > 0 else 0
                },
                "performance_metrics": {
                    "control_avg_response_time_ms": round(control_avg_response_time, 2),
                    "treatment_avg_response_time_ms": round(treatment_avg_response_time, 2),
                    "response_time_improvement": round(
                        ((control_avg_response_time - treatment_avg_response_time) / control_avg_response_time * 100)
                        if control_avg_response_time > 0 else 0, 2
                    ),
                    "control_error_rate": round(control_error_rate * 100, 2),
                    "treatment_error_rate": round(treatment_error_rate * 100, 2)
                },
                "quality_metrics": {
                    "control_avg_confidence": round(control_avg_confidence, 3),
                    "treatment_avg_confidence": round(treatment_avg_confidence, 3),
                    "confidence_improvement": round(
                        ((treatment_avg_confidence - control_avg_confidence) / control_avg_confidence * 100)
                        if control_avg_confidence > 0 else 0, 2
                    )
                },
                "algorithm_breakdown": {
                    "control_algorithm": control_results[0].algorithm_used.value if control_results else "unknown",
                    "treatment_algorithm": treatment_results[0].algorithm_used.value if treatment_results else "unknown"
                }
            }
        except Exception as e:
            logger.error(f"Error calculating test metrics: {e}")
            return {"error": str(e)}
    
    def get_active_tests(self) -> List[Dict[str, Any]]:
        """Get list of active A/B tests"""
        try:
            active_tests = []
            for test_name, config in self.test_configs.items():
                if self._is_test_active(config):
                    active_tests.append({
                        "test_name": test_name,
                        "traffic_split": config.traffic_split,
                        "control_algorithm": config.control_algorithm.value,
                        "treatment_algorithm": config.treatment_algorithm.value,
                        "description": config.description,
                        "start_date": config.start_date.isoformat() if config.start_date else None,
                        "end_date": config.end_date.isoformat() if config.end_date else None
                    })
            return active_tests
        except Exception as e:
            logger.error(f"Error getting active tests: {e}")
            return []
    
    def stop_test(self, test_name: str) -> bool:
        """Stop an active A/B test"""
        try:
            if test_name in self.test_configs:
                self.test_configs[test_name].enabled = False
                logger.info(f"Stopped A/B test: {test_name}")
                return True
            else:
                logger.warning(f"Test not found: {test_name}")
                return False
        except Exception as e:
            logger.error(f"Error stopping test: {e}")
            return False
    
    async def close(self):
        """Close the A/B test router and cleanup resources"""
        await self.openai_engine.close()