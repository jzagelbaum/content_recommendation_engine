"""
Models package for the recommendation system
"""

from .openai_models import (
    OpenAIRecommendationRequest,
    OpenAIRecommendationResponse,
    RecommendationItem,
    ABTestConfig,
    ABTestResult,
    UserInteraction,
    ContentItem,
    UserProfile,
    SyntheticDataRequest,
    SyntheticDataResponse,
    AlgorithmType,
    InteractionType,
    ContentType,
    validate_recommendation_request,
    validate_ab_test_config,
    convert_legacy_recommendation
)

__all__ = [
    'OpenAIRecommendationRequest',
    'OpenAIRecommendationResponse', 
    'RecommendationItem',
    'ABTestConfig',
    'ABTestResult',
    'UserInteraction',
    'ContentItem',
    'UserProfile',
    'SyntheticDataRequest',
    'SyntheticDataResponse',
    'AlgorithmType',
    'InteractionType',
    'ContentType',
    'validate_recommendation_request',
    'validate_ab_test_config',
    'convert_legacy_recommendation'
]