"""
Data models for OpenAI-powered recommendation system
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

class AlgorithmType(str, Enum):
    """Algorithm types for recommendations"""
    TRADITIONAL = "traditional"
    OPENAI = "openai"
    HYBRID = "hybrid"

class InteractionType(str, Enum):
    """Types of user interactions"""
    VIEW = "view"
    LIKE = "like"
    DISLIKE = "dislike"
    SHARE = "share"
    BOOKMARK = "bookmark"
    SKIP = "skip"

class ContentType(str, Enum):
    """Types of content"""
    MOVIE = "movie"
    TV_SERIES = "tv_series"
    DOCUMENTARY = "documentary"
    SHORT = "short"
    LIVE = "live"

class OpenAIRecommendationRequest(BaseModel):
    """Request model for OpenAI recommendations"""
    user_id: str = Field(..., description="Unique user identifier")
    user_profile: Dict[str, Any] = Field(..., description="User profile data")
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations requested")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context (mood, device, etc.)")
    exclude_items: Optional[List[str]] = Field(default=None, description="Items to exclude from recommendations")
    algorithm_preference: Optional[AlgorithmType] = Field(default=AlgorithmType.OPENAI, description="Preferred algorithm")
    include_explanation: bool = Field(default=True, description="Whether to include recommendation explanation")
    
    @validator('user_profile')
    def validate_user_profile(cls, v):
        """Validate user profile contains required fields"""
        required_fields = ['preferences']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"User profile must contain '{field}' field")
        return v

class RecommendationItem(BaseModel):
    """Individual recommendation item"""
    id: str = Field(..., description="Content item ID")
    title: str = Field(..., description="Content title")
    description: Optional[str] = Field(default=None, description="Content description")
    content_type: Optional[ContentType] = Field(default=None, description="Type of content")
    genre: Optional[List[str]] = Field(default=None, description="Content genres")
    category: Optional[str] = Field(default=None, description="Content category")
    rating: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="Content rating")
    release_year: Optional[int] = Field(default=None, description="Release year")
    duration_minutes: Optional[int] = Field(default=None, ge=0, description="Duration in minutes")
    popularity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Popularity score")
    
    # Recommendation specific fields
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in this recommendation")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to user preferences")
    explanation: Optional[str] = Field(default=None, description="Why this item was recommended")
    source: Optional[str] = Field(default=None, description="Recommendation source (vector, ai, hybrid)")
    
    # Scoring breakdown
    content_score: Optional[float] = Field(default=None, description="Content-based similarity score")
    collaborative_score: Optional[float] = Field(default=None, description="Collaborative filtering score")
    ai_score: Optional[float] = Field(default=None, description="AI-generated relevance score")
    personalization_score: Optional[float] = Field(default=None, description="Personalization boost score")
    final_score: Optional[float] = Field(default=None, description="Final combined score")

class OpenAIRecommendationResponse(BaseModel):
    """Response model for OpenAI recommendations"""
    user_id: str = Field(..., description="User ID from request")
    recommendations: List[RecommendationItem] = Field(..., description="List of recommended items")
    explanation: str = Field(..., description="Overall explanation of recommendations")
    algorithm_version: str = Field(..., description="Algorithm version used")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in recommendations")
    diversity_score: float = Field(..., ge=0.0, le=1.0, description="Diversity of recommendations")
    ai_insights: Optional[str] = Field(default=None, description="AI-generated insights about user preferences")
    processing_time_seconds: float = Field(..., ge=0.0, description="Time taken to generate recommendations")
    total_candidates_evaluated: int = Field(..., ge=0, description="Total number of items evaluated")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="When recommendations were generated")
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class ABTestConfig(BaseModel):
    """Configuration for A/B testing"""
    test_name: str = Field(..., description="Name of the A/B test")
    traffic_split: float = Field(..., ge=0.0, le=1.0, description="Percentage of traffic for OpenAI variant (0.0-1.0)")
    enabled: bool = Field(default=True, description="Whether the test is active")
    start_date: Optional[datetime] = Field(default=None, description="Test start date")
    end_date: Optional[datetime] = Field(default=None, description="Test end date")
    control_algorithm: AlgorithmType = Field(default=AlgorithmType.TRADITIONAL, description="Control algorithm")
    treatment_algorithm: AlgorithmType = Field(default=AlgorithmType.OPENAI, description="Treatment algorithm")
    description: Optional[str] = Field(default=None, description="Test description")
    
    @validator('traffic_split')
    def validate_traffic_split(cls, v):
        """Validate traffic split is between 0 and 1"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Traffic split must be between 0.0 and 1.0")
        return v

class ABTestResult(BaseModel):
    """Result of an A/B test assignment"""
    user_id: str = Field(..., description="User ID")
    test_name: str = Field(..., description="Name of the A/B test")
    variant: str = Field(..., description="Assigned variant (control/treatment)")
    algorithm_used: AlgorithmType = Field(..., description="Algorithm used for this user")
    recommendations: List[RecommendationItem] = Field(..., description="Generated recommendations")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When assignment was made")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    
    # Performance metrics
    response_time_ms: Optional[float] = Field(default=None, description="Response time in milliseconds")
    error_occurred: bool = Field(default=False, description="Whether an error occurred")
    error_message: Optional[str] = Field(default=None, description="Error message if applicable")

class UserInteraction(BaseModel):
    """User interaction with content"""
    interaction_id: str = Field(..., description="Unique interaction ID")
    user_id: str = Field(..., description="User ID")
    content_id: str = Field(..., description="Content ID")
    interaction_type: InteractionType = Field(..., description="Type of interaction")
    rating: Optional[float] = Field(default=None, ge=0.0, le=5.0, description="User rating (0-5)")
    watch_duration_minutes: Optional[int] = Field(default=None, ge=0, description="Watch duration")
    completion_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Completion rate (0-1)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Interaction timestamp")
    device: Optional[str] = Field(default=None, description="Device used")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class ContentItem(BaseModel):
    """Content item model"""
    id: str = Field(..., description="Unique content ID")
    title: str = Field(..., description="Content title")
    description: Optional[str] = Field(default=None, description="Content description")
    content_type: ContentType = Field(..., description="Type of content")
    genre: List[str] = Field(default_factory=list, description="Content genres")
    category: Optional[str] = Field(default=None, description="Content category")
    rating: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="Content rating")
    release_year: Optional[int] = Field(default=None, description="Release year")
    duration_minutes: Optional[int] = Field(default=None, ge=0, description="Duration in minutes")
    cast: List[str] = Field(default_factory=list, description="Cast members")
    director: Optional[str] = Field(default=None, description="Director")
    language: Optional[str] = Field(default=None, description="Primary language")
    country: Optional[str] = Field(default=None, description="Country of origin")
    content_rating: Optional[str] = Field(default=None, description="Content rating (PG, R, etc.)")
    
    # Computed fields
    popularity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Popularity score")
    imdb_rating: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="IMDB rating")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    # AI-enhanced fields
    ai_features: Optional[Dict[str, Any]] = Field(default=None, description="AI-extracted features")
    keywords: List[str] = Field(default_factory=list, description="Keywords for search")
    mood: Optional[str] = Field(default=None, description="Content mood")
    themes: List[str] = Field(default_factory=list, description="Content themes")
    target_audience: Optional[str] = Field(default=None, description="Target audience")

class UserProfile(BaseModel):
    """User profile model"""
    user_id: str = Field(..., description="Unique user ID")
    age: Optional[int] = Field(default=None, ge=13, le=120, description="User age")
    region: Optional[str] = Field(default=None, description="User region")
    preferences: List[str] = Field(default_factory=list, description="Preferred genres/categories")
    viewing_history: List[str] = Field(default_factory=list, description="Recently watched content")
    viewing_patterns: Optional[Dict[str, Any]] = Field(default=None, description="Viewing behavior patterns")
    subscription_type: Optional[str] = Field(default=None, description="Subscription level")
    personality_traits: List[str] = Field(default_factory=list, description="Personality traits")
    device_preferences: List[str] = Field(default_factory=list, description="Preferred devices")
    
    # Computed preferences
    preferred_duration_range: Optional[Dict[str, int]] = Field(default=None, description="Preferred content duration")
    preferred_time_slots: List[str] = Field(default_factory=list, description="Preferred viewing times")
    content_discovery_preference: Optional[str] = Field(default=None, description="How user discovers content")
    
    # Profile metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Profile creation date")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last profile update")
    last_activity: Optional[datetime] = Field(default=None, description="Last activity timestamp")

class SyntheticDataRequest(BaseModel):
    """Request for synthetic data generation"""
    num_users: int = Field(default=100, ge=1, le=10000, description="Number of users to generate")
    num_content_items: int = Field(default=500, ge=1, le=50000, description="Number of content items to generate")
    interactions_per_user: int = Field(default=50, ge=1, le=1000, description="Interactions per user")
    time_span_days: int = Field(default=365, ge=1, le=3650, description="Time span for interactions in days")
    
    # Distribution parameters
    demographic_distribution: Optional[Dict[str, Any]] = Field(default=None, description="Demographic distribution")
    content_distribution: Optional[Dict[str, Any]] = Field(default=None, description="Content distribution")
    include_ai_features: bool = Field(default=True, description="Whether to generate AI features")

class SyntheticDataResponse(BaseModel):
    """Response for synthetic data generation"""
    users: List[UserProfile] = Field(..., description="Generated user profiles")
    content_items: List[ContentItem] = Field(..., description="Generated content items")
    interactions: List[UserInteraction] = Field(..., description="Generated interactions")
    generation_stats: Dict[str, Any] = Field(..., description="Generation statistics")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")

# Utility functions for model validation and conversion
def validate_recommendation_request(request_data: Dict[str, Any]) -> OpenAIRecommendationRequest:
    """Validate and create a recommendation request"""
    return OpenAIRecommendationRequest(**request_data)

def validate_ab_test_config(config_data: Dict[str, Any]) -> ABTestConfig:
    """Validate and create an A/B test configuration"""
    return ABTestConfig(**config_data)

def convert_legacy_recommendation(legacy_rec: Dict[str, Any]) -> RecommendationItem:
    """Convert legacy recommendation format to new model"""
    return RecommendationItem(
        id=legacy_rec.get("id", ""),
        title=legacy_rec.get("title", ""),
        description=legacy_rec.get("description"),
        genre=legacy_rec.get("genre", []),
        category=legacy_rec.get("category"),
        rating=legacy_rec.get("rating"),
        confidence_score=legacy_rec.get("score", 0.5),
        relevance_score=legacy_rec.get("relevance", 0.5),
        explanation=legacy_rec.get("explanation")
    )