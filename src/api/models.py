"""
Data Models for Content Recommendation API
==========================================

Pydantic models for request/response validation and serialization.

Author: Content Recommendation Engine Team
Date: October 2025
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class InteractionType(str, Enum):
    """Types of user interactions"""
    VIEW = "view"
    RATING = "rating"
    LIKE = "like"
    SHARE = "share"
    PURCHASE = "purchase"
    DOWNLOAD = "download"
    BOOKMARK = "bookmark"

class AlgorithmType(str, Enum):
    """Types of recommendation algorithms"""
    COLLABORATIVE = "collaborative"
    CONTENT = "content"
    HYBRID = "hybrid"

class RecommendationRequest(BaseModel):
    """Request model for getting recommendations"""
    user_id: int = Field(..., description="User identifier")
    num_recommendations: int = Field(10, ge=1, le=100, description="Number of recommendations to return")
    algorithm: AlgorithmType = Field(AlgorithmType.HYBRID, description="Algorithm type to use")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters (genre, year, etc.)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context information")

class RecommendationItem(BaseModel):
    """Individual recommendation item"""
    item_id: int = Field(..., description="Item identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Recommendation score")
    rank: int = Field(..., ge=1, description="Recommendation rank")
    explanation: Optional[str] = Field(None, description="Explanation for the recommendation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional item metadata")

class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    user_id: int = Field(..., description="User identifier")
    algorithm: AlgorithmType = Field(..., description="Algorithm used")
    recommendations: List[RecommendationItem] = Field(..., description="List of recommendations")
    generated_at: datetime = Field(..., description="When recommendations were generated")
    total_count: int = Field(..., description="Number of recommendations returned")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")

class InteractionEvent(BaseModel):
    """Model for user interaction events"""
    user_id: int = Field(..., description="User identifier")
    item_id: int = Field(..., description="Item identifier")
    interaction_type: InteractionType = Field(..., description="Type of interaction")
    rating: Optional[float] = Field(None, ge=1.0, le=5.0, description="Rating value (1-5)")
    timestamp: Optional[datetime] = Field(None, description="Interaction timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier")
    device_type: Optional[str] = Field(None, description="Device type (mobile, desktop, tablet)")
    platform: Optional[str] = Field(None, description="Platform (web, app, tv)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.now()

class InteractionResponse(BaseModel):
    """Response model for interaction recording"""
    status: str = Field(..., description="Status of the operation")
    interaction_id: str = Field(..., description="Unique interaction identifier")
    recorded_at: datetime = Field(..., description="When the interaction was recorded")
    message: Optional[str] = Field(None, description="Additional message")

class TrendingRequest(BaseModel):
    """Request model for trending content"""
    time_window_hours: int = Field(24, ge=1, le=168, description="Time window in hours")
    limit: int = Field(20, ge=1, le=100, description="Number of trending items to return")
    category: Optional[str] = Field(None, description="Content category filter")
    region: Optional[str] = Field(None, description="Geographic region filter")

class TrendingItem(BaseModel):
    """Individual trending item"""
    item_id: int = Field(..., description="Item identifier")
    title: str = Field(..., description="Item title")
    category: str = Field(..., description="Item category")
    trend_score: float = Field(..., ge=0.0, description="Trending score")
    rank: int = Field(..., ge=1, description="Trending rank")
    interaction_count: Optional[int] = Field(None, description="Number of recent interactions")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class TrendingResponse(BaseModel):
    """Response model for trending content"""
    time_window_hours: int = Field(..., description="Time window used")
    category: Optional[str] = Field(None, description="Category filter applied")
    trending_items: List[TrendingItem] = Field(..., description="List of trending items")
    generated_at: datetime = Field(..., description="When trending data was generated")
    total_count: int = Field(..., description="Number of trending items returned")

class SimilarItemsRequest(BaseModel):
    """Request model for similar items"""
    item_id: int = Field(..., description="Item identifier to find similar items for")
    num_similar: int = Field(10, ge=1, le=50, description="Number of similar items to return")
    similarity_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")

class SimilarItem(BaseModel):
    """Individual similar item"""
    item_id: int = Field(..., description="Item identifier")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    rank: int = Field(..., ge=1, description="Similarity rank")
    similarity_type: Optional[str] = Field(None, description="Type of similarity (content, collaborative)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class SimilarItemsResponse(BaseModel):
    """Response model for similar items"""
    item_id: int = Field(..., description="Source item identifier")
    similar_items: List[SimilarItem] = Field(..., description="List of similar items")
    generated_at: datetime = Field(..., description="When similar items were generated")
    total_count: int = Field(..., description="Number of similar items returned")

class UserProfileRequest(BaseModel):
    """Request model for user profile operations"""
    user_id: int = Field(..., description="User identifier")
    include_preferences: bool = Field(True, description="Include user preferences")
    include_history: bool = Field(False, description="Include interaction history")
    history_limit: int = Field(100, ge=1, le=1000, description="Limit for interaction history")

class UserPreferences(BaseModel):
    """User preferences model"""
    favorite_genres: List[str] = Field(default_factory=list, description="Favorite genres")
    preferred_duration: Optional[str] = Field(None, description="Preferred content duration")
    content_types: List[str] = Field(default_factory=list, description="Preferred content types")
    languages: List[str] = Field(default_factory=list, description="Preferred languages")
    explicit_content: bool = Field(False, description="Allow explicit content")

class UserProfile(BaseModel):
    """User profile model"""
    user_id: int = Field(..., description="User identifier")
    preferences: UserPreferences = Field(..., description="User preferences")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="User statistics")
    last_active: Optional[datetime] = Field(None, description="Last activity timestamp")
    created_at: Optional[datetime] = Field(None, description="Account creation timestamp")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component health status")
    metrics: Optional[Dict[str, Any]] = Field(None, description="System metrics")

class BatchRecommendationRequest(BaseModel):
    """Request model for batch recommendations"""
    user_ids: List[int] = Field(..., min_items=1, max_items=100, description="List of user identifiers")
    num_recommendations: int = Field(10, ge=1, le=100, description="Number of recommendations per user")
    algorithm: AlgorithmType = Field(AlgorithmType.HYBRID, description="Algorithm type to use")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters")

class BatchRecommendationResponse(BaseModel):
    """Response model for batch recommendations"""
    recommendations: Dict[int, List[RecommendationItem]] = Field(..., description="Recommendations by user ID")
    generated_at: datetime = Field(..., description="When recommendations were generated")
    total_users: int = Field(..., description="Number of users processed")
    successful_users: int = Field(..., description="Number of successful recommendations")
    failed_users: List[int] = Field(default_factory=list, description="User IDs that failed")

class SearchRequest(BaseModel):
    """Request model for content search"""
    query: str = Field(..., min_length=1, description="Search query")
    user_id: Optional[int] = Field(None, description="User ID for personalized search")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    limit: int = Field(20, ge=1, le=100, description="Number of results to return")
    offset: int = Field(0, ge=0, description="Pagination offset")

class SearchResult(BaseModel):
    """Individual search result"""
    item_id: int = Field(..., description="Item identifier")
    title: str = Field(..., description="Item title")
    description: Optional[str] = Field(None, description="Item description")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    rank: int = Field(..., ge=1, description="Search result rank")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class SearchResponse(BaseModel):
    """Response model for content search"""
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of matching items")
    offset: int = Field(..., description="Pagination offset")
    limit: int = Field(..., description="Results limit")
    generated_at: datetime = Field(..., description="When search was performed")

# Configuration for JSON schema generation
class Config:
    """Pydantic configuration"""
    json_encoders = {
        datetime: lambda v: v.isoformat()
    }
    schema_extra = {
        "example": {
            "api_version": "1.0.0",
            "description": "Content Recommendation Engine API Models"
        }
    }