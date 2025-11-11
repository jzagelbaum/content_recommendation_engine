# OpenAI-Powered Recommendation System

This document describes the OpenAI-powered recommendation system that complements the traditional machine learning approach in the Content Recommendation Engine.

## Overview

The OpenAI integration provides an alternative recommendation engine that leverages:
- **Azure OpenAI Service** for natural language processing and embeddings
- **Vector Search** with Azure AI Search for semantic similarity
- **A/B Testing Framework** to compare traditional vs. OpenAI approaches
- **Synthetic Data Generation** for testing and development

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    A/B Testing Router                          │
├─────────────────────┬───────────────────────────────────────────┤
│                     │                                           │
│ Traditional ML      │ OpenAI-Powered Recommendations           │
│ Recommendations     │                                           │
│                     │ ┌─────────────────────────────────────┐   │
│ • Collaborative     │ │          OpenAI Service             │   │
│   Filtering         │ │                                     │   │
│ • Content-Based     │ │ • GPT-4 for content analysis       │   │
│ • Hybrid Approach   │ │ • Embeddings for similarity         │   │
│                     │ │ • Natural language explanations    │   │
│                     │ └─────────────────────────────────────┘   │
│                     │                                           │
│                     │ ┌─────────────────────────────────────┐   │
│                     │ │         Vector Search               │   │
│                     │ │                                     │   │
│                     │ │ • Azure AI Search                   │   │
│                     │ │ • HNSW algorithm                    │   │
│                     │ │ • Hybrid search capabilities       │   │
│                     │ └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. OpenAI Service (`src/openai/openai_service.py`)
- **Azure OpenAI Integration**: Connects to Azure OpenAI with proper authentication
- **Embedding Generation**: Creates vector embeddings for content and user profiles
- **Content Analysis**: Uses GPT-4 to analyze and categorize content
- **Batch Processing**: Handles large-scale operations efficiently
- **Rate Limiting**: Manages API quotas and prevents throttling

### 2. Embedding Service (`src/openai/embedding_service.py`)
- **Vector Search**: Manages Azure AI Search integration
- **Index Management**: Creates and maintains search indexes
- **Similarity Search**: Finds similar content using vector embeddings
- **Hybrid Recommendations**: Combines keyword and semantic search

### 3. OpenAI Recommendation Engine (`src/openai/openai_recommendation_engine.py`)
- **Content-Based Recommendations**: Uses embeddings for similarity matching
- **AI-Enhanced Insights**: Leverages GPT-4 for personalized explanations
- **Multi-Factor Scoring**: Combines content, AI, and personalization scores
- **Async Processing**: Handles concurrent recommendation requests

### 4. Data Generator (`src/openai/data_generator.py`)
- **Synthetic User Profiles**: Generates realistic user data for testing
- **Content Item Creation**: Creates diverse content with metadata
- **Interaction Simulation**: Generates realistic user interaction patterns
- **Batch Generation**: Supports large-scale data creation

### 5. A/B Testing Router (`src/api/ab_test_router.py`)
- **Traffic Splitting**: Routes users between traditional and OpenAI engines
- **Consistent Assignment**: Ensures users see the same variant
- **Metrics Collection**: Tracks performance and quality metrics
- **Configuration Management**: Supports multiple concurrent tests

## API Endpoints

### OpenAI Function App (`src/openai-functions/function_app.py`)

#### Health Check
```http
GET /api/health
```

#### OpenAI Recommendations
```http
POST /api/openai/recommendations
Content-Type: application/json

{
  "user_id": "user123",
  "user_profile": {
    "age": 25,
    "preferences": ["action", "comedy"],
    "viewing_history": ["item1", "item2"]
  },
  "num_recommendations": 10,
  "context": {
    "device": "mobile",
    "time_of_day": "evening"
  },
  "exclude_items": ["item3"]
}
```

#### A/B Test Recommendations
```http
POST /api/openai/ab-test
Content-Type: application/json

{
  "user_id": "user123",
  "user_profile": { ... },
  "num_recommendations": 10,
  "test_name": "openai_vs_traditional"
}
```

#### Synthetic Data Generation
```http
POST /api/openai/generate-data
Content-Type: application/json

{
  "data_type": "users", // or "content", "interactions"
  "count": 50,
  "content_type": "movie" // for content generation
}
```

#### A/B Test Configuration
```http
POST /api/ab-test/configure
Content-Type: application/json

{
  "test_name": "my_test",
  "traffic_split": 0.3,
  "enabled": true,
  "control_algorithm": "traditional",
  "treatment_algorithm": "openai",
  "description": "Testing OpenAI vs traditional recommendations"
}
```

#### A/B Test Results
```http
GET /api/ab-test/results?test_name=my_test&days_back=7
```

### Main API with A/B Testing (`src/api/function_app.py`)

#### Enhanced Recommendations
```http
POST /api/recommendations
Content-Type: application/json

{
  "user_id": "user123",
  "user_profile": { ... },
  "num_recommendations": 10,
  "enable_ab_test": true,
  "test_name": "openai_vs_traditional"
}
```

## Data Models

### OpenAI Request/Response Models (`src/models/openai_models.py`)

#### OpenAI Recommendation Request
```python
class OpenAIRecommendationRequest(BaseModel):
    user_id: str
    user_profile: UserProfile
    num_recommendations: int = 10
    context: Dict[str, Any] = {}
    exclude_items: List[str] = []
```

#### Recommendation Item
```python
class RecommendationItem(BaseModel):
    id: str
    title: str
    description: Optional[str]
    genre: List[str] = []
    category: Optional[str]
    rating: Optional[float]
    confidence_score: float
    relevance_score: float
    explanation: str
    source: str  # "openai" or "traditional"
    content_score: Optional[float]  # OpenAI-specific
    ai_score: Optional[float]       # OpenAI-specific
    personalization_score: Optional[float]  # OpenAI-specific
    final_score: Optional[float]    # OpenAI-specific
```

#### A/B Test Configuration
```python
class ABTestConfig(BaseModel):
    test_name: str
    traffic_split: float  # 0.0 to 1.0
    enabled: bool = True
    control_algorithm: AlgorithmType = AlgorithmType.TRADITIONAL
    treatment_algorithm: AlgorithmType = AlgorithmType.OPENAI
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    description: str = ""
```

## Configuration

### Environment Variables

#### Azure OpenAI Configuration
```bash
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_GPT_DEPLOYMENT=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

#### Azure AI Search Configuration
```bash
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_API_KEY=your-search-key
AZURE_SEARCH_INDEX_NAME=content-recommendations
```

#### A/B Testing Configuration
```bash
ENABLE_AB_TESTING=true
DEFAULT_AB_TEST_TRAFFIC_SPLIT=0.3
```

## Deployment

### Infrastructure Deployment
```bash
# Deploy infrastructure with OpenAI components
az deployment group create \
  --resource-group rg-content-rec-dev \
  --template-file infrastructure/main.bicep \
  --parameters @infrastructure/parameters/dev.parameters.json
```

### Function App Deployment
```bash
# Deploy OpenAI Function App
cd src/openai-functions
func azure functionapp publish <function-app-name>
```

### Main API Update
```bash
# Deploy main API with A/B testing support
cd src/api
func azure functionapp publish <main-function-app-name>
```

## A/B Testing Strategy

### Default Test Configuration
- **Traffic Split**: 30% OpenAI, 70% Traditional
- **Duration**: 2-4 weeks for statistical significance
- **Metrics**: Response time, recommendation quality, user engagement

### Key Metrics to Monitor

#### Performance Metrics
- **Response Time**: Latency comparison between algorithms
- **Error Rate**: Success rate of recommendation generation
- **Throughput**: Requests per second capacity

#### Quality Metrics
- **Confidence Scores**: Algorithm confidence in recommendations
- **Relevance Scores**: Contextual relevance of recommendations
- **User Engagement**: Click-through rates, conversion rates

#### Business Metrics
- **User Satisfaction**: Feedback scores and ratings
- **Content Discovery**: New content exposure rates
- **Revenue Impact**: Conversion and purchase metrics

### Statistical Analysis
- **Sample Size**: Minimum 1000 users per variant
- **Significance Level**: 95% confidence interval
- **Effect Size**: Minimum 5% improvement to be meaningful

## Best Practices

### 1. OpenAI Usage
- **Rate Limiting**: Implement exponential backoff
- **Cost Management**: Monitor token usage and costs
- **Fallback Strategy**: Always have traditional recommendations as backup
- **Prompt Engineering**: Use consistent, well-tested prompts

### 2. Vector Search Optimization
- **Index Configuration**: Use HNSW for optimal performance
- **Embedding Quality**: Ensure high-quality vector representations
- **Search Relevance**: Combine semantic and keyword search
- **Index Maintenance**: Regular reindexing for freshness

### 3. A/B Testing
- **Consistent Assignment**: Use stable hashing for user assignment
- **Gradual Rollout**: Start with small traffic percentages
- **Monitoring**: Continuous monitoring of key metrics
- **Quick Rollback**: Ability to disable tests immediately

### 4. Performance Optimization
- **Caching**: Cache embeddings and frequently accessed data
- **Async Processing**: Use async/await for concurrent operations
- **Batch Processing**: Group operations for efficiency
- **Resource Management**: Proper connection pooling and cleanup

## Troubleshooting

### Common Issues

#### OpenAI Service Issues
```bash
# Check OpenAI service status
curl -H "Authorization: Bearer $AZURE_OPENAI_API_KEY" \
     "$AZURE_OPENAI_ENDPOINT/openai/deployments?api-version=2024-02-01"
```

#### Search Service Issues
```bash
# Check search service status
curl -H "api-key: $AZURE_SEARCH_API_KEY" \
     "$AZURE_SEARCH_ENDPOINT/indexes?api-version=2023-11-01"
```

#### A/B Test Issues
- **Traffic Not Splitting**: Check hash function and user ID consistency
- **Metrics Not Recording**: Verify storage and logging configuration
- **Performance Degradation**: Monitor resource usage and scaling

### Monitoring and Alerts

#### Application Insights Queries
```kusto
// OpenAI recommendation performance
requests
| where name contains "openai/recommendations"
| summarize avg(duration), count() by bin(timestamp, 1h)

// A/B test distribution
customEvents
| where name == "ABTestAssignment"
| summarize count() by tostring(customDimensions.variant)

// Error rates by algorithm
exceptions
| where outerMessage contains "recommendation"
| summarize count() by tostring(customDimensions.algorithm)
```

## Future Enhancements

### 1. Advanced AI Features
- **Multi-Modal Recommendations**: Image and text analysis
- **Conversational Interface**: Chat-based recommendation discovery
- **Real-Time Learning**: Dynamic preference adaptation

### 2. Enhanced A/B Testing
- **Multi-Armed Bandit**: Dynamic traffic allocation
- **Bayesian Optimization**: Automated hyperparameter tuning
- **Causal Inference**: Understanding recommendation impact

### 3. Performance Improvements
- **Edge Deployment**: CDN-based recommendation caching
- **GPU Acceleration**: Faster embedding generation
- **Federated Learning**: Privacy-preserving personalization

### 4. Integration Enhancements
- **Real-Time Events**: Stream processing for immediate updates
- **External Data Sources**: Social media and behavioral data
- **Cross-Platform**: Mobile app and web consistency