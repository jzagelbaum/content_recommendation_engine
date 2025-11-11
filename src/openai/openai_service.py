"""
Azure OpenAI Service Integration
Provides core functionality for interacting with Azure OpenAI
"""

import os
import asyncio
import json
from typing import List, Dict, Any, Optional
from openai import AsyncAzureOpenAI
import logging

logger = logging.getLogger(__name__)

class AzureOpenAIService:
    """Service for interacting with Azure OpenAI"""
    
    def __init__(self):
        """Initialize Azure OpenAI client"""
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure OpenAI endpoint and API key must be provided")
        
        self.client = AsyncAzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        self.gpt_deployment = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text"""
        try:
            # Truncate text if too long
            if len(text) > 8000:
                text = text[:8000]
            
            response = await self.client.embeddings.create(
                input=text,
                model=self.embedding_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        try:
            all_embeddings = []
            
            # Process in batches to avoid rate limits
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Truncate texts if too long
                truncated_batch = [text[:8000] if len(text) > 8000 else text for text in batch]
                
                response = await self.client.embeddings.create(
                    input=truncated_batch,
                    model=self.embedding_deployment
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """Generate completion using GPT model"""
        try:
            response = await self.client.chat.completions.create(
                model=self.gpt_deployment,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise
    
    async def generate_recommendation_explanation(
        self, 
        user_profile: Dict[str, Any], 
        recommended_items: List[Dict[str, Any]],
        algorithm_used: str = "hybrid"
    ) -> str:
        """Generate natural language explanation for recommendations"""
        try:
            prompt = self._build_explanation_prompt(user_profile, recommended_items, algorithm_used)
            
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that explains product recommendations based on user preferences and behavior. Provide concise, friendly explanations that help users understand why items were recommended."
                },
                {"role": "user", "content": prompt}
            ]
            
            explanation = await self.generate_completion(messages, max_tokens=150, temperature=0.5)
            return explanation
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Based on your preferences and browsing history, these items were selected using our {algorithm_used} recommendation algorithm."
    
    async def generate_synthetic_user_data(
        self, 
        num_users: int = 100, 
        categories: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate synthetic user data for testing"""
        if categories is None:
            categories = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance", "Documentary"]
        
        try:
            prompt = f"""
            Generate {min(num_users, 20)} diverse synthetic user profiles for a streaming content recommendation system.
            Include the following content categories: {', '.join(categories)}
            
            For each user, provide a JSON object with:
            - user_id: unique identifier (string)
            - age: age in years (18-70)
            - preferences: array of 2-4 preferred categories
            - viewing_history: array of 3-5 recently watched content titles
            - viewing_patterns: object with avg_session_duration_minutes and preferred_time_of_day
            - subscription_type: "basic", "premium", or "family"
            
            Return as a valid JSON array of user objects.
            """
            
            messages = [
                {
                    "role": "system", 
                    "content": "You are a data generator that creates realistic synthetic user profiles for testing recommendation systems. Always return valid JSON."
                },
                {"role": "user", "content": prompt}
            ]
            
            response_text = await self.generate_completion(messages, max_tokens=2000, temperature=0.8)
            
            # Parse JSON response
            try:
                synthetic_data = json.loads(response_text)
                
                # If we need more users, generate additional batches
                if num_users > 20:
                    remaining_users = num_users - len(synthetic_data)
                    while remaining_users > 0:
                        batch_size = min(remaining_users, 20)
                        additional_batch = await self.generate_synthetic_user_data(batch_size, categories)
                        synthetic_data.extend(additional_batch)
                        remaining_users -= batch_size
                
                return synthetic_data[:num_users]
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, returning empty list")
                return []
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return []
    
    async def analyze_content_for_features(self, content_description: str) -> Dict[str, Any]:
        """Analyze content to extract features for recommendation"""
        try:
            prompt = f"""
            Analyze the following content description and extract key features for a recommendation system:
            
            Content: {content_description}
            
            Return a JSON object with:
            - genres: array of relevant genres
            - mood: string describing the overall mood (e.g., "dark", "uplifting", "suspenseful")
            - themes: array of main themes
            - target_audience: string describing target demographic
            - content_rating: estimated content rating
            - keywords: array of relevant keywords for search
            
            Respond with valid JSON only.
            """
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a content analyzer that extracts features from content descriptions for recommendation systems. Always return valid JSON."
                },
                {"role": "user", "content": prompt}
            ]
            
            response_text = await self.generate_completion(messages, max_tokens=300, temperature=0.3)
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse content analysis JSON")
                return {
                    "genres": [],
                    "mood": "unknown",
                    "themes": [],
                    "target_audience": "general",
                    "content_rating": "unrated",
                    "keywords": []
                }
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {}
    
    def _build_explanation_prompt(
        self, 
        user_profile: Dict[str, Any], 
        recommended_items: List[Dict[str, Any]],
        algorithm_used: str
    ) -> str:
        """Build prompt for recommendation explanation"""
        # Extract user preferences
        preferences = user_profile.get('preferences', [])
        viewing_history = user_profile.get('viewing_history', [])
        age = user_profile.get('age', 'Unknown')
        
        # Format recommended items
        items_text = "\n".join([
            f"- {item.get('title', item.get('name', 'Item'))}: {item.get('genre', item.get('category', 'Unknown genre'))}"
            for item in recommended_items[:3]  # Top 3 items
        ])
        
        return f"""
        User Profile:
        - Age: {age}
        - Preferred genres: {', '.join(preferences) if preferences else 'Not specified'}
        - Recent viewing: {', '.join(viewing_history[-3:]) if viewing_history else 'No recent history'}
        
        Recommended Content:
        {items_text}
        
        Algorithm used: {algorithm_used}
        
        Please provide a brief, friendly explanation (1-2 sentences) of why these items were recommended to this user.
        """
    
    async def close(self):
        """Close the OpenAI client"""
        if hasattr(self.client, 'close'):
            await self.client.close()