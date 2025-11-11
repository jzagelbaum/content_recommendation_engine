"""
OpenAI-Powered Recommendation Engine
Implements recommendation logic using Azure OpenAI and vector search
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from .openai_service import AzureOpenAIService
from .embedding_service import EmbeddingService
import logging

logger = logging.getLogger(__name__)

class OpenAIRecommendationEngine:
    """OpenAI-powered recommendation engine using semantic similarity and AI insights"""
    
    def __init__(self):
        """Initialize the OpenAI recommendation engine"""
        self.openai_service = AzureOpenAIService()
        self.embedding_service = EmbeddingService()
        self.algorithm_version = "openai-hybrid-v1.0"
    
    async def initialize(self) -> bool:
        """Initialize the recommendation engine and search index"""
        try:
            success = await self.embedding_service.initialize_index()
            if success:
                logger.info("OpenAI recommendation engine initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Error initializing OpenAI recommendation engine: {e}")
            return False
    
    async def get_recommendations(
        self,
        user_id: str,
        user_profile: Dict[str, Any],
        num_recommendations: int = 10,
        context: Optional[Dict[str, Any]] = None,
        exclude_items: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate recommendations using OpenAI and vector search"""
        try:
            start_time = datetime.utcnow()
            
            # Build user context for semantic search
            user_context = self._build_user_context(user_profile, context)
            
            # Get content-based recommendations using vector similarity
            content_recommendations = await self._get_content_based_recommendations(
                user_context, 
                user_profile,
                num_recommendations * 2  # Get more for better filtering
            )
            
            # Get AI-powered insights and collaborative signals
            ai_insights = await self._get_ai_insights(
                user_profile,
                content_recommendations[:10],  # Use top 10 for analysis
                context
            )
            
            # Combine and rank recommendations
            final_recommendations = await self._combine_and_rank_recommendations(
                content_recommendations,
                ai_insights,
                user_profile,
                exclude_items
            )
            
            # Generate explanation
            explanation = await self._generate_explanation(
                user_profile,
                final_recommendations[:3],
                ai_insights
            )
            
            # Calculate confidence and diversity scores
            confidence_score = self._calculate_confidence_score(final_recommendations)
            diversity_score = self._calculate_diversity_score(final_recommendations)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "user_id": user_id,
                "recommendations": final_recommendations[:num_recommendations],
                "explanation": explanation,
                "algorithm_version": self.algorithm_version,
                "confidence_score": confidence_score,
                "diversity_score": diversity_score,
                "ai_insights": ai_insights.get("summary", ""),
                "processing_time_seconds": processing_time,
                "total_candidates_evaluated": len(content_recommendations),
                "generated_at": start_time.isoformat(),
                "metadata": {
                    "openai_model_used": self.openai_service.gpt_deployment,
                    "embedding_model_used": self.openai_service.embedding_deployment,
                    "search_method": "hybrid_vector_semantic"
                }
            }
        except Exception as e:
            logger.error(f"Error generating OpenAI recommendations: {e}")
            raise
    
    async def _get_content_based_recommendations(
        self,
        user_context: str,
        user_profile: Dict[str, Any],
        num_items: int
    ) -> List[Dict[str, Any]]:
        """Get content-based recommendations using vector similarity"""
        try:
            # Get hybrid recommendations from embedding service
            recommendations = await self.embedding_service.get_content_recommendations_hybrid(
                user_query=user_context,
                user_preferences=user_profile,
                top_k=num_items,
                vector_weight=0.7
            )
            
            # Enhance with popularity and freshness scoring
            for item in recommendations:
                item["content_score"] = item.get("final_score", 0.0)
                item["popularity_boost"] = min(item.get("popularity_score", 0.0) * 0.1, 0.2)
                item["enhanced_score"] = item["content_score"] + item["popularity_boost"]
            
            return sorted(recommendations, key=lambda x: x.get("enhanced_score", 0), reverse=True)
        except Exception as e:
            logger.error(f"Error getting content-based recommendations: {e}")
            return []
    
    async def _get_ai_insights(
        self,
        user_profile: Dict[str, Any],
        content_recommendations: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Use OpenAI to generate insights and additional recommendations"""
        try:
            # Build prompt for AI insights
            prompt = self._build_insights_prompt(user_profile, content_recommendations, context)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert recommendation system analyst. Based on user profiles and content recommendations, provide insights about:
1. User behavior patterns
2. Content preferences analysis  
3. Seasonal/trending recommendations
4. Complementary content suggestions
5. Discovery opportunities for new genres

Respond with a JSON object containing insights and additional recommendations."""
                },
                {"role": "user", "content": prompt}
            ]
            
            response_text = await self.openai_service.generate_completion(
                messages, 
                max_tokens=600, 
                temperature=0.3
            )
            
            # Parse AI insights
            insights = self._parse_ai_insights(response_text)
            return insights
        except Exception as e:
            logger.error(f"Error getting AI insights: {e}")
            return {"summary": "AI insights unavailable", "additional_recommendations": []}
    
    async def _combine_and_rank_recommendations(
        self,
        content_recommendations: List[Dict[str, Any]],
        ai_insights: Dict[str, Any],
        user_profile: Dict[str, Any],
        exclude_items: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Combine different recommendation sources and re-rank"""
        try:
            exclude_set = set(exclude_items or [])
            all_recommendations = {}
            
            # Process content-based recommendations
            for item in content_recommendations:
                item_id = item.get("id")
                if item_id and item_id not in exclude_set:
                    item["content_score"] = item.get("enhanced_score", 0.0) * 0.6
                    all_recommendations[item_id] = item
            
            # Process AI insights recommendations
            ai_recommendations = ai_insights.get("additional_recommendations", [])
            for item in ai_recommendations:
                item_id = item.get("id")
                if item_id and item_id not in exclude_set:
                    if item_id in all_recommendations:
                        all_recommendations[item_id]["ai_score"] = item.get("relevance_score", 0.0) * 0.3
                    else:
                        item["content_score"] = 0.0
                        item["ai_score"] = item.get("relevance_score", 0.0) * 0.3
                        all_recommendations[item_id] = item
            
            # Apply personalization boost
            for item in all_recommendations.values():
                personalization_boost = self._calculate_personalization_boost(item, user_profile)
                item["personalization_score"] = personalization_boost * 0.1
                
                # Calculate final score
                final_score = (
                    item.get("content_score", 0.0) +
                    item.get("ai_score", 0.0) +
                    item.get("personalization_score", 0.0)
                )
                item["final_recommendation_score"] = final_score
            
            # Sort by final score
            recommendations = list(all_recommendations.values())
            recommendations.sort(key=lambda x: x.get("final_recommendation_score", 0), reverse=True)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error combining recommendations: {e}")
            return content_recommendations
    
    async def _generate_explanation(
        self,
        user_profile: Dict[str, Any],
        top_recommendations: List[Dict[str, Any]],
        ai_insights: Dict[str, Any]
    ) -> str:
        """Generate personalized explanation for recommendations"""
        try:
            # Use AI insights for more contextual explanation
            insight_summary = ai_insights.get("summary", "")
            
            explanation = await self.openai_service.generate_recommendation_explanation(
                user_profile,
                top_recommendations,
                self.algorithm_version
            )
            
            # Enhance with AI insights if available
            if insight_summary:
                enhanced_explanation = f"{explanation} {insight_summary}"
                return enhanced_explanation[:500]  # Keep it concise
            
            return explanation
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "These recommendations were selected based on your preferences and viewing history."
    
    def _build_user_context(
        self, 
        user_profile: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build comprehensive user context for semantic search"""
        context_parts = []
        
        # Basic preferences
        if preferences := user_profile.get('preferences'):
            context_parts.append(f"Preferred genres: {', '.join(preferences)}")
        
        # Viewing history
        if viewing_history := user_profile.get('viewing_history'):
            recent_content = viewing_history[-5:] if len(viewing_history) > 5 else viewing_history
            context_parts.append(f"Recently watched: {', '.join(recent_content)}")
        
        # Demographic info
        if age := user_profile.get('age'):
            context_parts.append(f"Age group: {age}")
        
        # Viewing patterns
        if viewing_patterns := user_profile.get('viewing_patterns'):
            if preferred_time := viewing_patterns.get('preferred_time_of_day'):
                context_parts.append(f"Watches content during: {preferred_time}")
        
        # Current context
        if context:
            if mood := context.get('mood'):
                context_parts.append(f"Current mood: {mood}")
            if device := context.get('device'):
                context_parts.append(f"Viewing on: {device}")
            if session_type := context.get('session_type'):
                context_parts.append(f"Session type: {session_type}")
        
        return " ".join(context_parts)
    
    def _build_insights_prompt(
        self,
        user_profile: Dict[str, Any],
        content_recommendations: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for AI insights generation"""
        # Format user profile
        profile_text = []
        if age := user_profile.get('age'):
            profile_text.append(f"Age: {age}")
        if preferences := user_profile.get('preferences'):
            profile_text.append(f"Preferred genres: {', '.join(preferences)}")
        if viewing_history := user_profile.get('viewing_history'):
            profile_text.append(f"Recent viewing: {', '.join(viewing_history[-3:])}")
        
        # Format current recommendations
        rec_text = []
        for item in content_recommendations[:5]:
            title = item.get('title', 'Unknown')
            genre = item.get('genre', ['Unknown'])
            if isinstance(genre, list):
                genre = ', '.join(genre)
            rec_text.append(f"- {title} ({genre})")
        
        # Add context if available
        context_text = ""
        if context:
            context_parts = []
            if mood := context.get('mood'):
                context_parts.append(f"Mood: {mood}")
            if time_of_day := context.get('time_of_day'):
                context_parts.append(f"Time: {time_of_day}")
            context_text = f"\nCurrent context: {', '.join(context_parts)}" if context_parts else ""
        
        return f"""
        User Profile:
        {chr(10).join(profile_text)}
        {context_text}
        
        Current Content Recommendations:
        {chr(10).join(rec_text)}
        
        Analyze this user's preferences and provide:
        1. A brief summary of their viewing patterns
        2. 2-3 additional content suggestions that complement the current recommendations
        3. Any seasonal or trending opportunities
        
        Respond in JSON format:
        {{
            "summary": "Brief analysis of user preferences",
            "additional_recommendations": [
                {{"id": "item_id", "title": "content_title", "relevance_score": 0.8, "reason": "why recommended"}}
            ],
            "trending_opportunities": ["genre1", "genre2"],
            "user_segment": "segment description"
        }}
        """
    
    def _parse_ai_insights(self, response_text: str) -> Dict[str, Any]:
        """Parse AI insights from response text"""
        try:
            import json
            insights = json.loads(response_text)
            
            # Validate and sanitize the response
            if not isinstance(insights, dict):
                return {"summary": "Invalid AI response format", "additional_recommendations": []}
            
            # Ensure required fields exist
            insights.setdefault("summary", "")
            insights.setdefault("additional_recommendations", [])
            insights.setdefault("trending_opportunities", [])
            insights.setdefault("user_segment", "general")
            
            return insights
        except Exception as e:
            logger.warning(f"Failed to parse AI insights: {e}")
            return {
                "summary": "AI analysis completed",
                "additional_recommendations": [],
                "trending_opportunities": [],
                "user_segment": "general"
            }
    
    def _calculate_personalization_boost(
        self, 
        item: Dict[str, Any], 
        user_profile: Dict[str, Any]
    ) -> float:
        """Calculate personalization boost based on user profile match"""
        boost = 0.0
        
        # Genre preference match
        item_genres = item.get('genre', [])
        if isinstance(item_genres, str):
            item_genres = [item_genres]
        
        user_preferences = user_profile.get('preferences', [])
        
        if item_genres and user_preferences:
            genre_matches = len(set(item_genres) & set(user_preferences))
            boost += genre_matches / len(user_preferences) * 0.5
        
        # Rating preference
        item_rating = item.get('rating', 0.0)
        user_min_rating = user_profile.get('min_rating', 0.0)
        if item_rating >= user_min_rating:
            boost += 0.2
        
        # Duration preference
        item_duration = item.get('duration_minutes', 0)
        preferred_duration = user_profile.get('preferred_duration')
        if preferred_duration and item_duration:
            duration_diff = abs(item_duration - preferred_duration) / preferred_duration
            if duration_diff < 0.3:  # Within 30% of preferred duration
                boost += 0.1
        
        return min(boost, 1.0)
    
    def _calculate_confidence_score(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the recommendations"""
        if not recommendations:
            return 0.0
        
        # Use the top 3 recommendations for confidence calculation
        top_scores = [item.get('final_recommendation_score', 0.0) for item in recommendations[:3]]
        
        if not top_scores:
            return 0.0
        
        # Average of top scores, normalized
        avg_score = sum(top_scores) / len(top_scores)
        
        # Confidence is higher when scores are consistently high
        score_variance = sum((score - avg_score) ** 2 for score in top_scores) / len(top_scores)
        consistency_bonus = max(0, 0.2 - score_variance)
        
        confidence = min(avg_score + consistency_bonus, 1.0)
        return max(confidence, 0.0)
    
    def _calculate_diversity_score(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate diversity score for the recommendations"""
        if len(recommendations) < 2:
            return 0.0
        
        # Collect genres from recommendations
        all_genres = []
        for item in recommendations[:10]:  # Check top 10
            genres = item.get('genre', [])
            if isinstance(genres, str):
                genres = [genres]
            all_genres.extend(genres)
        
        if not all_genres:
            return 0.0
        
        # Calculate genre diversity
        unique_genres = len(set(all_genres))
        total_items = len(recommendations[:10])
        
        # Diversity score: unique genres / total items, capped at 1.0
        diversity = min(unique_genres / total_items, 1.0)
        
        return diversity
    
    async def close(self):
        """Close the recommendation engine and cleanup resources"""
        await self.openai_service.close()
        await self.embedding_service.close()