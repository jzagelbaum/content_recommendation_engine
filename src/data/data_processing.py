"""
Data Processing Pipeline for Content Recommendation Engine
=========================================================

This module contains Synapse Analytics / Spark-based data processing pipelines for:
- Data ingestion and validation
- Feature engineering and transformation
- User behavior analysis
- Content analysis and enrichment
- Real-time streaming data processing

Author: Content Recommendation Engine Team
Date: October 2025
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.stat import Correlation
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

# Add the src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Main data processing class for the recommendation engine
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the data processor
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.spark = None
        self._init_spark_session()
    
    def _init_spark_session(self):
        """Initialize Spark session with appropriate configuration"""
        app_name = f"ContentRecommendation-{self.config.environment}"
        
        builder = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        
        # Add Azure storage configuration if available
        if self.config.azure.storage_connection_string:
            builder = builder.config(
                "fs.azure.account.key." + self.config.azure.data_lake_account_name + ".dfs.core.windows.net",
                self._extract_storage_key()
            )
        
        self.spark = builder.getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        
        logger.info(f"Spark session initialized: {app_name}")
    
    def _extract_storage_key(self) -> str:
        """Extract storage account key from connection string"""
        conn_str = self.config.azure.storage_connection_string
        if "AccountKey=" in conn_str:
            return conn_str.split("AccountKey=")[1].split(";")[0]
        return ""
    
    def ingest_raw_data(self, data_sources: Dict[str, str]) -> Dict[str, any]:
        """
        Ingest raw data from various sources
        
        Args:
            data_sources: Dictionary mapping data type to source path
            
        Returns:
            Dictionary of DataFrames
        """
        logger.info("Starting raw data ingestion...")
        
        dataframes = {}
        
        for data_type, source_path in data_sources.items():
            try:
                if source_path.endswith('.parquet'):
                    df = self.spark.read.parquet(source_path)
                elif source_path.endswith('.csv'):
                    df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(source_path)
                elif source_path.endswith('.json'):
                    df = self.spark.read.json(source_path)
                else:
                    logger.warning(f"Unsupported file format for {data_type}: {source_path}")
                    continue
                
                # Add ingestion metadata
                df = df.withColumn("ingestion_timestamp", current_timestamp()) \
                       .withColumn("source_file", lit(source_path))
                
                dataframes[data_type] = df
                logger.info(f"Ingested {data_type}: {df.count()} records from {source_path}")
                
            except Exception as e:
                logger.error(f"Failed to ingest {data_type} from {source_path}: {e}")
        
        return dataframes
    
    def validate_and_clean_interactions(self, interactions_df) -> any:
        """
        Validate and clean user interaction data
        
        Args:
            interactions_df: Raw interactions DataFrame
            
        Returns:
            Cleaned interactions DataFrame
        """
        logger.info("Validating and cleaning interaction data...")
        
        # Define schema validation
        required_columns = ['user_id', 'item_id', 'rating', 'timestamp']
        for col in required_columns:
            if col not in interactions_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Data cleaning and validation
        cleaned_df = interactions_df \
            .filter(col("user_id").isNotNull() & col("item_id").isNotNull()) \
            .filter(col("rating").between(1, 5)) \
            .filter(col("timestamp").isNotNull()) \
            .dropDuplicates(["user_id", "item_id", "timestamp"])
        
        # Convert timestamp to proper format
        cleaned_df = cleaned_df.withColumn(
            "timestamp", 
            when(col("timestamp").cast("timestamp").isNull(), 
                 to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss"))
            .otherwise(col("timestamp").cast("timestamp"))
        )
        
        # Add derived features
        cleaned_df = cleaned_df \
            .withColumn("rating_date", to_date(col("timestamp"))) \
            .withColumn("rating_hour", hour(col("timestamp"))) \
            .withColumn("rating_dayofweek", dayofweek(col("timestamp"))) \
            .withColumn("is_weekend", when(dayofweek(col("timestamp")).isin([1, 7]), 1).otherwise(0))
        
        # Filter out users/items with insufficient interactions
        min_interactions = self.config.data.min_interactions
        
        user_counts = cleaned_df.groupBy("user_id").count().filter(col("count") >= min_interactions)
        item_counts = cleaned_df.groupBy("item_id").count().filter(col("count") >= min_interactions)
        
        final_df = cleaned_df \
            .join(user_counts.select("user_id"), "user_id", "inner") \
            .join(item_counts.select("item_id"), "item_id", "inner")
        
        logger.info(f"Cleaned interactions: {final_df.count()} records, "
                   f"{final_df.select('user_id').distinct().count()} users, "
                   f"{final_df.select('item_id').distinct().count()} items")
        
        return final_df
    
    def validate_and_clean_content(self, content_df) -> any:
        """
        Validate and clean content metadata
        
        Args:
            content_df: Raw content DataFrame
            
        Returns:
            Cleaned content DataFrame
        """
        logger.info("Validating and cleaning content data...")
        
        # Ensure required columns exist
        text_columns = ['title', 'description', 'genre', 'tags']
        for col_name in text_columns:
            if col_name not in content_df.columns:
                content_df = content_df.withColumn(col_name, lit(""))
        
        # Clean and standardize text fields
        cleaned_df = content_df \
            .filter(col("item_id").isNotNull()) \
            .dropDuplicates(["item_id"])
        
        # Clean text fields
        for col_name in text_columns:
            cleaned_df = cleaned_df \
                .withColumn(col_name, coalesce(col(col_name), lit(""))) \
                .withColumn(col_name, trim(col(col_name))) \
                .withColumn(col_name, regexp_replace(col(col_name), "[^a-zA-Z0-9\\s\\-,.]", ""))
        
        # Add derived features
        cleaned_df = cleaned_df \
            .withColumn("title_length", length(col("title"))) \
            .withColumn("description_length", length(col("description"))) \
            .withColumn("has_description", when(length(col("description")) > 0, 1).otherwise(0)) \
            .withColumn("genre_primary", split(col("genre"), ",").getItem(0)) \
            .withColumn("tag_count", size(split(col("tags"), ",")))
        
        # Handle missing numerical fields
        if "release_year" in cleaned_df.columns:
            current_year = datetime.now().year
            cleaned_df = cleaned_df \
                .withColumn("release_year", 
                           when(col("release_year").between(1900, current_year), col("release_year"))
                           .otherwise(None)) \
                .withColumn("content_age", current_year - col("release_year"))
        
        if "duration_minutes" in cleaned_df.columns:
            cleaned_df = cleaned_df \
                .withColumn("duration_minutes", 
                           when(col("duration_minutes").between(1, 1000), col("duration_minutes"))
                           .otherwise(None))
        
        logger.info(f"Cleaned content: {cleaned_df.count()} items")
        
        return cleaned_df
    
    def generate_user_features(self, interactions_df) -> any:
        """
        Generate user behavior features
        
        Args:
            interactions_df: User interactions DataFrame
            
        Returns:
            User features DataFrame
        """
        logger.info("Generating user features...")
        
        # Window functions for user statistics
        user_window = Window.partitionBy("user_id")
        
        # Basic user statistics
        user_stats = interactions_df \
            .groupBy("user_id") \
            .agg(
                count("*").alias("total_interactions"),
                avg("rating").alias("avg_rating"),
                stddev("rating").alias("rating_std"),
                min("rating").alias("min_rating"),
                max("rating").alias("max_rating"),
                countDistinct("item_id").alias("unique_items"),
                min("timestamp").alias("first_interaction"),
                max("timestamp").alias("last_interaction")
            )
        
        # Add derived features
        user_stats = user_stats \
            .withColumn("rating_range", col("max_rating") - col("min_rating")) \
            .withColumn("days_active", 
                       datediff(col("last_interaction"), col("first_interaction")) + 1) \
            .withColumn("interactions_per_day", 
                       col("total_interactions") / greatest(col("days_active"), lit(1))) \
            .withColumn("rating_std", coalesce(col("rating_std"), lit(0.0)))
        
        # Genre preferences
        genre_prefs = interactions_df \
            .join(self._get_content_genres(), "item_id", "left") \
            .filter(col("genre_primary").isNotNull()) \
            .groupBy("user_id", "genre_primary") \
            .agg(
                count("*").alias("genre_count"),
                avg("rating").alias("genre_avg_rating")
            ) \
            .withColumn("rank", row_number().over(
                Window.partitionBy("user_id").orderBy(desc("genre_count"))
            )) \
            .filter(col("rank") == 1) \
            .select("user_id", 
                   col("genre_primary").alias("favorite_genre"),
                   col("genre_avg_rating").alias("favorite_genre_rating"))
        
        # Temporal patterns
        temporal_features = interactions_df \
            .groupBy("user_id") \
            .agg(
                avg("rating_hour").alias("avg_interaction_hour"),
                mode("rating_dayofweek").alias("most_active_day"),
                avg("is_weekend").alias("weekend_preference")
            )
        
        # Combine all user features
        user_features = user_stats \
            .join(genre_prefs, "user_id", "left") \
            .join(temporal_features, "user_id", "left")
        
        # Fill missing values
        user_features = user_features \
            .fillna({
                "favorite_genre": "Unknown",
                "favorite_genre_rating": 0.0,
                "weekend_preference": 0.0
            })
        
        logger.info(f"Generated features for {user_features.count()} users")
        
        return user_features
    
    def generate_item_features(self, interactions_df, content_df) -> any:
        """
        Generate item popularity and content features
        
        Args:
            interactions_df: User interactions DataFrame
            content_df: Content metadata DataFrame
            
        Returns:
            Item features DataFrame
        """
        logger.info("Generating item features...")
        
        # Basic item statistics from interactions
        item_stats = interactions_df \
            .groupBy("item_id") \
            .agg(
                count("*").alias("total_ratings"),
                countDistinct("user_id").alias("unique_users"),
                avg("rating").alias("avg_rating"),
                stddev("rating").alias("rating_std"),
                min("timestamp").alias("first_rating"),
                max("timestamp").alias("last_rating")
            )
        
        # Add derived popularity features
        item_stats = item_stats \
            .withColumn("rating_std", coalesce(col("rating_std"), lit(0.0))) \
            .withColumn("days_since_first_rating", 
                       datediff(current_date(), col("first_rating"))) \
            .withColumn("days_since_last_rating", 
                       datediff(current_date(), col("last_rating"))) \
            .withColumn("ratings_per_day", 
                       col("total_ratings") / greatest(
                           datediff(col("last_rating"), col("first_rating")) + 1, lit(1)
                       ))
        
        # Calculate popularity percentiles
        popularity_percentiles = item_stats \
            .select("item_id", "total_ratings") \
            .withColumn("popularity_percentile", 
                       percent_rank().over(Window.orderBy("total_ratings")))
        
        # Combine with content features
        item_features = content_df \
            .join(item_stats, "item_id", "left") \
            .join(popularity_percentiles, "item_id", "left") \
            .fillna({
                "total_ratings": 0,
                "unique_users": 0,
                "avg_rating": 0.0,
                "rating_std": 0.0,
                "popularity_percentile": 0.0,
                "ratings_per_day": 0.0
            })
        
        # Add content-based features
        if "genre" in item_features.columns:
            # Genre encoding
            genre_indexer = StringIndexer(inputCol="genre_primary", outputCol="genre_index")
            item_features = genre_indexer.fit(item_features).transform(item_features)
        
        logger.info(f"Generated features for {item_features.count()} items")
        
        return item_features
    
    def _get_content_genres(self):
        """Helper method to get content genres (placeholder for actual implementation)"""
        # This should be replaced with actual content data join
        return self.spark.createDataFrame([], StructType([
            StructField("item_id", IntegerType(), True),
            StructField("genre_primary", StringType(), True)
        ]))
    
    def create_interaction_matrix(self, interactions_df) -> any:
        """
        Create user-item interaction matrix for collaborative filtering
        
        Args:
            interactions_df: User interactions DataFrame
            
        Returns:
            Interaction matrix DataFrame
        """
        logger.info("Creating interaction matrix...")
        
        # Pivot interactions to create user-item matrix
        matrix_df = interactions_df \
            .groupBy("user_id") \
            .pivot("item_id") \
            .agg(avg("rating")) \
            .fillna(0.0)
        
        logger.info(f"Created interaction matrix: {matrix_df.count()} users")
        
        return matrix_df
    
    def calculate_item_similarity(self, interactions_df) -> any:
        """
        Calculate item-item similarity matrix
        
        Args:
            interactions_df: User interactions DataFrame
            
        Returns:
            Item similarity DataFrame
        """
        logger.info("Calculating item similarity...")
        
        # Create item vectors based on user ratings
        item_vectors = interactions_df \
            .groupBy("item_id") \
            .pivot("user_id") \
            .agg(avg("rating")) \
            .fillna(0.0)
        
        # This is a simplified version - in practice, you'd want to use
        # more sophisticated similarity calculations
        logger.info("Item similarity calculation completed")
        
        return item_vectors
    
    def detect_trending_content(self, interactions_df, content_df, 
                               time_window_days: int = 7) -> any:
        """
        Detect trending content based on recent interaction patterns
        
        Args:
            interactions_df: User interactions DataFrame
            content_df: Content metadata DataFrame
            time_window_days: Time window for trend detection
            
        Returns:
            Trending content DataFrame
        """
        logger.info(f"Detecting trending content (last {time_window_days} days)...")
        
        # Define time window
        cutoff_date = current_date() - expr(f"INTERVAL {time_window_days} DAYS")
        
        # Calculate recent activity
        recent_interactions = interactions_df \
            .filter(col("rating_date") >= cutoff_date) \
            .groupBy("item_id") \
            .agg(
                count("*").alias("recent_interactions"),
                countDistinct("user_id").alias("recent_unique_users"),
                avg("rating").alias("recent_avg_rating")
            )
        
        # Calculate trend score
        trending_items = recent_interactions \
            .withColumn("trend_score", 
                       col("recent_interactions") * col("recent_avg_rating") * 
                       log(col("recent_unique_users") + 1)) \
            .withColumn("trending_rank", 
                       row_number().over(Window.orderBy(desc("trend_score")))) \
            .filter(col("trending_rank") <= 100)  # Top 100 trending items
        
        # Join with content metadata
        trending_content = trending_items \
            .join(content_df.select("item_id", "title", "genre", "release_year"), 
                  "item_id", "inner") \
            .orderBy(desc("trend_score"))
        
        logger.info(f"Identified {trending_content.count()} trending items")
        
        return trending_content
    
    def save_processed_data(self, dataframes: Dict[str, any], output_path: str):
        """
        Save processed data to storage
        
        Args:
            dataframes: Dictionary of DataFrames to save
            output_path: Base output path
        """
        logger.info(f"Saving processed data to {output_path}...")
        
        for name, df in dataframes.items():
            try:
                output_file_path = f"{output_path}/{name}"
                
                # Write as Parquet with partitioning for large datasets
                if name in ['interactions', 'user_features']:
                    df.write.mode("overwrite").partitionBy("user_id") \
                      .parquet(output_file_path)
                else:
                    df.write.mode("overwrite").parquet(output_file_path)
                
                logger.info(f"Saved {name}: {df.count()} records to {output_file_path}")
                
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")
    
    def run_full_pipeline(self, input_paths: Dict[str, str], output_path: str):
        """
        Run the complete data processing pipeline
        
        Args:
            input_paths: Dictionary mapping data types to input paths
            output_path: Base output path for processed data
        """
        logger.info("Starting full data processing pipeline...")
        
        try:
            # Step 1: Ingest raw data
            raw_data = self.ingest_raw_data(input_paths)
            
            # Step 2: Validate and clean data
            if 'interactions' in raw_data:
                clean_interactions = self.validate_and_clean_interactions(raw_data['interactions'])
            else:
                raise ValueError("Interactions data is required")
            
            if 'content' in raw_data:
                clean_content = self.validate_and_clean_content(raw_data['content'])
            else:
                raise ValueError("Content data is required")
            
            # Step 3: Generate features
            user_features = self.generate_user_features(clean_interactions)
            item_features = self.generate_item_features(clean_interactions, clean_content)
            
            # Step 4: Create matrices and similarities
            interaction_matrix = self.create_interaction_matrix(clean_interactions)
            
            # Step 5: Detect trending content
            trending_content = self.detect_trending_content(clean_interactions, clean_content)
            
            # Step 6: Save processed data
            processed_data = {
                'interactions': clean_interactions,
                'content': clean_content,
                'user_features': user_features,
                'item_features': item_features,
                'interaction_matrix': interaction_matrix,
                'trending_content': trending_content
            }
            
            self.save_processed_data(processed_data, output_path)
            
            logger.info("Data processing pipeline completed successfully")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Data processing pipeline failed: {e}")
            raise
    
    def stop(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


def main():
    """Main function for running data processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data processing pipeline")
    parser.add_argument("--interactions-path", required=True, help="Path to interactions data")
    parser.add_argument("--content-path", required=True, help="Path to content data")
    parser.add_argument("--output-path", required=True, help="Output path for processed data")
    parser.add_argument("--config-path", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Initialize configuration and processor
    config = Config(args.config_path) if args.config_path else Config()
    processor = DataProcessor(config)
    
    try:
        # Define input paths
        input_paths = {
            'interactions': args.interactions_path,
            'content': args.content_path
        }
        
        # Run pipeline
        processor.run_full_pipeline(input_paths, args.output_path)
        
    finally:
        processor.stop()


if __name__ == "__main__":
    main()