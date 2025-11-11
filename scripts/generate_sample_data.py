#!/usr/bin/env python3
"""
scripts/generate_sample_data.py
Generate sample data for testing the recommendation system
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SampleDataGenerator:
    """Generate realistic sample data for recommendation system"""
    
    CATEGORIES = [
        "electronics", "books", "clothing", "home", "sports",
        "beauty", "toys", "food", "automotive", "health"
    ]
    
    PRODUCT_PREFIXES = {
        "electronics": ["Smart", "Wireless", "Pro", "Ultra", "Premium"],
        "books": ["The Art of", "Guide to", "Introduction to", "Mastering", "Complete"],
        "clothing": ["Classic", "Modern", "Vintage", "Designer", "Casual"],
        "home": ["Luxury", "Comfort", "Essential", "Deluxe", "Compact"],
        "sports": ["Performance", "Professional", "Training", "Advanced", "Elite"]
    }
    
    def __init__(self, num_users: int, num_products: int, num_interactions: int):
        self.num_users = num_users
        self.num_products = num_products
        self.num_interactions = num_interactions
        
    def generate_users(self) -> List[Dict[str, Any]]:
        """Generate sample user data"""
        logger.info(f"Generating {self.num_users} users...")
        users = []
        
        for i in range(self.num_users):
            user = {
                "user_id": f"user_{i:06d}",
                "age": random.randint(18, 75),
                "preferences": random.sample(self.CATEGORIES, k=random.randint(2, 5)),
                "created_at": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                "is_premium": random.random() < 0.3
            }
            users.append(user)
        
        logger.info(f"✓ Generated {len(users)} users")
        return users
    
    def generate_products(self) -> List[Dict[str, Any]]:
        """Generate sample product data"""
        logger.info(f"Generating {self.num_products} products...")
        products = []
        
        for i in range(self.num_products):
            category = random.choice(self.CATEGORIES)
            prefix = random.choice(self.PRODUCT_PREFIXES.get(category, ["Premium"]))
            
            product = {
                "product_id": f"prod_{i:06d}",
                "name": f"{prefix} {category.capitalize()} Item {i}",
                "category": category,
                "price": round(random.uniform(9.99, 999.99), 2),
                "rating": round(random.uniform(3.0, 5.0), 1),
                "description": f"A high-quality {category} product with excellent features",
                "in_stock": random.random() < 0.85,
                "created_at": (datetime.now() - timedelta(days=random.randint(1, 730))).isoformat()
            }
            products.append(product)
        
        logger.info(f"✓ Generated {len(products)} products")
        return products
    
    def generate_interactions(
        self, 
        users: List[Dict[str, Any]], 
        products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate sample user-product interactions"""
        logger.info(f"Generating {self.num_interactions} interactions...")
        interactions = []
        
        interaction_types = ["view", "click", "add_to_cart", "purchase", "review"]
        weights = [0.5, 0.25, 0.15, 0.08, 0.02]
        
        for i in range(self.num_interactions):
            user = random.choice(users)
            product = random.choice(products)
            
            # Users more likely to interact with products in their preferred categories
            if product["category"] in user["preferences"]:
                interaction_type = random.choices(
                    interaction_types, 
                    weights=[0.3, 0.25, 0.2, 0.15, 0.1]
                )[0]
            else:
                interaction_type = random.choices(interaction_types, weights=weights)[0]
            
            interaction = {
                "interaction_id": f"int_{i:08d}",
                "user_id": user["user_id"],
                "product_id": product["product_id"],
                "type": interaction_type,
                "timestamp": (datetime.now() - timedelta(
                    days=random.randint(0, 90),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )).isoformat(),
                "session_id": f"session_{random.randint(1, self.num_users * 5)}"
            }
            
            # Add purchase-specific data
            if interaction_type == "purchase":
                interaction["quantity"] = random.randint(1, 3)
                interaction["total_price"] = round(
                    product["price"] * interaction["quantity"], 2
                )
            
            # Add review-specific data
            if interaction_type == "review":
                interaction["rating"] = random.randint(1, 5)
                interaction["review_text"] = f"Sample review for {product['name']}"
            
            interactions.append(interaction)
        
        logger.info(f"✓ Generated {len(interactions)} interactions")
        return interactions
    
    def save_data(
        self, 
        users: List[Dict[str, Any]], 
        products: List[Dict[str, Any]], 
        interactions: List[Dict[str, Any]],
        output_dir: Path
    ):
        """Save generated data to JSON files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {
            "users.json": users,
            "products.json": products,
            "interactions.json": interactions
        }
        
        for filename, data in files.items():
            filepath = output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"✓ Saved {filename} ({len(data)} records)")
        
        # Save summary
        summary = {
            "generated_at": datetime.now().isoformat(),
            "num_users": len(users),
            "num_products": len(products),
            "num_interactions": len(interactions),
            "categories": self.CATEGORIES
        }
        
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved summary.json")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample data for recommendation system"
    )
    parser.add_argument(
        "-u", "--users",
        type=int,
        default=1000,
        help="Number of users to generate (default: 1000)"
    )
    parser.add_argument(
        "-p", "--products",
        type=int,
        default=500,
        help="Number of products to generate (default: 500)"
    )
    parser.add_argument(
        "-i", "--interactions",
        type=int,
        default=5000,
        help="Number of interactions to generate (default: 5000)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting sample data generation...")
    logger.info(f"  Users: {args.users}")
    logger.info(f"  Products: {args.products}")
    logger.info(f"  Interactions: {args.interactions}")
    logger.info(f"  Output: {args.output}")
    
    generator = SampleDataGenerator(
        num_users=args.users,
        num_products=args.products,
        num_interactions=args.interactions
    )
    
    users = generator.generate_users()
    products = generator.generate_products()
    interactions = generator.generate_interactions(users, products)
    
    output_dir = Path(args.output)
    generator.save_data(users, products, interactions, output_dir)
    
    logger.info("✅ Sample data generation complete!")
    logger.info(f"📁 Data saved to: {output_dir}")


if __name__ == "__main__":
    main()
