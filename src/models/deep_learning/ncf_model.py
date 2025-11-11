"""
Neural Collaborative Filtering (NCF) Model
==========================================

Implementation of Neural Collaborative Filtering as described in:
"Neural Collaborative Filtering" (He et al., 2017)
https://arxiv.org/abs/1708.05031

This model learns non-linear user-item interactions using deep neural networks,
outperforming traditional matrix factorization (SVD) especially for implicit feedback.

Architecture:
    User ID → Embedding (64d) ┐
                              ├→ MLP → Concat → Dense → Prediction
    Item ID → Embedding (64d) ┘

Key Features:
- Generalized Matrix Factorization (GMF) path for linear interactions
- Multi-Layer Perceptron (MLP) path for non-linear interactions
- Fusion layer combining GMF + MLP
- Better cold-start handling with side features

Author: Content Recommendation Engine Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional, Any
import logging
import mlflow
import mlflow.tensorflow
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class NeuralCollaborativeFiltering:
    """
    Neural Collaborative Filtering (NCF) model for recommendation
    
    Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
    to learn both linear and non-linear user-item interactions.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        mlp_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-5,
        learning_rate: float = 0.001,
        use_gmf: bool = True,
        use_mlp: bool = True
    ):
        """
        Initialize NCF model
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            embedding_dim: Dimension of user/item embeddings
            mlp_layers: List of hidden layer sizes for MLP path
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
            learning_rate: Learning rate for optimizer
            use_gmf: Whether to use GMF (linear) path
            use_mlp: Whether to use MLP (non-linear) path
        """
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.mlp_layers = mlp_layers
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.use_gmf = use_gmf
        self.use_mlp = use_mlp
        
        self.model = None
        self.user_encoder = {}
        self.item_encoder = {}
        self.user_decoder = {}
        self.item_decoder = {}
        self.history = None
        
    def build_model(self) -> Model:
        """
        Build the NCF model architecture
        
        Returns:
            Compiled Keras model
        """
        # Input layers
        user_input = layers.Input(shape=(1,), dtype=tf.int32, name='user_id')
        item_input = layers.Input(shape=(1,), dtype=tf.int32, name='item_id')
        
        paths = []
        
        # GMF Path (Generalized Matrix Factorization)
        if self.use_gmf:
            # GMF user embedding
            gmf_user_embedding = layers.Embedding(
                input_dim=self.num_users,
                output_dim=self.embedding_dim,
                embeddings_regularizer=regularizers.l2(self.l2_reg),
                name='gmf_user_embedding'
            )(user_input)
            gmf_user_vec = layers.Flatten(name='gmf_user_flatten')(gmf_user_embedding)
            
            # GMF item embedding
            gmf_item_embedding = layers.Embedding(
                input_dim=self.num_items,
                output_dim=self.embedding_dim,
                embeddings_regularizer=regularizers.l2(self.l2_reg),
                name='gmf_item_embedding'
            )(item_input)
            gmf_item_vec = layers.Flatten(name='gmf_item_flatten')(gmf_item_embedding)
            
            # Element-wise product (like matrix factorization)
            gmf_vector = layers.Multiply(name='gmf_multiply')([gmf_user_vec, gmf_item_vec])
            paths.append(gmf_vector)
        
        # MLP Path (Multi-Layer Perceptron)
        if self.use_mlp:
            # MLP user embedding
            mlp_user_embedding = layers.Embedding(
                input_dim=self.num_users,
                output_dim=self.embedding_dim,
                embeddings_regularizer=regularizers.l2(self.l2_reg),
                name='mlp_user_embedding'
            )(user_input)
            mlp_user_vec = layers.Flatten(name='mlp_user_flatten')(mlp_user_embedding)
            
            # MLP item embedding
            mlp_item_embedding = layers.Embedding(
                input_dim=self.num_items,
                output_dim=self.embedding_dim,
                embeddings_regularizer=regularizers.l2(self.l2_reg),
                name='mlp_item_embedding'
            )(item_input)
            mlp_item_vec = layers.Flatten(name='mlp_item_flatten')(mlp_item_embedding)
            
            # Concatenate user and item embeddings
            mlp_vector = layers.Concatenate(name='mlp_concat')([mlp_user_vec, mlp_item_vec])
            
            # MLP layers
            for i, layer_size in enumerate(self.mlp_layers):
                mlp_vector = layers.Dense(
                    layer_size,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(self.l2_reg),
                    name=f'mlp_layer_{i}'
                )(mlp_vector)
                mlp_vector = layers.Dropout(self.dropout_rate, name=f'mlp_dropout_{i}')(mlp_vector)
            
            paths.append(mlp_vector)
        
        # Fusion layer
        if len(paths) > 1:
            # Concatenate GMF and MLP paths
            fusion_vector = layers.Concatenate(name='fusion_concat')(paths)
        else:
            fusion_vector = paths[0]
        
        # Output layer
        prediction = layers.Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='prediction'
        )(fusion_vector)
        
        # Build model
        model = Model(
            inputs=[user_input, item_input],
            outputs=prediction,
            name='NCF'
        )
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        logger.info(f"NCF model built: GMF={self.use_gmf}, MLP={self.use_mlp}")
        return model
    
    def prepare_data(
        self,
        interactions_df: pd.DataFrame,
        negative_sampling_ratio: int = 4
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Prepare training data with negative sampling
        
        Args:
            interactions_df: DataFrame with columns [user_id, item_id, rating]
            negative_sampling_ratio: Number of negative samples per positive sample
            
        Returns:
            Tuple of (train_data, val_data) dictionaries
        """
        logger.info("Preparing NCF training data...")
        
        # Encode user and item IDs
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['item_id'].unique()
        
        self.user_encoder = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_encoder = {iid: idx for idx, iid in enumerate(unique_items)}
        self.user_decoder = {idx: uid for uid, idx in self.user_encoder.items()}
        self.item_decoder = {idx: iid for iid, idx in self.item_encoder.items()}
        
        # Create positive samples
        interactions_df['user_idx'] = interactions_df['user_id'].map(self.user_encoder)
        interactions_df['item_idx'] = interactions_df['item_id'].map(self.item_encoder)
        
        # Binary labels (implicit feedback: 1 if interaction exists)
        interactions_df['label'] = 1
        
        # Generate negative samples
        negative_samples = self._generate_negative_samples(
            interactions_df,
            negative_sampling_ratio
        )
        
        # Combine positive and negative samples
        all_samples = pd.concat([interactions_df, negative_samples], ignore_index=True)
        all_samples = all_samples.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split train/validation (80/20)
        split_idx = int(0.8 * len(all_samples))
        train_df = all_samples[:split_idx]
        val_df = all_samples[split_idx:]
        
        # Prepare data for model
        train_data = {
            'user_id': train_df['user_idx'].values,
            'item_id': train_df['item_idx'].values,
            'labels': train_df['label'].values
        }
        
        val_data = {
            'user_id': val_df['user_idx'].values,
            'item_id': val_df['item_idx'].values,
            'labels': val_df['label'].values
        }
        
        logger.info(f"Data prepared: {len(train_data['labels'])} train, {len(val_data['labels'])} val samples")
        logger.info(f"Positive ratio - Train: {train_data['labels'].mean():.3f}, Val: {val_data['labels'].mean():.3f}")
        
        return train_data, val_data
    
    def _generate_negative_samples(
        self,
        interactions_df: pd.DataFrame,
        ratio: int
    ) -> pd.DataFrame:
        """Generate negative samples (items user hasn't interacted with)"""
        all_items = set(range(self.num_items))
        negative_samples = []
        
        # Group by user to find non-interacted items
        user_items = interactions_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        
        for user_idx, interacted_items in user_items.items():
            # Available items (not interacted)
            available_items = list(all_items - interacted_items)
            
            # Sample negative items
            num_negatives = min(len(interacted_items) * ratio, len(available_items))
            sampled_items = np.random.choice(available_items, size=num_negatives, replace=False)
            
            for item_idx in sampled_items:
                negative_samples.append({
                    'user_idx': user_idx,
                    'item_idx': item_idx,
                    'label': 0
                })
        
        return pd.DataFrame(negative_samples)
    
    def train(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        epochs: int = 50,
        batch_size: int = 256,
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Train the NCF model
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: Optional list of Keras callbacks
            
        Returns:
            Training history and metrics
        """
        if self.model is None:
            self.build_model()
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_auc',
                    patience=5,
                    mode='max',
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
        
        logger.info(f"Training NCF model for {epochs} epochs...")
        
        # Train model
        self.history = self.model.fit(
            x=[train_data['user_id'], train_data['item_id']],
            y=train_data['labels'],
            validation_data=(
                [val_data['user_id'], val_data['item_id']],
                val_data['labels']
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on validation set
        val_metrics = self.model.evaluate(
            x=[val_data['user_id'], val_data['item_id']],
            y=val_data['labels'],
            verbose=0,
            return_dict=True
        )
        
        logger.info(f"Training complete - Val AUC: {val_metrics['auc']:.4f}")
        
        return {
            'history': self.history.history,
            'final_val_metrics': val_metrics
        }
    
    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        batch_size: int = 512
    ) -> np.ndarray:
        """
        Predict interaction probability for user-item pairs
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            batch_size: Batch size for prediction
            
        Returns:
            Array of prediction scores
        """
        # Encode IDs
        user_indices = np.array([self.user_encoder.get(uid, 0) for uid in user_ids])
        item_indices = np.array([self.item_encoder.get(iid, 0) for iid in item_ids])
        
        # Predict
        predictions = self.model.predict(
            [user_indices, item_indices],
            batch_size=batch_size,
            verbose=0
        )
        
        return predictions.flatten()
    
    def get_recommendations(
        self,
        user_id: str,
        num_recommendations: int = 10,
        exclude_items: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top-N recommendations for a user
        
        Args:
            user_id: User ID
            num_recommendations: Number of recommendations to return
            exclude_items: Items to exclude from recommendations
            
        Returns:
            List of recommended items with scores
        """
        if user_id not in self.user_encoder:
            logger.warning(f"User {user_id} not in training data")
            return []
        
        user_idx = self.user_encoder[user_id]
        
        # Get all items
        all_item_indices = np.arange(self.num_items)
        
        # Exclude items
        if exclude_items:
            exclude_indices = [
                self.item_encoder[iid] 
                for iid in exclude_items 
                if iid in self.item_encoder
            ]
            all_item_indices = np.array([
                idx for idx in all_item_indices 
                if idx not in exclude_indices
            ])
        
        # Create user-item pairs
        user_indices = np.full(len(all_item_indices), user_idx)
        
        # Predict scores
        scores = self.model.predict(
            [user_indices, all_item_indices],
            batch_size=1024,
            verbose=0
        ).flatten()
        
        # Get top-N
        top_indices = np.argsort(scores)[-num_recommendations:][::-1]
        
        recommendations = []
        for idx in top_indices:
            item_idx = all_item_indices[idx]
            item_id = self.item_decoder[item_idx]
            score = scores[idx]
            
            recommendations.append({
                'item_id': item_id,
                'score': float(score),
                'rank': len(recommendations) + 1
            })
        
        return recommendations
    
    def get_user_embedding(self, user_id: str, embedding_type: str = 'mlp') -> Optional[np.ndarray]:
        """
        Get learned user embedding vector
        
        Args:
            user_id: User ID
            embedding_type: 'mlp' or 'gmf'
            
        Returns:
            User embedding vector
        """
        if user_id not in self.user_encoder:
            return None
        
        user_idx = self.user_encoder[user_id]
        layer_name = f'{embedding_type}_user_embedding'
        
        embedding_layer = self.model.get_layer(layer_name)
        embedding_vector = embedding_layer(np.array([user_idx]))[0].numpy()
        
        return embedding_vector
    
    def get_item_embedding(self, item_id: str, embedding_type: str = 'mlp') -> Optional[np.ndarray]:
        """
        Get learned item embedding vector
        
        Args:
            item_id: Item ID
            embedding_type: 'mlp' or 'gmf'
            
        Returns:
            Item embedding vector
        """
        if item_id not in self.item_encoder:
            return None
        
        item_idx = self.item_encoder[item_id]
        layer_name = f'{embedding_type}_item_embedding'
        
        embedding_layer = self.model.get_layer(layer_name)
        embedding_vector = embedding_layer(np.array([item_idx]))[0].numpy()
        
        return embedding_vector
    
    def save_model(self, save_dir: str):
        """Save model and encoders"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        self.model.save(save_path / 'ncf_model.keras')
        
        # Save encoders
        joblib.dump({
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'user_decoder': self.user_decoder,
            'item_decoder': self.item_decoder,
            'config': {
                'num_users': self.num_users,
                'num_items': self.num_items,
                'embedding_dim': self.embedding_dim,
                'mlp_layers': self.mlp_layers
            }
        }, save_path / 'ncf_metadata.pkl')
        
        logger.info(f"NCF model saved to {save_path}")
    
    def load_model(self, load_dir: str):
        """Load model and encoders"""
        load_path = Path(load_dir)
        
        # Load Keras model
        self.model = keras.models.load_model(load_path / 'ncf_model.keras')
        
        # Load encoders
        metadata = joblib.load(load_path / 'ncf_metadata.pkl')
        self.user_encoder = metadata['user_encoder']
        self.item_encoder = metadata['item_encoder']
        self.user_decoder = metadata['user_decoder']
        self.item_decoder = metadata['item_decoder']
        
        config = metadata['config']
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.embedding_dim = config['embedding_dim']
        self.mlp_layers = config['mlp_layers']
        
        logger.info(f"NCF model loaded from {load_path}")
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            logger.warning("Model not built yet. Call build_model() first.")


# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    logging.basicConfig(level=logging.INFO)
    
    # Create sample interactions
    interactions_data = {
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u3'] * 100,
        'item_id': ['i1', 'i2', 'i3', 'i4', 'i5'] * 100,
        'rating': [5, 4, 3, 5, 4] * 100
    }
    interactions_df = pd.DataFrame(interactions_data)
    
    # Initialize NCF
    ncf = NeuralCollaborativeFiltering(
        num_users=3,
        num_items=5,
        embedding_dim=32,
        mlp_layers=[64, 32, 16]
    )
    
    # Build model
    ncf.build_model()
    ncf.summary()
    
    # Prepare data
    train_data, val_data = ncf.prepare_data(interactions_df)
    
    # Train
    metrics = ncf.train(train_data, val_data, epochs=10, batch_size=32)
    print(f"Final validation AUC: {metrics['final_val_metrics']['auc']:.4f}")
    
    # Get recommendations
    recs = ncf.get_recommendations('u1', num_recommendations=3)
    print(f"\nRecommendations for u1: {recs}")
