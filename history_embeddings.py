import torch
import h5py
import pickle
import pandas as pd
import numpy as np
import torch.nn as nn
import pyarrow.parquet as pq

from config.run_config import config as run_config

class ArticleEmbedder:
    """
        This gets us e_i, the article history embeddings
    """
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.features = ['category', 'subcategory', 'article_type']
        self.vocab_dict = {}
        self.model = None
        self.sentiment_vocab = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    def fit_from_parquet(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        
        # Create vocabularies
        for feature in self.features:
            values = df[feature].values
            if isinstance(values[0], (list, np.ndarray)):
                values = np.array([v[0] if len(v) > 0 else 0 for v in values])
            
            unique_values = np.unique(values)
            self.vocab_dict[feature] = {
                val: idx for idx, val in enumerate(unique_values)
            }
        
        # Initialize model
        self.model = nn.ModuleDict({
            feature: nn.Embedding(
                num_embeddings=len(self.vocab_dict[feature]),
                embedding_dim=self.embedding_dim
            ) for feature in self.features
        })
        
        self.model.update({
            'sentiment_label': nn.Embedding(
                num_embeddings=len(self.sentiment_vocab),
                embedding_dim=self.embedding_dim // 2
            ),
            'sentiment_proj': nn.Linear(1, self.embedding_dim // 2)
        })
        
        # Initialize weights
        for layer in self.model.values():
            if isinstance(layer, nn.Embedding):
                nn.init.xavier_uniform_(layer.weight)
        
        return self
    
    def encode_features(self, df):
        encoded = {}
        for feature in self.features:
            values = df[feature].values
            if isinstance(values[0], (list, np.ndarray)):
                values = np.array([v[0] if len(v) > 0 else 0 for v in values])
            
            indices = np.array([self.vocab_dict[feature][val] for val in values])
            encoded[feature] = torch.from_numpy(indices)
        
        sentiment_indices = np.array([self.sentiment_vocab[val] for val in df['sentiment_label'].values])
        sentiment_scores = np.clip(df['sentiment_score'].values, -1, 1).astype(np.float32)
        
        encoded.update({
            'sentiment_label': torch.from_numpy(sentiment_indices),
            'sentiment_score': torch.from_numpy(sentiment_scores)
        })
        
        return encoded
    
    def get_embeddings(self, encoded_features):
        with torch.no_grad():
            embeddings = [
                self.model[feature](encoded_features[feature])
                for feature in self.features
            ]
            
            sentiment_emb = torch.cat([
                self.model['sentiment_label'](encoded_features['sentiment_label']),
                self.model['sentiment_proj'](encoded_features['sentiment_score'].unsqueeze(-1))
            ], dim=-1)
            
            embeddings.append(sentiment_emb)
            return torch.cat(embeddings, dim=-1)
    
    def __call__(self, df):
        return self.get_embeddings(self.encode_features(df))
    
    def embed_from_parquet(self, parquet_path):
        return self(pd.read_parquet(parquet_path))
    
class ArticleEmbeddingManager:
    def __init__(self, article_embedder, articles_path=run_config['articles_data']):
        """
        Initialize the embedding manager
        
        Args:
            article_embedder: Trained ArticleEmbedder instance
        """
        self.article_embedder = article_embedder
        # Calculate actual embedding dimension from a small sample
        pf = pq.ParquetFile(articles_path)
        sample_df = pf.read_row_group(0).to_pandas().head(1)
        with torch.no_grad():
            sample_embedding = article_embedder(sample_df)
        self.embedding_dim = sample_embedding.shape[1]
        self.embedding_file = None
        self.embeddings_cache = None
        print(f"Detected embedding dimension: {self.embedding_dim}")
        
    def precompute_embeddings(self, articles_path: str, save_path: str, batch_size: int = 10000):
        """
        Precompute embeddings for all articles and save to HDF5 format
        """
        # Read articles data
        articles_df = pd.read_parquet(articles_path)
        print(f"Computing {len(articles_df)} embeddings of dimension {self.embedding_dim}")
        
        # Create HDF5 file
        with h5py.File(save_path, 'w') as f:
            embeddings_dataset = f.create_dataset(
                'embeddings',
                shape=(len(articles_df), self.embedding_dim),
                dtype='float32'
            )
            
            article_ids = articles_df.index.values
            id_dataset = f.create_dataset(
                'article_ids',
                data=article_ids,
                dtype='int64'
            )
            
            for i in range(0, len(articles_df), batch_size):
                print(f"Processing batch {i//batch_size + 1}/{(len(articles_df) + batch_size - 1)//batch_size}")
                batch_df = articles_df.iloc[i:i+batch_size]
                with torch.no_grad():
                    batch_embeddings = self.article_embedder(batch_df).cpu().numpy()
                embeddings_dataset[i:i+len(batch_df)] = batch_embeddings
            
            print(f"Successfully saved embeddings to {save_path}")
    
    def load_embeddings(self, file_path: str):
        """Load pre-computed embeddings and cache them in memory"""
        with h5py.File(file_path, 'r') as f:
            self.embeddings_cache = f['embeddings'][:]
            article_ids = f['article_ids'][:]
            self.id_to_idx = {int(aid): idx for idx, aid in enumerate(article_ids)}
    
    def get_embeddings(self, article_ids):
        """Get embeddings for given article IDs"""
        if self.embeddings_cache is None:
            raise RuntimeError("Embeddings not loaded. Call load_embeddings first.")
        
        # Convert to numpy if tensor
        if torch.is_tensor(article_ids):
            article_ids = article_ids.cpu().numpy()
            
        # Flatten if multi-dimensional
        original_shape = article_ids.shape
        article_ids = article_ids.reshape(-1)
            
        # Convert IDs to indices
        indices = np.array([self.id_to_idx.get(int(aid), 0) for aid in article_ids])
        
        # Get embeddings directly from cache
        embeddings = self.embeddings_cache[indices]
        
        # Reshape back to original shape
        embeddings = embeddings.reshape(original_shape + (-1,))
        
        return torch.from_numpy(embeddings).float()
    
    def close(self):
        """Clear the embeddings cache"""
        self.embeddings_cache = None
            
    def __del__(self):
        """Destructor to ensure file is closed"""
        self.close()

class UserHistoryEmbedder(nn.Module):
    def __init__(self, embedding_manager, bins_path, embedding_dim=64):
        super().__init__()
        self.embedding_manager = embedding_manager
        self.embedding_dim = embedding_dim
        self.W1 = nn.Linear(embedding_dim * 6, embedding_dim * 4)

        # Load pre-calculated bins
        with open(bins_path, 'rb') as f:
            bins = pickle.load(f)
        self.read_time_bins = bins['read_time_bins']
        self.scroll_bins = bins['scroll_bins']
        self.num_bins = len(self.read_time_bins)
        
        # Full-sized engagement embeddings (embedding_dim each instead of embedding_dim//4)
        self.engagement_embedder = nn.ModuleDict({
            'read_time': nn.Embedding(num_embeddings=self.num_bins, embedding_dim=embedding_dim),
            'scroll': nn.Embedding(num_embeddings=self.num_bins, embedding_dim=embedding_dim)
        })
        
        # Initialize weights
        for layer in self.engagement_embedder.values():
            nn.init.xavier_uniform_(layer.weight)
    
    def get_bin_indices(self, values, bins):
        indices = np.digitize(values, bins) - 1
        return np.clip(indices, 0, self.num_bins - 1)
    
    def forward(self, article_ids, read_times, scroll_percentages):

        # Get pre-computed base embeddings
        base_embeddings = self.embedding_manager.get_embeddings(article_ids)  # Shape: (batch, seq_len, 256)

        # Get engagement bin indices
        read_time_indices = torch.from_numpy(
            self.get_bin_indices(read_times.cpu().numpy(), self.read_time_bins)
        ).to(read_times.device)
        
        scroll_indices = torch.from_numpy(
            self.get_bin_indices(scroll_percentages.cpu().numpy(), self.scroll_bins)
        ).to(scroll_percentages.device)
        
        # Get engagement embeddings - now each is embedding_dim
        read_time_emb = self.engagement_embedder['read_time'](read_time_indices)  # Shape: (batch, seq_len, 64)
        scroll_emb = self.engagement_embedder['scroll'](scroll_indices)  # Shape: (batch, seq_len, 64)
        
        # Concatenate all embeddings
        combined = torch.cat([base_embeddings, read_time_emb, scroll_emb], dim=-1)  # Shape: (batch, seq_len, 256+64+64)
        
        return self.W1(combined)