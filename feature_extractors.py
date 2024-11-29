import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class HierarchicalFeatureExtractor:
    def __init__(self, behaviors_df, articles_df):
        self.behaviors_df = behaviors_df
        self.articles_df = articles_df.set_index('article_id')

    @staticmethod
    def parse_list_fast(x):
        if isinstance(x, (list, np.ndarray)): return x
        if pd.isna(x): return []
        try:
            if isinstance(x, str):
                return [int(float(i)) for i in x.strip('[]').replace('\x00', '').replace("'", "").split(',') if i.strip()]
            return []
        except: return []

    def extract_hourly_features(self, article_id, impression_time):
        hour_back = (pd.Timestamp(impression_time) - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
        
        if pd.isna(article_id):
            mask = (
                (self.behaviors_df['impression_time'] >= hour_back) & 
                (self.behaviors_df['impression_time'] <= impression_time) & 
                self.behaviors_df['article_id'].isna()
            )
            recent_data = self.behaviors_df[mask]
            return np.array([
                len(recent_data),
                recent_data['read_time'].mean() if len(recent_data) > 0 else 0,
                recent_data['next_read_time'].mean() if len(recent_data) > 0 else 0
            ])
        
        mask = (
            (self.behaviors_df['impression_time'] >= hour_back) & 
            (self.behaviors_df['impression_time'] <= impression_time)
        )
        recent = self.behaviors_df[mask]
        
        if len(recent) == 0:
            return np.zeros(3)
            
        inview_lists = [self.parse_list_fast(x) for x in recent['article_ids_inview']]
        clicked_lists = [self.parse_list_fast(x) for x in recent['article_ids_clicked']]
        
        inview_count = sum(article_id in lst for lst in inview_lists)
        click_count = sum(article_id in lst for lst in clicked_lists)
        
        return np.array([inview_count, click_count, click_count / (inview_count + 1e-10)])

    def extract_global_features(self, article_id, publication_time):
        if pd.isna(article_id):
            front_mask = self.behaviors_df['article_id'].isna()
            front_data = self.behaviors_df[front_mask]
            if len(front_data) == 0:
                return np.zeros(6)
            return np.array([
                front_data['read_time'].mean(),
                front_data['scroll_percentage'].mean(),
                len(front_data),
                front_data['next_read_time'].mean(),
                front_data['next_scroll_percentage'].mean(),
                1
            ])

        try:
            article = self.articles_df.loc[article_id]
            if pd.notna(publication_time):
                time_diff = (pd.Timestamp.now() - pd.Timestamp(publication_time)).total_seconds() / 3600
                exposure_change = article['total_pageviews'] / max(1, time_diff) if time_diff <= 48 else 0
            else:
                exposure_change = 0

            return np.array([
                float(article.get('total_inviews', 0)),
                exposure_change,
                float(article.get('total_read_time', 0)),
                int(article.get('premium', False)),
                float(article.get('sentiment_score', 0)),
                0
            ])
        except:
            return np.zeros(6)

    def extract_session_features(self, session_id):
        mask = self.behaviors_df['session_id'] == session_id
        session_data = self.behaviors_df[mask]
        
        if len(session_data) == 0:
            return np.zeros(3)
            
        times = pd.to_datetime(session_data['impression_time'])
        diffs = np.diff(times.astype(np.int64)) / 1e9
        
        return np.array([
            session_data['article_id'].isna().sum(),
            np.mean(diffs) if len(diffs) > 0 else 0,
            np.std(diffs) if len(diffs) > 1 else 0
        ])

    def extract_impression_features(self, impression_id):
        mask = self.behaviors_df['impression_id'] == impression_id
        impression = self.behaviors_df[mask]
        
        if len(impression) == 0:
            return np.zeros(4)
            
        first = impression.iloc[0]
        inview = self.parse_list_fast(first['article_ids_inview'])
        clicked = self.parse_list_fast(first['article_ids_clicked'])
        
        return np.array([
            len(inview),
            len(clicked),
            first.get('next_read_time', 0),
            first.get('next_scroll_percentage', 0)
        ])

    def extract_features(self, article_id, impression_id, session_id, impression_time, publication_time):
        return np.concatenate([
            self.extract_global_features(article_id, publication_time),
            self.extract_hourly_features(article_id, impression_time),
            self.extract_session_features(session_id),
            self.extract_impression_features(impression_id)
        ])

class ListwiseFeatureExtractor:
    def __init__(self, behaviors_df, articles_df):
        self.behaviors_df = behaviors_df
        self.articles_df = articles_df.set_index('article_id')

    def safe_parse_list(self, list_str):
        """Safely parse string representation of a list of article IDs."""
        if isinstance(list_str, (list, np.ndarray)):
            return list_str.tolist() if isinstance(list_str, np.ndarray) else list_str
        if pd.isna(list_str):
            return []
        try:
            if isinstance(list_str, str):
                clean_str = list_str.replace('[', '').replace(']', '').replace('\x00', '').replace("'", "")
                return [int(float(x.strip())) for x in clean_str.split(',') if x.strip()]
            return []
        except:
            return []

    def compute_group_statistics(self, article_ids):
        """Compute statistical metrics for a group of articles."""
        # Convert to list and check length
        article_ids = self.safe_parse_list(article_ids)
        if len(article_ids) == 0:
            return np.zeros(12)  # Return zeros if no articles

        metrics = []
        for article_id in article_ids:
            if article_id in self.articles_df.index:
                article = self.articles_df.loc[article_id]
                metrics.append([
                    float(article['total_inviews'] if pd.notna(article['total_inviews']) else 0),
                    float(article['total_pageviews'] if pd.notna(article['total_pageviews']) else 0),
                    float(article['total_read_time'] if pd.notna(article['total_read_time']) else 0),
                    float((pd.Timestamp.now() - pd.to_datetime(article['published_time'])).total_seconds() / (24 * 3600))
                ])
            else:
                metrics.append([0, 0, 0, 0])

        metrics = np.array(metrics)
        if len(metrics) == 0:
            return np.zeros(12)

        return np.array([
            metrics.mean(axis=0),  # Mean of each metric (4 values)
            metrics.std(axis=0) if len(metrics) > 1 else np.zeros(4),   # Std of each metric (4 values)
            metrics.max(axis=0),   # Max of each metric (4 values)
        ]).flatten()

    def compute_time_based_features(self, impression_times):
        """Compute time-based features for a sequence of impressions."""
        if len(impression_times) <= 1:
            return np.zeros(3)
    
        times = pd.to_datetime(impression_times)
        time_diffs = np.array(times.diff()[1:].total_seconds())  # Convert to numpy array
    
        if len(time_diffs) == 0:
            return np.zeros(3)
            
        return np.array([
            np.mean(time_diffs),
            np.std(time_diffs) if len(time_diffs) > 1 else 0,
            np.max(time_diffs)
        ])

    def compute_quality_features(self, article_id):
        """Compute article quality features."""
        if pd.isna(article_id) or article_id not in self.articles_df.index:
            return np.zeros(4)

        article = self.articles_df.loc[article_id]
        
        quality_metrics = [
            float(article['total_inviews'] if pd.notna(article['total_inviews']) else 0),
            float(article['total_pageviews'] if pd.notna(article['total_pageviews']) else 0),
            float(article['total_read_time'] if pd.notna(article['total_read_time']) else 0),
            float(article['sentiment_score'] if pd.notna(article['sentiment_score']) else 0)
        ]

        return np.array(quality_metrics)

    def extract_listwise_features(self, impression_id, session_id):
        """Extract comprehensive listwise features for an impression within a session."""
        # Get session data 
        session_data = self.behaviors_df[self.behaviors_df['session_id'] == session_id]
        if session_data.empty:
            return np.zeros(31)

        # Get impression data
        impression_data = session_data[session_data['impression_id'] == impression_id]
        if impression_data.empty:
            return np.zeros(31)
        impression_data = impression_data.iloc[0]

        # Extract article IDs
        inview_articles = self.safe_parse_list(impression_data['article_ids_inview'])
        clicked_articles = self.safe_parse_list(impression_data['article_ids_clicked'])

        # Compute features for inview articles
        inview_stats = self.compute_group_statistics(inview_articles)

        # Compute features for clicked articles
        clicked_stats = self.compute_group_statistics(clicked_articles)

        # Compute session-level time features
        session_times = session_data['impression_time'].tolist()
        time_features = self.compute_time_based_features(session_times)

        # Compute quality features for the current article
        quality_features = self.compute_quality_features(impression_data['article_id'])

        # Combine all features
        return np.concatenate([
            inview_stats,      # 12 features
            clicked_stats,     # 12 features
            time_features,     # 3 features
            quality_features   # 4 features
        ])                     # Total: 31 features

class CombinedFeatureExtractor(nn.Module):
    def __init__(self, behaviors_df, articles_df, bin_boundaries, embedding_dim=64):
        super().__init__()
        self.behaviors_df = behaviors_df
        self.articles_df = articles_df.set_index('article_id')
        self.bin_boundaries = bin_boundaries
        self.embedding_dim = embedding_dim

        # Initialize both feature extractors
        self.hierarchical = HierarchicalFeatureExtractor(behaviors_df, articles_df)
        self.listwise = ListwiseFeatureExtractor(behaviors_df, articles_df)

        # Create embeddings for hierarchical features
        self.hierarchical_embeddings = nn.ModuleDict({
            'global': nn.Embedding(len(bin_boundaries['global'])-1, embedding_dim),
            'hour': nn.Embedding(len(bin_boundaries['hour'])-1, embedding_dim),
            'session': nn.Embedding(len(bin_boundaries['session'])-1, embedding_dim),
            'impression': nn.Embedding(len(bin_boundaries['impression'])-1, embedding_dim)
        })

        # Create embeddings for listwise features
        self.listwise_embeddings = nn.ModuleDict({
            'inview': nn.Embedding(len(bin_boundaries['listwise_inview'])-1, embedding_dim),
            'clicked': nn.Embedding(len(bin_boundaries['listwise_clicked'])-1, embedding_dim),
            'time': nn.Embedding(len(bin_boundaries['listwise_time'])-1, embedding_dim),
            'quality': nn.Embedding(len(bin_boundaries['listwise_quality'])-1, embedding_dim)
        })

        # Define feature splits for both types
        self.hierarchical_splits = {
            'global': slice(0, 6),
            'hour': slice(6, 9),
            'session': slice(9, 12),
            'impression': slice(12, 16)
        }

        self.listwise_splits = {
            'inview': slice(0, 12),
            'clicked': slice(12, 24),
            'time': slice(24, 27),
            'quality': slice(27, 31)
        }

    def fast_bin(self, features, splits, group):
        """Fast binning using np.digitize"""
        group_features = features[splits[group]].mean()
        boundary_key = f"listwise_{group}" if splits == self.listwise_splits else group
        boundaries = self.bin_boundaries[boundary_key]
        return np.digitize([group_features], boundaries)[0] - 1

    def debug_fast_bin(self, features, splits, group):
        group_features = features[splits[group]].mean()
        boundary_key = f"listwise_{group}" if splits == self.listwise_splits else group
        boundaries = self.bin_boundaries[boundary_key]
        num_bins = len(boundaries) - 1
        
        # Handle nan values
        if np.isnan(group_features):
            return 0
            
        bin_idx = np.digitize([group_features], boundaries)[0] - 1
        bin_idx = min(max(bin_idx, 0), num_bins - 1)
        
        return bin_idx

    def forward(self, article_id, impression_id, session_id, impression_time, publication_time):
        h_features = self.hierarchical.extract_features(
            article_id, impression_id, session_id, impression_time, publication_time
        )
        l_features = self.listwise.extract_listwise_features(impression_id, session_id)
    
        embeddings = []
    
        # Process hierarchical features
        for group in ['global', 'hour', 'session', 'impression']:
            bin_idx = torch.tensor(self.debug_fast_bin(h_features, self.hierarchical_splits, group), dtype=torch.long)
            bin_idx = bin_idx.to(next(self.hierarchical_embeddings[group].parameters()).device)
            emb = self.hierarchical_embeddings[group](bin_idx)
            embeddings.append(emb)
    
        # Process listwise features
        for group in ['inview', 'clicked', 'time', 'quality']:
            bin_idx = torch.tensor(self.debug_fast_bin(l_features, self.listwise_splits, group), dtype=torch.long)
            bin_idx = bin_idx.to(next(self.listwise_embeddings[group].parameters()).device)
            emb = self.listwise_embeddings[group](bin_idx)
            embeddings.append(emb)
    
        # Combine embeddings
        combined = torch.cat(embeddings, dim=-1)
        return combined

def extract_features_batch(behaviors_batch, articles_df, extractor, batch_size=32):
   """Extract features for a batch of behavior data efficiently."""
   pub_time_map = articles_df.set_index('article_id')['published_time'].to_dict()
   all_features = []
   
   for start_idx in range(0, len(behaviors_batch), batch_size):
       end_idx = min(start_idx + batch_size, len(behaviors_batch))
       mini_batch = behaviors_batch.iloc[start_idx:end_idx]
       
       batch_features = []
       for _, row in mini_batch.iterrows():
           article_id = int(row['article_id']) if pd.notna(row['article_id']) else None
           pub_time = pub_time_map.get(article_id)
           
           features = extractor(
               article_id=row['article_id'],
               impression_id=row['impression_id'],
               session_id=row['session_id'],
               impression_time=row['impression_time'], 
               publication_time=pub_time
           )
           batch_features.append(features)
           
       all_features.extend(batch_features)
   
   return torch.stack(all_features)

class BehaviorDataset(Dataset):
    def __init__(self, behaviors_df, articles_df, sample_size=None):
        self.behaviors = behaviors_df.sample(n=min(sample_size, len(behaviors_df)), random_state=42) if sample_size else behaviors_df
        self.articles_df = articles_df.set_index('article_id')
        self.extractor = HierarchicalFeatureExtractor(behaviors_df, articles_df)

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        pub_time = self.articles_df.loc[row['article_id']]['published_time'] if row['article_id'] in self.articles_df.index else pd.NaT
        features = self.extractor.extract_features(
            row['article_id'],
            row['impression_id'],
            row['session_id'],
            row['impression_time'],
            pub_time
        )
        return torch.tensor(features, dtype=torch.float32)