import pickle
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import KBinsDiscretizer

class HierarchicalFeatureExtractor:
    def __init__(self, behaviors_df, articles_df):
        self.behaviors_df = behaviors_df
        self.articles_df = articles_df.set_index('article_id')

        # Precompute front-page features more comprehensively
        front_page_data = behaviors_df[behaviors_df['article_id'].isna()]
        self.front_page_features = np.array([
            front_page_data['read_time'].mean() if len(front_page_data) > 0 else 0,
            front_page_data['scroll_percentage'].mean() if len(front_page_data) > 0 else 0,
            len(front_page_data),  # Number of front page visits
            front_page_data['next_read_time'].mean() if len(front_page_data) > 0 else 0,  # Average next read time
            front_page_data['next_scroll_percentage'].mean() if len(front_page_data) > 0 else 0,  # Average next scroll
            1  # Front-page flag
        ])

    def safe_parse_list(self, list_str):
        """Safely parse string representation of a list of article IDs."""
        if isinstance(list_str, (list, np.ndarray)):
            return list_str
        if pd.isna(list_str):
            return []
        try:
            # Handle string representations of lists
            if isinstance(list_str, str):
                clean_str = list_str.replace('[', '').replace(']', '').replace('\x00', '').replace("'", "")
                return [int(float(x.strip())) for x in clean_str.split(',') if x.strip()]
            return []
        except:
            return []

    def extract_global_features(self, article_id, publication_time):
        """Extract global features with enhanced front-page handling."""
        if pd.isna(article_id):
            return self.front_page_features

        try:
            article = self.articles_df.loc[article_id]
            if pd.notna(publication_time):
                time_diff = (pd.Timestamp.now() - pd.Timestamp(publication_time)).total_seconds() / 3600
                exposure_change = (
                    article['total_pageviews'] / max(1, time_diff)
                    if time_diff <= 48
                    else 0
                )
            else:
                exposure_change = 0

            return np.array([
                float(article['total_inviews'] if pd.notna(article['total_inviews']) else 0),
                exposure_change,
                float(article['total_read_time'] if pd.notna(article['total_read_time']) else 0),
                int(article['premium'] if pd.notna(article['premium']) else False),
                float(article['sentiment_score'] if pd.notna(article['sentiment_score']) else 0),
                0  # Not front page
            ])
        except:
            return np.zeros(6)

    def extract_hourly_features(self, article_id, impression_time):
        """Extract hourly features with front-page specific metrics."""
        if pd.isna(article_id):
            # For front page, count total front page views in the last hour
            hour_window = pd.Timestamp(impression_time) - pd.Timedelta(hours=1)
            recent_front_page = self.behaviors_df[
                (self.behaviors_df['impression_time'] >= hour_window) & 
                (self.behaviors_df['article_id'].isna())
            ]
            
            return np.array([
                len(recent_front_page),  # Front page views in last hour
                recent_front_page['read_time'].mean() if len(recent_front_page) > 0 else 0,  # Avg read time
                recent_front_page['next_read_time'].mean() if len(recent_front_page) > 0 else 0  # Avg next read time
            ])

        # Regular article hourly features
        hour_window = pd.Timestamp(impression_time) - pd.Timedelta(hours=1)
        recent_behaviors = self.behaviors_df[self.behaviors_df['impression_time'] >= hour_window]

        def batch_count(column, target_id):
            """Optimized batch count with preprocessing."""
            parsed_lists = column.dropna().apply(self.safe_parse_list)
            flattened = [item for sublist in parsed_lists for item in sublist]
            return flattened.count(target_id)

        inview_count = batch_count(recent_behaviors['article_ids_inview'], article_id)
        click_count = batch_count(recent_behaviors['article_ids_clicked'], article_id)
        ctr = click_count / (inview_count + 1e-10)

        return np.array([inview_count, click_count, ctr])

    def extract_session_features(self, session_id):
        """Extract session features with front-page awareness."""
        session_data = self.behaviors_df[self.behaviors_df['session_id'] == session_id]

        if session_data.empty:
            return np.zeros(3)

        # Time differences between impressions
        impression_times = pd.to_datetime(session_data['impression_time'])
        time_diffs = impression_times.diff().dt.total_seconds().dropna()

        # Count front page visits in session
        front_page_visits = session_data['article_id'].isna().sum()

        return np.array([
            front_page_visits,
            time_diffs.mean() if not time_diffs.empty else 0,
            time_diffs.std() if len(time_diffs) > 1 else 0
        ])

    def extract_impression_features(self, impression_id):
        """Extract impression features with enhanced front-page metrics."""
        impression_data = self.behaviors_df[self.behaviors_df['impression_id'] == impression_id]

        if impression_data.empty:
            return np.zeros(4)

        inview_articles = self.safe_parse_list(impression_data['article_ids_inview'].iloc[0])
        clicked_articles = self.safe_parse_list(impression_data['article_ids_clicked'].iloc[0])

        # Additional metrics for front page
        next_read = impression_data['next_read_time'].iloc[0] if pd.notna(impression_data['next_read_time'].iloc[0]) else 0
        next_scroll = impression_data['next_scroll_percentage'].iloc[0] if pd.notna(impression_data['next_scroll_percentage'].iloc[0]) else 0

        return np.array([
            len(inview_articles),
            len(clicked_articles),
            next_read,
            next_scroll
        ])

    def extract_features(self, article_id, impression_id, session_id, impression_time, publication_time):
        """Combine all hierarchical feature groups with front-page handling."""
        # Extract all feature groups
        global_feats = self.extract_global_features(article_id, publication_time)
        hourly_feats = self.extract_hourly_features(article_id, impression_time)
        session_feats = self.extract_session_features(session_id)
        impression_feats = self.extract_impression_features(impression_id)
        
        # Concatenate all feature arrays
        return np.concatenate([
            global_feats,      # 6 features
            hourly_feats,      # 3 features
            session_feats,     # 3 features
            impression_feats   # 4 features
        ])                     # Total: 16 features
    
class PrecomputedListwiseFeatureMiner:
    def __init__(self, behaviors_df=None, articles_df=None, pickle_dir='precomputed_features/'):
        """
        Initialize with either dataframes or load from pickle files
        """
        self.pickle_dir = pickle_dir
        os.makedirs(pickle_dir, exist_ok=True)
        
        if behaviors_df is not None and articles_df is not None:
            print("Initializing with full dataset...")
            self._prepare_data(behaviors_df, articles_df)
            self._compute_and_save_all_metrics()
        else:
            print("Loading from existing pickle files...")
            self._load_all_metrics()

    def _prepare_data(self, behaviors_df, articles_df):
        """Prepare the full dataset"""
        self.behaviors_df = behaviors_df.copy()
        
        # Get unique article IDs from behaviors
        article_ids = set()
        for col in ['article_id', 'article_ids_inview', 'article_ids_clicked']:
            if col == 'article_id':
                article_ids.update(self.behaviors_df[col].dropna().astype(float).astype(int).unique())
            else:
                for id_list in self.behaviors_df[col].dropna():
                    article_ids.update(self.safe_parse_list(id_list))
        
        # Filter articles
        self.articles_df = articles_df[articles_df['article_id'].isin(article_ids)].copy()
        self.articles_df = self.articles_df.set_index('article_id')
        
        print(f"Prepared dataset with {len(self.behaviors_df)} behaviors and {len(self.articles_df)} articles")

    def safe_parse_list(self, list_str):
        """Safely parse string representation of a list of article IDs."""
        if isinstance(list_str, (list, np.ndarray)):
            return [int(x) for x in list_str if pd.notna(x)]
        if pd.isna(list_str):
            return []
        try:
            if isinstance(list_str, str):
                clean_str = list_str.replace('[', '').replace(']', '').replace('\x00', '').replace("'", "")
                return [int(float(x.strip())) for x in clean_str.split(',') if x.strip()]
            return []
        except:
            return []

    def _compute_and_save_all_metrics(self):
        """Compute and save all metrics to pickle files"""
        print("Computing article metrics...")
        self.article_metrics = self._precompute_article_metrics()
        with open(f'{self.pickle_dir}article_metrics.pkl', 'wb') as f:
            pickle.dump(self.article_metrics, f)
        
        print("Computing session metrics...")
        self.session_metrics = self._precompute_session_metrics()
        with open(f'{self.pickle_dir}session_metrics.pkl', 'wb') as f:
            pickle.dump(self.session_metrics, f)
        
        print("Computing impression metrics...")
        self.impression_metrics = self._precompute_impression_metrics()
        with open(f'{self.pickle_dir}impression_metrics.pkl', 'wb') as f:
            pickle.dump(self.impression_metrics, f)
        
        self.computation_time = datetime.now()
        with open(f'{self.pickle_dir}computation_time.pkl', 'wb') as f:
            pickle.dump(self.computation_time, f)

    def _precompute_article_metrics(self):
        """Precompute all article-level metrics."""
        metrics = pd.DataFrame(index=self.articles_df.index)
        
        # Basic popularity metrics
        metrics['total_inviews'] = self.articles_df['total_inviews'].fillna(0)
        metrics['total_pageviews'] = self.articles_df['total_pageviews'].fillna(0)
        metrics['total_read_time'] = self.articles_df['total_read_time'].fillna(0)
        
        # Days since publication
        current_time = pd.Timestamp.now()
        metrics['days_since_publication'] = (
            current_time - pd.to_datetime(self.articles_df['published_time'])
        ).dt.total_seconds() / (24 * 3600)
        
        print("Computing hourly inviews matrix...")
        hourly_inviews = []
        for hours in range(24):
            time_window = current_time - pd.Timedelta(hours=hours+1)
            recent_behaviors = self.behaviors_df[
                self.behaviors_df['impression_time'] >= time_window
            ]
            article_counts = self._batch_count_inviews(recent_behaviors, metrics.index)
            hourly_inviews.append(article_counts)
        
        metrics['hourly_inviews_matrix'] = [np.array(x) for x in zip(*hourly_inviews)]
        metrics['avg_hourly_inviews'] = metrics['hourly_inviews_matrix'].apply(np.mean)
        metrics['inview_stability'] = metrics['hourly_inviews_matrix'].apply(np.std)
        metrics['peak_inviews'] = metrics['hourly_inviews_matrix'].apply(np.max)
        metrics['recent_inviews'] = metrics['hourly_inviews_matrix'].apply(lambda x: x[0])
        
        return metrics

    def _batch_count_inviews(self, behaviors_subset, article_ids):
        """Efficiently count inviews for multiple articles."""
        counts = np.zeros(len(article_ids))
        
        for _, row in behaviors_subset.iterrows():
            inview_set = set(self.safe_parse_list(row['article_ids_inview']))
            for i, article_id in enumerate(article_ids):
                if article_id in inview_set:
                    counts[i] += 1
                    
        return counts

    def _get_article_group_metrics(self, article_ids):
        """Get precomputed metrics for a group of articles."""
        if len(article_ids) == 0:
            return {
                'mean_metrics': np.zeros(4),
                'std_metrics': np.zeros(4),
                'max_metrics': np.zeros(4),
                'quality_metrics': np.zeros(4)
            }
            
        metrics = []
        quality_metrics = []
        
        for article_id in article_ids:
            if article_id in self.article_metrics.index:
                article_data = self.article_metrics.loc[article_id]
                metrics.append([
                    article_data['total_inviews'],
                    article_data['total_pageviews'],
                    article_data['total_read_time'],
                    article_data['days_since_publication']
                ])
                quality_metrics.append([
                    article_data['avg_hourly_inviews'],
                    article_data['inview_stability'],
                    article_data['peak_inviews'],
                    article_data['recent_inviews']
                ])
            else:
                metrics.append([0, 0, 0, 0])
                quality_metrics.append([0, 0, 0, 0])
                
        metrics = np.array(metrics)
        quality_metrics = np.array(quality_metrics)
        
        return {
            'mean_metrics': metrics.mean(axis=0),
            'std_metrics': metrics.std(axis=0),
            'max_metrics': metrics.max(axis=0),
            'quality_metrics': quality_metrics.mean(axis=0)
        }

    def _precompute_impression_metrics(self):
        """Precompute impression-level metrics."""
        impression_metrics = {}
        total_impressions = len(self.behaviors_df)
        
        for idx, row in enumerate(self.behaviors_df.itertuples(), 1):
            if idx % 1000 == 0:  # Progress update every 1000 impressions
                print(f"Processing impression {idx}/{total_impressions}")
                
            inview_articles = self.safe_parse_list(getattr(row, 'article_ids_inview', []))
            clicked_articles = self.safe_parse_list(getattr(row, 'article_ids_clicked', []))
            
            inview_metrics = self._get_article_group_metrics(inview_articles)
            clicked_metrics = self._get_article_group_metrics(clicked_articles)
            
            impression_metrics[row.impression_id] = {
                'inview_metrics': inview_metrics,
                'clicked_metrics': clicked_metrics,
                'num_inview': len(inview_articles),
                'num_clicked': len(clicked_articles)
            }
        
        return impression_metrics

    def _precompute_session_metrics(self):
        """Precompute session-level metrics."""
        session_metrics = {}
        
        for session_id in self.behaviors_df['session_id'].unique():
            session_data = self.behaviors_df[self.behaviors_df['session_id'] == session_id]
            
            impression_times = pd.to_datetime(session_data['impression_time'])
            time_diffs = impression_times.diff().dt.total_seconds()[1:]
            
            session_metrics[session_id] = {
                'time_diff_mean': time_diffs.mean() if len(time_diffs) > 0 else 0,
                'time_diff_std': time_diffs.std() if len(time_diffs) > 0 else 0,
                'time_diff_max': time_diffs.max() if len(time_diffs) > 0 else 0,
                'session_length': len(session_data),
                'unique_articles': session_data['article_id'].nunique()
            }
        
        return session_metrics

    def _load_all_metrics(self):
        """Load all metrics from pickle files"""
        try:
            with open(f'{self.pickle_dir}article_metrics.pkl', 'rb') as f:
                self.article_metrics = pickle.load(f)
            with open(f'{self.pickle_dir}session_metrics.pkl', 'rb') as f:
                self.session_metrics = pickle.load(f)
            with open(f'{self.pickle_dir}impression_metrics.pkl', 'rb') as f:
                self.impression_metrics = pickle.load(f)
            with open(f'{self.pickle_dir}computation_time.pkl', 'rb') as f:
                self.computation_time = pickle.load(f)
            
            print(f"Successfully loaded metrics computed at: {self.computation_time}")
        except FileNotFoundError:
            raise FileNotFoundError("Pickle files not found. Initialize with dataframes first.")

    def get_features(self, impression_id, session_id):
        """Quick lookup of precomputed features."""
        impression_data = self.impression_metrics.get(impression_id, {})
        session_data = self.session_metrics.get(session_id, {})
        
        features = np.concatenate([
            impression_data.get('inview_metrics', {}).get('mean_metrics', np.zeros(4)),
            impression_data.get('inview_metrics', {}).get('std_metrics', np.zeros(4)),
            impression_data.get('inview_metrics', {}).get('max_metrics', np.zeros(4)),
            impression_data.get('inview_metrics', {}).get('quality_metrics', np.zeros(4)),
            impression_data.get('clicked_metrics', {}).get('mean_metrics', np.zeros(4)),
            impression_data.get('clicked_metrics', {}).get('std_metrics', np.zeros(4)),
            impression_data.get('clicked_metrics', {}).get('max_metrics', np.zeros(4)),
            impression_data.get('clicked_metrics', {}).get('quality_metrics', np.zeros(4)),
            [
                session_data.get('time_diff_mean', 0),
                session_data.get('time_diff_std', 0),
                session_data.get('time_diff_max', 0),
                session_data.get('session_length', 0),
                session_data.get('unique_articles', 0)
            ]
        ])
        
        return features

    def get_feature_names(self):
        """Return the names of all features in order."""
        basic_metrics = ['inviews', 'pageviews', 'read_time', 'days_since_pub']
        quality_metrics = ['avg_hourly_inviews', 'inview_stability', 'peak_inviews', 'recent_inviews']
        stats = ['mean', 'std', 'max']
        
        feature_names = []
        
        # Inview article features
        for stat in stats:
            feature_names.extend([f'inview_{stat}_{metric}' for metric in basic_metrics])
        feature_names.extend([f'inview_quality_{metric}' for metric in quality_metrics])
        
        # Clicked article features
        for stat in stats:
            feature_names.extend([f'clicked_{stat}_{metric}' for metric in basic_metrics])
        feature_names.extend([f'clicked_quality_{metric}' for metric in quality_metrics])
        
        # Session features
        feature_names.extend([
            'session_time_diff_mean',
            'session_time_diff_std',
            'session_time_diff_max',
            'session_length',
            'session_unique_articles'
        ])
        
        return feature_names

    def get_dataset_info(self):
        """Return information about the dataset"""
        return {
            'num_behaviors': len(self.behaviors_df),
            'num_articles': len(self.articles_df),
            'num_sessions': self.behaviors_df['session_id'].nunique(),
            'num_impressions': self.behaviors_df['impression_id'].nunique(),
            'date_range': (
                self.behaviors_df['impression_time'].min(),
                self.behaviors_df['impression_time'].max()
            )
        }

    @classmethod
    def load_from_pickle(cls, pickle_dir='precomputed_features/'):
        """Class method to load directly from pickle files"""
        return cls(pickle_dir=pickle_dir)

class InstantInterestFeatureMiner:
    """Extension class that utilizes PrecomputedListwiseFeatureMiner for instant interest modeling"""
    
    def __init__(self, precomputed_miner, n_bins=50):
        self.miner = precomputed_miner
        self.n_bins = n_bins
        self.discretizers = {}
        self._compute_hourly_features()
        self._fit_discretizers()
    
    def _compute_hourly_features(self):
        """Compute additional hourly features not already in PrecomputedListwiseFeatureMiner"""
        self.hourly_metrics = pd.DataFrame(index=self.miner.article_metrics.index)
        self.hourly_metrics['hourly_change'] = (
            self.miner.article_metrics['hourly_inviews_matrix']
            .apply(lambda x: (x[0] - x[1]) / max(1, x[1]) if len(x) > 1 else 0)
        )
    
    def _fit_discretizers(self):
        """Fit discretizers for all numerical features using equal-frequency binning"""
        # Collect all feature values for fitting
        feature_values = {name: [] for name in self.get_feature_names()}
        
        # Sample data to fit discretizers
        for impression_id in self.miner.impression_metrics.keys():
            session_id = self.miner.behaviors_df[
                self.miner.behaviors_df['impression_id'] == impression_id
            ]['session_id'].iloc[0]
            
            article_id = self.miner.behaviors_df[
                self.miner.behaviors_df['impression_id'] == impression_id
            ]['article_id'].iloc[0]
            
            if pd.notna(article_id):
                features = self.get_instant_interests(int(article_id), impression_id, session_id)
                for name, value in features.items():
                    feature_values[name].append(value)
        
        # Fit discretizer for each feature
        for feature_name in self.get_feature_names():
            values = np.array(feature_values[feature_name]).reshape(-1, 1)
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode='ordinal',  # We want integers for embedding lookup
                strategy='quantile'  # Equal-frequency binning
            )
            discretizer.fit(values)
            self.discretizers[feature_name] = discretizer
    
    def get_instant_interests(self, article_id, impression_id, session_id, discretize=False):
        """Get comprehensive instant interest features by combining existing metrics"""
        features = {}
        
        # 1. Global Non-Personalized Interests
        if article_id in self.miner.article_metrics.index:
            article_data = self.miner.article_metrics.loc[article_id]
            features.update({
                'total_inviews': article_data['total_inviews'],
                'avg_hourly_inviews': article_data['avg_hourly_inviews'],
                'inview_stability': article_data['inview_stability'],
                'recent_inviews': article_data['recent_inviews'],
                'hourly_change': self.hourly_metrics.loc[article_id, 'hourly_change']
            })
        else:
            features.update({
                'total_inviews': 0,
                'avg_hourly_inviews': 0,
                'inview_stability': 0,
                'recent_inviews': 0,
                'hourly_change': 0
            })
        
        # 2. Session Features
        session_metrics = self.miner.session_metrics.get(session_id, {})
        features.update({
            'session_length': session_metrics.get('session_length', 0),
            'session_unique_articles': session_metrics.get('unique_articles', 0),
            'session_time_diff_mean': session_metrics.get('time_diff_mean', 0),
            'session_time_diff_std': session_metrics.get('time_diff_std', 0)
        })
        
        # 3. Impression Features
        impression_data = self.miner.impression_metrics.get(impression_id, {})
        if impression_data:
            inview_metrics = impression_data.get('inview_metrics', {})
            features.update({
                'impression_count': impression_data.get('num_inview', 0),
                'impression_mean_inviews': np.mean(inview_metrics.get('mean_metrics', [0])),
                'impression_std_inviews': np.mean(inview_metrics.get('std_metrics', [0])),
                'impression_quality': np.mean(inview_metrics.get('quality_metrics', [0]))
            })
        else:
            features.update({
                'impression_count': 0,
                'impression_mean_inviews': 0,
                'impression_std_inviews': 0,
                'impression_quality': 0
            })
        
        if discretize:
            # Discretize each feature using fitted discretizers
            discretized_features = {}
            for name, value in features.items():
                value_array = np.array([value]).reshape(-1, 1)
                bin_id = int(self.discretizers[name].transform(value_array)[0])
                discretized_features[name] = bin_id
            return discretized_features
        
        return features

    def get_feature_names(self):
        """Return list of instant interest feature names"""
        return [
            # Global features
            'total_inviews', 'avg_hourly_inviews', 'inview_stability', 
            'recent_inviews', 'hourly_change',
            # Session features
            'session_length', 'session_unique_articles',
            'session_time_diff_mean', 'session_time_diff_std',
            # Impression features
            'impression_count', 'impression_mean_inviews',
            'impression_std_inviews', 'impression_quality'
        ]

    def get_num_bins(self):
        """Return number of bins for embedding layers"""
        return self.n_bins

def enhance_hierarchical_extractor(hierarchical_extractor, instant_miner):
    """Factory function to enhance HierarchicalFeatureExtractor with binned instant interests"""
    
    original_extract = hierarchical_extractor.extract_features
    
    def enhanced_extract(article_id, impression_id, session_id, impression_time, publication_time):
        # Get original hierarchical features
        hierarchical_features = original_extract(article_id, impression_id, session_id, 
                                              impression_time, publication_time)
        
        # Add discretized instant interest features
        instant_features = instant_miner.get_instant_interests(
            article_id, impression_id, session_id, discretize=True
        )
        
        # Convert to array of bin indices
        instant_feature_bins = np.array([
            instant_features[f] for f in instant_miner.get_feature_names()
        ])
        
        # Combine features
        combined = np.concatenate([hierarchical_features, instant_feature_bins])
        
        return combined
    
    hierarchical_extractor.extract_features = enhanced_extract
    return hierarchical_extractor

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

class CombinedFeatureExtractor(nn.Module):
    def __init__(self, behaviors_df, articles_df, n_bins=50, embedding_dim=64):
        super().__init__()
        self.behaviors_df = behaviors_df
        self.articles_df = articles_df.set_index('article_id')
        self.n_bins = n_bins
        self.embedding_dim = embedding_dim
        
        self.hierarchical = HierarchicalFeatureExtractor(behaviors_df, articles_df)
        self.listwise = ListwiseFeatureMiner(behaviors_df, articles_df)
        
        self.discretizers = {
            'global': self._create_discretizer(6),
            'hour': self._create_discretizer(3),
            'session': self._create_discretizer(3),
            'impression': self._create_discretizer(4)
        }
        
        self.embeddings = nn.ModuleDict({
            'global': nn.Embedding(n_bins, embedding_dim),
            'hour': nn.Embedding(n_bins, embedding_dim),
            'session': nn.Embedding(n_bins, embedding_dim),
            'impression': nn.Embedding(n_bins, embedding_dim)
        })
    
    def _create_discretizer(self, n_features):
        return {
            f'feature_{i}': KBinsDiscretizer(n_bins=self.n_bins, 
                                           encode='ordinal', 
                                           strategy='quantile')
            for i in range(n_features)
        }
    
    def fit_discretizers(self, features):
        start_idx = 0
        for group, group_discretizers in self.discretizers.items():
            n_features = len(group_discretizers)
            group_features = features[:, start_idx:start_idx + n_features]
            
            for i, discretizer in enumerate(group_discretizers.values()):
                feature_values = group_features[:, i].reshape(-1, 1)
                discretizer.fit(feature_values)
            
            start_idx += n_features
    
    def get_binned_features(self, features):
        binned_features = {}
        start_idx = 0
        
        for group, group_discretizers in self.discretizers.items():
            n_features = len(group_discretizers)
            group_features = features[:, start_idx:start_idx + n_features]
            
            binned = []
            for i, discretizer in enumerate(group_discretizers.values()):
                feature_values = group_features[:, i].reshape(-1, 1)
                binned_value = discretizer.transform(feature_values).ravel()
                binned.append(binned_value)
            
            binned_features[group] = np.column_stack(binned)
            start_idx += n_features
            
        return binned_features
    
    def forward(self, article_id, impression_id, session_id, impression_time, publication_time):
        hierarchical_features = self.hierarchical.extract_features(
            article_id, impression_id, session_id, impression_time, publication_time
        )
        
        binned_features = self.get_binned_features(hierarchical_features)
        
        embeddings = []
        for group in ['global', 'hour', 'session', 'impression']:
            binned = torch.tensor(binned_features[group], dtype=torch.long)
            group_embedding = torch.sum(self.embeddings[group](binned), dim=1)
            embeddings.append(group_embedding)
        
        e_L_u = torch.cat(embeddings, dim=-1)
        return e_L_u