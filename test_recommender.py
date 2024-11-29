import torch
import pandas as pd
import pyarrow.parquet as pq

from history_embeddings import ArticleEmbedder, ArticleEmbeddingManager, UserHistoryEmbedder
from historical_interest import UserSideInterest, HistoricUserInterest, MultiModalInterest
from feature_extractors import CombinedFeatureExtractor
from models import InterestFusionNetwork, NewsRecommender

from recommendations import get_batch_recommendations
from config.run_config import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

history_pq = config['train_history']
behaviors_pq = config['train_behaviors']
articles_pq = config['article_data']
engagement_bins = config['engagement_bins']
feature_bins = config['feature_bins']
article_pre_embeddings = config['article_embeddings']
reduced_img_embeddings = config['reduced_imgs']
reduced_text_embeddings = config['reduced_text']

behaviors_df = pd.read_parquet(behaviors_pq)
articles_df = pd.read_parquet(articles_pq)
history_df = pd.read_parquet(history_pq)

with open(feature_bins, 'rb') as f:
    combined_boundaries = pq.load(feature_bins)

article_embedder = ArticleEmbedder(embedding_dim=64)
article_embedder.fit_from_parquet(articles_pq)
article_embedder = article_embedder.to(device)

embedding_manager = ArticleEmbeddingManager(article_embedder)
embedding_manager.load_embeddings(article_pre_embeddings)
user_history_embedder = UserHistoryEmbedder(embedding_manager, engagement_bins).to(device)
user_side = UserSideInterest(user_history_embedder, article_embedder, embedding_dim=64).to(device)
multimodal = MultiModalInterest(reduced_img_embeddings, reduced_text_embeddings, device).to(device)

historic_model = HistoricUserInterest(user_side, multimodal).to(device)

feature_extractor = CombinedFeatureExtractor(
    behaviors_df=behaviors_df,
    articles_df=articles_df,
    bin_boundaries=combined_boundaries,
    embedding_dim=64
)

fusion_network = InterestFusionNetwork(
    embedding_dim=512,
    combined_feature_dim=512
)

feature_extractor = feature_extractor.to(device)
fusion_network = fusion_network.to(device)

recommender = NewsRecommender(historic_model, feature_extractor, fusion_network).to(device)
test_users = history_df['user_id'].unique()[:2]  # Test first 2 users
recommendations = get_batch_recommendations(test_users)