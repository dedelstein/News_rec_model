import torch
import torch.nn as nn

class InterestFusionNetwork(nn.Module):
    def __init__(self, embedding_dim, combined_feature_dim):
        super().__init__()
        total_dim = embedding_dim + combined_feature_dim
        
        self.batch_norm = nn.BatchNorm1d(total_dim)
        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Linear(total_dim // 2, total_dim),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        
    def forward(self, e_H_u, e_L_u):
        concat = torch.cat([e_H_u, e_L_u], dim=-1)
        context = self.batch_norm(concat)
        gate_weights = self.gate(context)
        gated = gate_weights * context
        return self.mlp(gated)

class NewsRecommender(nn.Module):
    def __init__(self, historic_model, feature_extractor, fusion_network):
        super().__init__()
        self.historic_model = historic_model
        self.feature_extractor = feature_extractor
        self.fusion_network = fusion_network

    def forward(self, histories, behaviors, target_articles):
        article_ids = histories['article_ids'].long()
        read_times = histories['read_times'].float()
        scroll_percentages = histories['scroll_percentages'].float()

        e_H = self.historic_model(
            article_ids=article_ids,
            read_times=read_times,
            scroll_percentages=scroll_percentages,
            target_article_df=target_articles
        )

        batch_e_L_e_c = []
        for i, behavior in enumerate(behaviors):
            target_article = target_articles.iloc[[i]]
            e_L_e_c = self.feature_extractor(
                article_id=target_article.index[0],
                impression_id=behavior['impression_id'],
                session_id=behavior['session_id'],
                impression_time=behavior['impression_time'],
                publication_time=target_article['published_time'].iloc[0]
            )
            batch_e_L_e_c.append(e_L_e_c)

        e_L_e_c = torch.stack(batch_e_L_e_c)

        return self.fusion_network(e_H, e_L_e_c)