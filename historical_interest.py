import torch
import torch.nn as nn
import numpy as np
import pickle

# e_i = target article
# e_j = historical data, dimensionally aligned with e_i
# Attention = MLP(𝒆 𝑗, 𝒆𝑖, 𝒆𝑖 − 𝒆 𝑗, 𝒆𝑖 ⊙ 𝒆 𝑗) = MLP(concat(historical, target, target - historical, target * historical (pointwise)))

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, activation_type='gelu'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim//4)
        self.fc2 = nn.Linear(input_dim//4, output_dim)
        self.activation_type = activation_type
        self.activation = self._get_activation()
    
    def _get_activation(self):
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        return activations.get(self.activation_type.lower(), nn.GELU())
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        return self.fc2(x)

class PointwiseAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = MLP(input_dim * 4, 1)
    
    def forward(self, target, history):
        # Create interaction features as described in paper
        diff = target - history
        pw_prod = target * history
        concat = torch.cat([history, target, diff, pw_prod], dim=-1)
        return self.mlp(concat)

class PointwiseAttentionExpanded(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = MLP(input_dim * 4, 1)
    
    def forward(self, target, history):
        """
        Compute pointwise attention scores for batched inputs
        
        Args:
            target: Shape [batch_size, 1, embed_dim] or [batch_size, num_targets, embed_dim]
            history: Shape [batch_size, num_history, embed_dim]
            
        Returns:
            attention_scores: Shape [batch_size, num_targets, num_history, 1]
        """
        # Add target sequence dimension if needed
        if len(target.shape) == 2:
            target = target.unsqueeze(1)  # [batch_size, 1, embed_dim]
            
        batch_size, num_targets, embed_dim = target.shape
        num_history = history.shape[1]
        
        # Reshape target to align with history for broadcasting,  [batch_size, num_targets, 1, embed_dim]
        target_expanded = target.unsqueeze(2)
        
        # Reshape history to align with target for broadcasting, [batch_size, 1, num_history, embed_dim]
        history_expanded = history.unsqueeze(1)
        
        # Create interaction features, each will be [batch_size, num_targets, num_history, embed_dim]
        diff = target_expanded - history_expanded
        pw_prod = target_expanded * history_expanded
        
        # Concatenate along feature dimension, [batch_size, num_targets, num_history, embed_dim * 4]
        concat = torch.cat([
            history_expanded.expand(-1, num_targets, -1, -1),
            target_expanded.expand(-1, -1, num_history, -1),
            diff, 
            pw_prod
        ], dim=-1)
        
        # Reshape for MLP, [batch_size * num_targets * num_history, embed_dim * 4]
        flat_concat = concat.view(-1, embed_dim * 4)
        
        # Apply MLP, [batch_size * num_targets * num_history, 1]
        scores = self.mlp(flat_concat)
        
        # Reshape back, [batch_size, num_targets, num_history, 1]
        attention_scores = scores.view(batch_size, num_targets, num_history, 1)
        
        return attention_scores

#e_u
class UserSideInterest(nn.Module):
    def __init__(self, user_history_embedder, article_embedder, embedding_dim):
        super().__init__()
        self.user_history_embedder = user_history_embedder
        self.article_embedder = article_embedder
        self.embedding_dim = embedding_dim
        self.attention = PointwiseAttention(embedding_dim * 4)
        
    def forward(self, article_ids, read_times, scroll_percentages, target_article_df):
        batch_size, seq_len = article_ids.shape
        
        # Get historical embeddings (e_j) - returns transformed embeddings via W1, then get e_i
        historical_combined = self.user_history_embedder(article_ids, read_times, scroll_percentages)
        target_embedding = self.article_embedder(target_article_df)
        
        # Expand target embedding for broadcasting
        target_expanded = target_embedding.unsqueeze(1)  # Shape: (batch_size, 1, 256)
        
        # Compute attention weights, e_i, e_j &  apply attention weights without softmax
        attention_weights = self.attention(target_expanded, historical_combined)
        user_interest = torch.sum(attention_weights * historical_combined, dim=1)
        
        return user_interest
    
    def get_embedding_dim(self):
        """Returns the output embedding dimension"""
        return self.embedding_dim

# e_u'
class MultiModalInterest(nn.Module):
    def __init__(self, img_embeddings_path, text_embeddings_path,  device):
        super().__init__()
        self.device = device
        
        # Load embeddings
        with open(img_embeddings_path, 'rb') as f:
            self.img_embeddings = pickle.load(f)
        with open(text_embeddings_path, 'rb') as f:
            self.text_embeddings = pickle.load(f)
            
        # Convert dictionary keys to integers
        self.img_embeddings = {int(k): v for k, v in self.img_embeddings.items()}
        self.text_embeddings = {int(k): v for k, v in self.text_embeddings.items()}
            
        # Get embedding dimensions
        img_dim = next(iter(self.img_embeddings.values())).shape[0]
        text_dim = next(iter(self.text_embeddings.values())).shape[0]
        self.total_dim = img_dim + text_dim
        
        # Create zero vectors for missing entries
        self.zero_img = np.zeros(img_dim)
        self.zero_text = np.zeros(text_dim)
        
        # Initialize attention mechanism
        self.attention = PointwiseAttention(self.total_dim).to(device)
    
    def _get_article_embedding(self, article_id: int) -> np.ndarray:
        """Get combined embedding for a single article ID"""
        try:
            article_id = int(article_id)
        except:
            article_id = 0
            
        img_emb = self.img_embeddings.get(article_id, self.zero_img)
        text_emb = self.text_embeddings.get(article_id, self.zero_text)
            
        return np.concatenate([img_emb, text_emb])
    
    def _get_combined_embeddings(self, article_ids) -> torch.Tensor:
        """Combine image and text embeddings for given article IDs"""
        # Handle the case where article_ids is a list containing a single array
        if isinstance(article_ids, list) and len(article_ids) == 1 and isinstance(article_ids[0], np.ndarray):
            article_ids = article_ids[0]
        
        # Convert to numpy array and flatten
        article_ids = np.asarray(article_ids).ravel()
        
        # Get embeddings
        combined_embeddings = []
        for aid in article_ids:
            emb = self._get_article_embedding(aid)
            combined_embeddings.append(emb)
                
        # Stack and convert to tensor
        stacked = np.stack(combined_embeddings)
        return torch.tensor(stacked, dtype=torch.float32).to(self.device)
    
    def forward(self, target_article_ids, history_article_ids) -> torch.Tensor:
        """Compute e'u (multimodal user interest) for given target and history articles"""
        # Get embeddings
        target_embeddings = self._get_combined_embeddings(target_article_ids)  # [num_targets, embed_dim]
        history_embeddings = self._get_combined_embeddings(history_article_ids)  # [num_history, embed_dim]
        
        # Add batch and sequence dimensions to target
        target_embeddings = target_embeddings.unsqueeze(1)  # [num_targets, 1, embed_dim]
        
        # Compute attention scores
        attention_scores = self.attention(target_embeddings, history_embeddings)  # [num_targets, num_history, 1]
        
        # Compute e'u
        e_u_prime = torch.sum(attention_scores * history_embeddings.unsqueeze(0), dim=1)
        return e_u_prime
    
class HistoricUserInterest(nn.Module):
    """
    Combines user side interests (e_u from historical behaviors + engagement) 
    with multimodal interests (e_u' from image and text embeddings)
    """
    def __init__(self, user_side_interest, multimodal_interest):
        super().__init__()
        self.user_side_interest = user_side_interest
        self.multimodal_interest = multimodal_interest
        
    def forward(self, article_ids, read_times, scroll_percentages, target_article_df):
        # Get e_u from UserSideInterest
        e_u = self.user_side_interest(
            article_ids=article_ids,
            read_times=read_times,
            scroll_percentages=scroll_percentages,
            target_article_df=target_article_df
        )
        
        # Get e_u' from MultiModalInterest
        # Extract article IDs from target_article_df
        target_article_ids = target_article_df.index.values
        history_article_ids = article_ids.cpu().numpy()
        
        e_u_prime = self.multimodal_interest(
            target_article_ids=target_article_ids,
            history_article_ids=history_article_ids
        )
        
        # Combine both representations
        combined = torch.cat([e_u, e_u_prime], dim=-1)
        
        # Apply optional projection
        return combined