# Historical User Interest Model

This project implements a hierarchical user interest modeling system for news recommendation, focusing on combining both historical user behaviors and multimodal content representations.

## Core Components

### 1. Data Processing (`data_processing.py`)
- `save_imgs_reduced()`: Reduces image embeddings using PCA
- `save_articles_reduced()`: Reduces text embeddings using PCA
- `prepare_reduced_embeddings()`: Main function to prepare both image and text reduced embeddings
- `compute_engagement_bins()`: Creates bins for user engagement metrics (read times and scroll percentages)

### 2. History Embeddings (`history_embeddings.py`)

#### ArticleEmbedder
- Creates base article embeddings (e_i) using categorical features
- Handles article metadata including category, subcategory, article type, and sentiment
- Produces fixed-dimension embeddings for each article

#### ArticleEmbeddingManager
- Manages precomputation and caching of article embeddings
- Provides efficient lookup of embeddings by article ID
- Saves embeddings in HDF5 format for efficient storage and retrieval

#### UserHistoryEmbedder
- Combines article embeddings with user engagement data
- Processes historical user interactions including read times and scroll percentages
- Transforms engagement metrics into learned embeddings

### 3. Historical Interest Model (`historical_interest.py`)

#### Base Components
- `MLP`: Multi-layer perceptron for feature transformation
- `PointwiseAttention`: Implements attention mechanism without softmax normalization

#### Main Models
- `UserSideInterest`: Combines article embeddings (e_i) with user history (e_j) to create e_u
- `MultiModalInterest`: Processes reduced image and text embeddings to create e_u'
- `HistoricUserInterest`: Final model that combines both e_u and e_u' into a unified user interest representation

## Usage Flow

1. First, prepare the reduced embeddings:
```python
prepare_reduced_embeddings(img_path='path/to/images.parquet', 
                         word2vec_path='path/to/word2vec.parquet')
```

2. Compute engagement bins:
```python
compute_engagement_bins(history_path='path/to/history.parquet')
```

3. Initialize the models:
```python
# Initialize base components
article_embedder = ArticleEmbedder(embedding_dim=64)
article_embedder.fit_from_parquet('path/to/articles.parquet')

embedding_manager = ArticleEmbeddingManager(article_embedder)
user_history_embedder = UserHistoryEmbedder(embedding_manager, 'path/to/bins.pkl')

# Initialize interest models
user_side = UserSideInterest(user_history_embedder, article_embedder, embedding_dim=64)
multimodal = MultiModalInterest('path/to/imgs_reduced.pkl', 
                              'path/to/word2vec_reduced.pkl', 
                              device)

# Create final combined model
historic_model = HistoricUserInterest(user_side, multimodal)
```

## Architecture Details

The system follows a hierarchical architecture for modeling user interests:

1. Base Level: Article representations using categorical features and sentiment
2. Engagement Level: User interaction patterns through read times and scroll behavior
3. Historical Level: Attention-based combination of user history and target articles
4. Multimodal Level: Integration of reduced image and text representations
5. Final Level: Unified representation combining behavioral and content-based interests

The model aims to capture both long-term user preferences through historical behaviors and short-term interests through multimodal content understanding.

