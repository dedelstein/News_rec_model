import torch.nn as nn
import pickle
import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
import pandas as pd

from config.run_config import config as run_config
from config.model_config import config as model_config

def save_imgs_reduced(file_path, output_path='imgs_reduced.pkl', n_components=128):

    img_data_raw = pq.ParquetFile(file_path).read().to_pandas()

    img_data_ids = list(img_data_raw['article_id'])
    img_data_embeddings = list(img_data_raw['image_embedding'])
    pca = PCA(n_components=n_components)
    img_data_pca = pca.fit_transform(np.array(img_data_embeddings))

    img_embeddings_reduced = {}
    for idx, article_id in enumerate(img_data_ids):
        img_embeddings_reduced[article_id] = img_data_pca[idx]

    with open(output_path, 'wb') as f:
        pickle.dump(img_embeddings_reduced, f)
    
    return img_embeddings_reduced

def save_articles_reduced(file_path, output_path='word2vec_reduced.pkl', n_components=128):
    word2vec_data_raw = pq.ParquetFile(file_path).read().to_pandas()

    word2vec_data_ids = list(word2vec_data_raw['article_id'])
    word2vec_data_embeddings = list(word2vec_data_raw['document_vector'])
    pca = PCA(n_components=n_components)
    word2vec_data_pca = pca.fit_transform(np.array(word2vec_data_embeddings))

    word2vec_embeddings_reduced = {}
    for idx, article_id in enumerate(word2vec_data_ids):
        word2vec_embeddings_reduced[article_id] = word2vec_data_pca[idx]

    with open(output_path, 'wb') as f:
        pickle.dump(word2vec_embeddings_reduced, f)

    return word2vec_data_embeddings

def prepare_reduced_embeddings(img_path=run_config['img_data'], word2vec_path=run_config['word_2_vec'], n_components=model_config['pca_components']):
    """
    Prepare and save reduced embeddings from raw parquet files
    """
    print("Reducing and saving embeddings...")
       
    # Reduce and save image embeddings
    img_embeddings = save_imgs_reduced(
        file_path=img_path,
        output_path="imgs_reduced.pkl",
        n_components=n_components
    )
    print(f"Saved reduced image embeddings for {len(img_embeddings)} articles")
    
    # Reduce and save text embeddings
    text_embeddings = save_articles_reduced(
        file_path=word2vec_path,
        output_path="word2vec_reduced.pkl",
        n_components=n_components
    )
    print(f"Saved reduced text embeddings for {len(text_embeddings)} articles")
    return text_embeddings, img_embeddings

def compute_engagement_bins(history_path = run_config['history_data'], output_path = 'engagement_bins.pkl', n_bins: int = 50):
    """
    Compute bins for read times and scroll percentages from history data
    
    Args:
        history_path: Path to history.parquet file
        output_path: Path to save the computed bins
        n_bins: Number of bins to create (default 50 as mentioned in the paper)
    """
    # Read history data
    history_df = pd.read_parquet(history_path)
    
    # Extract read times and scroll percentages
    read_times = np.concatenate(history_df['read_time_fixed'].dropna().values)
    scroll_percentages = np.concatenate(history_df['scroll_percentage_fixed'].dropna().values)
    
    # Remove nulls/nans from scroll percentages
    scroll_percentages = scroll_percentages[~pd.isna(scroll_percentages)]
    
    # Get max values
    max_read_time = np.percentile(read_times, 99.9)  # Using 99.9th percentile to avoid extreme outliers
    max_scroll = 100  # Scroll percentage maximum is 100
    
    # Compute bins using quantiles for more robust binning
    # For read times: divide the range from 0 to max_read_time
    read_time_bins = np.quantile(read_times[read_times > 0], 
                                np.linspace(0, 1, n_bins+1)[1:-1])
    read_time_bins = np.clip(read_time_bins, 0, max_read_time)
    
    # For scroll percentages: divide the range from 0 to 100
    scroll_bins = np.quantile(scroll_percentages[scroll_percentages > 0], 
                            np.linspace(0, 1, n_bins+1)[1:-1])
    scroll_bins = np.clip(scroll_bins, 0, max_scroll)
    
    # Add min and max bins
    read_time_bins = np.concatenate([[0], read_time_bins, [max_read_time]])
    scroll_bins = np.concatenate([[0], scroll_bins, [max_scroll]])
    
    # Ensure bins are unique and sorted
    read_time_bins = np.unique(read_time_bins)
    scroll_bins = np.unique(scroll_bins)
    
    # Save bins
    bins_dict = {
        'read_time_bins': read_time_bins,
        'scroll_bins': scroll_bins
    }
    
    # Save bins to file
    with open(output_path, 'wb') as f:
        pickle.dump(bins_dict, f)
        
    return bins_dict

