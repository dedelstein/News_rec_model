import torch

config = {
    'word2vec_data':"data/document_vector.parquet",
    'article_data':"data/articles.parquet",
    'train_behaviors': "data/train/behaviors.parquet",
    'train_history': "data/train/history.parquet",
    'engagement_bins': "data/engagement_bins.pkl",
    'feature_bins': "data/combined_bin_boundaries.pkl",
    'article_embeddings': "data/article_embeddings.h5",
    'reduced_imgs': "data/reduced_img_embeddings.pkl",
    'reduced_text': "data/reduced_text_embeddings.pkl",

    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'lr': 3e-4,
    'epochs': 50,
    'batch_size': 256,

    'ckpt_save_path': "ckpt/",
    'test_ckpt_path': "ckpt/ckpt_ebnerd_demo_final.pth",
}
