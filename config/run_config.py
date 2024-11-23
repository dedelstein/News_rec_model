import torch

config = {
    'word2vec_data':"data/document_vector.parquet",

    'train_dataset': "data/ebnerd",
    'train_data_processed': "data/ebnerd_demo.train",

    'validation_dataset': "data/ebnerd",
    'validation_data_processed': "./dataset/ebnerd_demo.validation",

    'test_dataset': "data/ebnerd_testset",
    'test_data_processed': "data/ebnerd_testset.test",

    'neg_label_max_num':10,

    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'lr': 3e-4,
    'epochs': 50,
    'batch_size': 256,

    'ckpt_save_path': "ckpt/",
    'test_ckpt_path': "ckpt/ckpt_ebnerd_demo_final.pth",
}
