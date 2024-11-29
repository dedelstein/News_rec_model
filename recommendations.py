import torch
import pandas as pd
import numpy as np

def get_batch_recommendations(user_ids, history_df, behaviors_df, articles_df, recommender, device, batch_size=32):
    all_recommendations = {}
    
    for user_id in user_ids:
        user_history = history_df[history_df['user_id'] == user_id].iloc[0]
        user_behavior = behaviors_df[behaviors_df['user_id'] == user_id].iloc[-1]
        candidate_articles = [int(x) for x in user_behavior['article_ids_inview'] if pd.notna(x)]

        # Convert history data to numpy arrays
        history_articles = np.array(user_history['article_id_fixed'])
        history_reads = np.array(user_history['read_time_fixed'], dtype=np.float32)
        history_scrolls = np.array(user_history['scroll_percentage_fixed'], dtype=np.float32)

        batch_scores = []
        for i in range(0, len(candidate_articles), batch_size):
            batch_candidates = candidate_articles[i:i+batch_size]
            batch_size_actual = len(batch_candidates)

            # Create properly sized batch tensors
            histories = {
                'article_ids': torch.tensor([history_articles] * batch_size_actual, dtype=torch.long).to(device),
                'read_times': torch.tensor([history_reads] * batch_size_actual, dtype=torch.float32).to(device),
                'scroll_percentages': torch.tensor([history_scrolls] * batch_size_actual, dtype=torch.float32).to(device)
            }

            # Create batched behaviors and articles
            behaviors = [user_behavior] * batch_size_actual
            target_articles_df = pd.DataFrame([
                articles_df[articles_df['article_id'] == aid].iloc[0] 
                for aid in batch_candidates
            ])

            with torch.no_grad():
                scores = recommender(histories, behaviors, target_articles_df)
                batch_scores.extend(zip(batch_candidates, scores.cpu().numpy().flatten()))

        sorted_scores = sorted(batch_scores, key=lambda x: x[1], reverse=True)
        all_recommendations[user_id] = sorted_scores

    return all_recommendations