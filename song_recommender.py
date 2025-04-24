"""Interactive song recommender (Rainbow DQN + content similarity).

Çalıştırmak için:
    python song_recommender.py

Ana iyileştirmeler
------------------
* NumPy 2.0 uyumlu min‑max normalizasyon.
* Cosine benzerliği **z‑score standardize** edilmiş öznitelikler üzerinden
  hesaplanır → enerji/valence dengesizliğine çözüm.
* Benzerlik eşiği 0.70, blend katsayısı `alpha = 0.85`.
* Oturum‑içi çeşitlilik filtresi; aynı şarkıyı tekrar göstermez.
"""
from __future__ import annotations

import os
import random
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

# ---- Project‑specific imports ------------------------------------------------#
from model.rainbow_agent import RainbowAgent, DEFAULT_CFG
from environment.music_env import MusicRecommendationEnv
from data.preprocess import preprocess_data
from data.data_loader import load_data
from config import Config
# -----------------------------------------------------------------------------#

# -----------------------------------------------------------------------------#
# Utility helpers
# -----------------------------------------------------------------------------#

def seed_everything(seed: int) -> None:
    """Set RNG seeds for full reproducibility (as far as possible)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min‑max normalise *arr* to [0, 1] (NumPy ≥2.0 safe)."""
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    span = arr_max - arr_min
    return (arr - arr_min) / (span + 1e-8) if span else np.zeros_like(arr)

# -----------------------------------------------------------------------------#
# Recommendation core
# -----------------------------------------------------------------------------#

def recommend_for(
    query: str,
    agent: "RainbowAgent",
    songs_df: pd.DataFrame,
    feature_cols: List[str],
    *,
    num_history: int = 5,
    top_k: int = 10,
    seen_ids: Optional[Set[str]] = None,
    sim_threshold: float = 0.70,   # daha katı
    alpha: float = 0.85,           # similarity baskın
) -> Optional[Tuple[pd.DataFrame, Set[str]]]:
    """Print and return **top_k** recommendations for *query*.

    Skor formülü
    -------------
        score = alpha · similarity_std + (1‑alpha) · q_norm

    Returns *(rows, new_ids)* veya **None**.
    """

    # 1) Query eşleşmeleri (case‑insensitive substring)
    matches = songs_df[songs_df["name"].str.contains(query, case=False, regex=False)]
    if matches.empty:
        print(f"❌ Song '{query}' not found in dataset.")
        return None

    if len(matches) > 1:
        print("Multiple matches found (showing first 5):")
        print(matches[["name", "artists"]].drop_duplicates().head(5))

    seed_song = (
        matches.sort_values("popularity", ascending=False).iloc[0]
        if "popularity" in matches.columns
        else matches.sample(n=1, random_state=None).iloc[0]
    )

    # 2) State vektörü: rastgele geçmiş + seed özellikleri (agent beklentisine dokunma)
    history_feats = (
        songs_df[songs_df["id"] != seed_song["id"]]
        .sample(n=num_history, random_state=None)[feature_cols]
        .to_numpy()
        .flatten()
    )
    state_vec = np.concatenate([history_feats, seed_song[feature_cols].to_numpy()]).astype(np.float32)

    # 3) Q‑değerleri → [0,1]
    with torch.no_grad():
        q_vals = agent.online_net(torch.from_numpy(state_vec).unsqueeze(0).to(agent.device))
    q_norm = _normalize(q_vals.squeeze().cpu().numpy())

    # 4) Standartlaştırılmış cosine benzerliği
    feats = songs_df[feature_cols].to_numpy(dtype=np.float32)
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-8
    feats_std = (feats - mean) / std
    seed_std = (seed_song[feature_cols].to_numpy(dtype=np.float32, copy=True) - mean.squeeze()) / std.squeeze()
    sim = cosine_similarity(feats_std, seed_std.reshape(1, -1)).ravel()

    # 5) Aday maskesi + çeşitlilik
    mask = (sim >= sim_threshold) & (songs_df["id"] != seed_song["id"])
    if seen_ids:
        mask &= ~songs_df["id"].isin(seen_ids)
    if not mask.any():
        print("⚠️  No candidates passed similarity/diversity filter.")
        return None

    # 6) Blend ve top‑k
    scores = alpha * sim + (1 - alpha) * q_norm
    cand_idx = np.where(mask)[0]
    best_local = np.argpartition(-scores[cand_idx], min(top_k, cand_idx.size))[:top_k]
    top_idx = cand_idx[best_local[np.argsort(-scores[cand_idx][best_local])]]

    # 7) Görüntüle
    print("\n🎵 Top Recommendations:")
    for idx in top_idx:
        row = songs_df.iloc[idx]
        print(f"- {row['name']} by {row['artists']} (score={scores[idx]:.4f}, sim={sim[idx]:.4f})")

    new_ids = set(songs_df.iloc[top_idx]["id"].tolist())
    return songs_df.iloc[top_idx], new_ids

# -----------------------------------------------------------------------------#
# CLI entry point
# -----------------------------------------------------------------------------#

def main() -> None:
    seed_everything(Config.RANDOM_SEED)

    print("Loading & preprocessing data …")
    df_raw = load_data(Config.DATA_PATH, sample_size=Config.SAMPLE_SIZE, random_state=Config.RANDOM_SEED)
    df_proc = preprocess_data(df_raw)

    env = MusicRecommendationEnv(
        df_proc,
        Config.FEATURE_COLS,
        num_songs_per_state=Config.NUM_SONGS_PER_STATE,
        max_steps=Config.MAX_STEPS,
    )

    cfg = {**DEFAULT_CFG, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    agent = RainbowAgent(state_dim=env.state_dim, action_dim=env.action_space, cfg=cfg)
    agent.load("./saved_models/dqn_model_episode_800.pth")

    print("\n✅ Model loaded. Type a song name or 'exit' to quit.")
    seen_ids: Set[str] = set()

    try:
        while True:
            query = input("\nEnter a song name (or 'exit'): ").strip()
            if query.lower() == "exit":
                break
            res = recommend_for(query, agent, df_proc, Config.FEATURE_COLS, seen_ids=seen_ids)
            if res is not None:
                _, new_ids = res
                seen_ids |= new_ids
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Bye!")


if __name__ == "__main__":
    main()
