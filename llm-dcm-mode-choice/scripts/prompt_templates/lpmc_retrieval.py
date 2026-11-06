"""Prompt/retrieval functions preserved from the submitted-study source.

Only template construction and retrieval are included; no model execution.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler


LOS_FEATURES = [
    "distance",
    "dur_walking",
    "dur_cycling",
    "dur_pt_access",
    "dur_pt_rail",
    "dur_pt_bus",
    "dur_pt_int",
    "pt_interchanges",
    "dur_driving",
    "cost_transit",
    "cost_driving_fuel",
    "cost_driving_ccharge",
    "driving_traffic_percent",
    "driving_license",
    "car_ownership",
]

MODE_INT_TO_STR = {1: "walk", 2: "cycle", 3: "public transport", 4: "drive"}

MODES = [1, 2, 3, 4]

class _BaseRetriever:
    def __init__(self, df_train: pd.DataFrame):
        self.df_train = df_train.copy().reset_index(drop=True)

        missing = [c for c in LOS_FEATURES if c not in self.df_train.columns]
        if missing:
            raise ValueError(f"训练数据缺少 LOS 列: {missing}")
        if "travel_mode" not in self.df_train.columns:
            raise ValueError("训练数据缺少 'travel_mode' 列")

        # 用 90% 训练数据拟合 StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(
            self.df_train[LOS_FEATURES].values.astype(np.float32)
        )

        # L2 归一化 → dot product = 余弦相似度
        norms       = np.linalg.norm(X_scaled, axis=1, keepdims=True)
        norms       = np.where(norms == 0, 1.0, norms)
        self.X_norm = (X_scaled / norms).astype(np.float32)   # (N, d)

        self.modes_arr = self.df_train["travel_mode"].values

    # 编码单条 query → (1, d)
    def _encode_one(self, query_row: pd.Series) -> np.ndarray:
        q_raw    = np.array([[query_row[f] for f in LOS_FEATURES]], dtype=np.float32)
        q_scaled = self.scaler.transform(q_raw)
        return q_scaled / max(np.linalg.norm(q_scaled), 1e-9)

    # 编码批量 query → (M, d)
    def _encode_many(self, query_df: pd.DataFrame) -> np.ndarray:
        Q_scaled = self.scaler.transform(
            query_df[LOS_FEATURES].values.astype(np.float32)
        )
        norms = np.linalg.norm(Q_scaled, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (Q_scaled / norms).astype(np.float32)

    def _mode_dist(self) -> dict:
        counts = self.df_train["travel_mode"].value_counts().sort_index()
        return {MODE_INT_TO_STR[m]: int(counts.get(m, 0)) for m in MODES}

class FlatRetriever(_BaseRetriever):
    """
    对每个 query trip，从整个 90% 训练池中找最相似的 k 条，不区分 mode。
    返回的示例自然分布——更常见的 mode（如 LPMC 中的 drive）会被更多检索到。

    用法:
        retriever = FlatRetriever(df_train_90)
        examples  = retriever.get_examples(query_row, k=10)         # 单条
        all_ex    = retriever.get_examples_batch(df_test, k=10)     # 批量 → list of df
    """

    def __init__(self, df_train: pd.DataFrame):
        super().__init__(df_train)
        print(f"[flat_retriever] 训练池大小: {len(self.df_train):,} | mode 分布: {self._mode_dist()}")

    def get_examples(self, query_row: pd.Series, k: int = 10) -> pd.DataFrame:
        q_norm  = self._encode_one(query_row)
        sims    = (self.X_norm @ q_norm.T).flatten()
        top_idx = np.argsort(-sims)[:k]
        ex      = self.df_train.iloc[top_idx].copy()
        ex["similarity_score"] = sims[top_idx]
        return ex.reset_index(drop=True)

    def get_examples_batch(self, query_df: pd.DataFrame, k: int = 10) -> list:
        Q_norm     = self._encode_many(query_df)
        sim_matrix = Q_norm @ self.X_norm.T          # (M, N)

        results = []
        for i in range(len(query_df)):
            sims    = sim_matrix[i]
            top_idx = np.argsort(-sims)[:k]
            ex      = self.df_train.iloc[top_idx].copy()
            ex["similarity_score"] = sims[top_idx]
            results.append(ex.reset_index(drop=True))
        return results
