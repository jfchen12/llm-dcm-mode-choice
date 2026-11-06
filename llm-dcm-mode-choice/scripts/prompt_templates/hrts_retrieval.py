"""Prompt/retrieval functions preserved from the submitted-study source.

Only template construction and retrieval are included; no model execution.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler


LOS_FEATURES = [
    "tt_1", "tt_2", "tt_3", "tt_4",        # 各备选出行时间（分钟）：1=车 2=公交 3=自行车 4=步行
    "cost_1", "cost_2", "cost_3", "cost_4", # 各备选成本（欧元；自行车/步行恒为 0）
    "av_1", "av_2", "av_3",                 # 可得性（av_4=步行恒为 1，方差为 0，已省略）
    "HasDrivingLicense",                    # 有无驾照（约 14.5% 缺失，按训练集中位数填补）
    "NumberOfCars",                         # 家庭车辆数
    "TripDistanceKm",                       # 出行距离（km）—— 新增，对相似检索很关键
    "DepartureHour",                        # 出发时刻（小时，含小数）—— 新增
]

MODE_INT_TO_STR = {1: "drive", 2: "public transport", 3: "cycle", 4: "walk"}

MODES = [1, 2, 3, 4]

LABEL_COL = "choice_mode"

class _BaseRetriever:
    def __init__(self, df_train: pd.DataFrame):
        self.df_train = df_train.copy().reset_index(drop=True)

        missing = [c for c in LOS_FEATURES if c not in self.df_train.columns]
        if missing:
            raise ValueError(f"训练数据缺少特征列: {missing}")
        if LABEL_COL not in self.df_train.columns:
            raise ValueError(f"训练数据缺少 '{LABEL_COL}' 列")

        # 取特征 → 用训练集中位数填补缺失（社经列有 NaN，否则 scaler 报错）
        feat = self.df_train[LOS_FEATURES].astype(np.float32)
        self.fill_values = feat.median()
        feat = feat.fillna(self.fill_values)

        # 用 90% 训练数据拟合 StandardScaler（常量列如 cost_3/cost_4 会被安全置 0）
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(feat.values)

        # L2 归一化 → dot product = 余弦相似度
        norms       = np.linalg.norm(X_scaled, axis=1, keepdims=True)
        norms       = np.where(norms == 0, 1.0, norms)
        self.X_norm = (X_scaled / norms).astype(np.float32)   # (N, d)

        self.modes_arr = self.df_train[LABEL_COL].values

    # 编码单条 query → (1, d)
    def _encode_one(self, query_row: pd.Series) -> np.ndarray:
        q        = query_row.reindex(LOS_FEATURES).astype(np.float32).fillna(self.fill_values)
        q_raw    = q.values.reshape(1, -1)
        q_scaled = self.scaler.transform(q_raw)
        return q_scaled / max(np.linalg.norm(q_scaled), 1e-9)

    # 编码批量 query → (M, d)
    def _encode_many(self, query_df: pd.DataFrame) -> np.ndarray:
        Q        = query_df[LOS_FEATURES].astype(np.float32).fillna(self.fill_values)
        Q_scaled = self.scaler.transform(Q.values)
        norms    = np.linalg.norm(Q_scaled, axis=1, keepdims=True)
        norms    = np.where(norms == 0, 1.0, norms)
        return (Q_scaled / norms).astype(np.float32)

    def _mode_dist(self) -> dict:
        counts = self.df_train[LABEL_COL].value_counts().sort_index()
        return {MODE_INT_TO_STR[m]: int(counts.get(m, 0)) for m in MODES}

class FlatRetriever(_BaseRetriever):
    """
    对每个 query trip，从整个 90% 训练池中找最相似的 k 条，不区分 mode。
    返回的示例自然分布——更常见的 mode（HSL 中的小汽车）会被更多检索到。

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
