import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

from .abstract import AbstractFeatureGenerator

from autogluon.common.features.types import (
    R_BOOL,
    R_CATEGORY,
    R_FLOAT,
    R_INT,
    R_OBJECT,
    S_BOOL,
    S_DATETIME_AS_OBJECT,
    S_IMAGE_BYTEARRAY,
    S_IMAGE_PATH,
)

class TargetAwareFeatureCompressionFeatureGenerator(AbstractFeatureGenerator):
    """
    Target-Aware Feature Compression (TAFC)

    Outputs (always):
      - TAFC_score
      - TAFC_bin
      - TAFC_support

    Multiclass only (additional):
      - TAFC_score_class_<class_label>
    """

    def __init__(
        self,
        n_bins=1000,
        n_splits=5,
        smoothing=0,
        random_state=42,
        stratify=True,
        target_type="multiclass",  # "binary", "multiclass", "regression"
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_bins = int(n_bins)
        self.n_splits = int(n_splits)
        self.smoothing = float(smoothing)
        self.random_state = int(random_state)
        self.stratify = bool(stratify)
        self.target_type = target_type

        self.global_mean_ = None
        self.bin_edges_ = None
        self.group_stats_ = None

        # multiclass-specific
        self.label_encoder_ = None
        self.classes_ = None
        self.group_stats_per_class_ = None
        self.global_class_means_ = None

    @staticmethod
    def _make_key(X: pd.DataFrame) -> pd.Series:
        return X.astype(str).agg("|".join, axis=1)

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X_index = X.index
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        # ----- target handling -----
        if self.target_type == "multiclass":
            self.label_encoder_ = LabelEncoder()
            y_enc = self.label_encoder_.fit_transform(y)
            self.classes_ = self.label_encoder_.classes_
            y_used = pd.Series(y_enc)
        else:
            y_used = y.astype(float)

        key = self._make_key(X)
        self.global_mean_ = float(y_used.mean())

        n = len(X)
        oof_score = np.empty(n, dtype=float)
        oof_support = np.empty(n, dtype=float)

        if self.target_type == "multiclass":
            K = len(self.classes_)
            oof_class_scores = np.zeros((n, K), dtype=float)
            self.global_class_means_ = np.array(
                [(y_used == k).mean() for k in range(K)]
            )

        # ----- CV strategy -----
        if self.stratify and self.target_type != "regression":
            cv = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            splits = cv.split(X, y)
        else:
            cv = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            splits = cv.split(X)

        # ----- OOF encoding -----
        for tr_idx, va_idx in splits:
            tr_key = key.iloc[tr_idx]
            tr_y = y_used.iloc[tr_idx]

            stats = tr_y.groupby(tr_key).agg(["mean", "count"])
            m = self.smoothing
            stats["smoothed"] = (
                stats["count"] * stats["mean"] + m * self.global_mean_
            ) / (stats["count"] + m)

            va_key = key.iloc[va_idx]
            score = va_key.map(stats["smoothed"])
            support = va_key.map(stats["count"])

            oof_score[va_idx] = score.fillna(self.global_mean_).to_numpy()
            oof_support[va_idx] = support.fillna(0).to_numpy()

            # ----- multiclass: class-conditional -----
            if self.target_type == "multiclass":
                for k in range(K):
                    tr_yk = (tr_y == k).astype(float)
                    stats_k = tr_yk.groupby(tr_key).agg(["mean", "count"])
                    stats_k["smoothed"] = (
                        stats_k["count"] * stats_k["mean"]
                        + m * self.global_class_means_[k]
                    ) / (stats_k["count"] + m)

                    oof_class_scores[va_idx, k] = (
                        va_key.map(stats_k["smoothed"])
                        .fillna(self.global_class_means_[k])
                        .to_numpy()
                    )

        # ----- binning -----
        raw_edges = np.quantile(
            oof_score, q=np.linspace(0, 1, self.n_bins + 1)
        )
        self.bin_edges_ = np.unique(raw_edges)

        # ----- full-data stats -----
        full_stats = y_used.groupby(key).agg(["mean", "count"])
        m = self.smoothing
        full_stats["smoothed"] = (
            full_stats["count"] * full_stats["mean"] + m * self.global_mean_
        ) / (full_stats["count"] + m)
        self.group_stats_ = full_stats[["smoothed", "count"]]

        if self.target_type == "multiclass":
            self.group_stats_per_class_ = {}
            for k in range(K):
                yk = (y_used == k).astype(float)
                fs = yk.groupby(key).agg(["mean", "count"])
                fs["smoothed"] = (
                    fs["count"] * fs["mean"]
                    + m * self.global_class_means_[k]
                ) / (fs["count"] + m)
                self.group_stats_per_class_[k] = fs["smoothed"]

        bins = np.digitize(oof_score, self.bin_edges_[1:-1], right=True)

        # ----- output -----
        out = X.copy()
        out["TAFC_score"] = oof_score
        out["TAFC_bin"] = bins.astype(int)
        out["TAFC_support"] = oof_support.astype(int)

        if self.target_type == "multiclass":
            for k, cls in enumerate(self.classes_):
                out[f"TAFC_score_class_{cls}"] = oof_class_scores[:, k]
        out.index = X_index
        return out, dict()

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_index = X.index   
        if self.group_stats_ is None:
            raise RuntimeError("fit_transform must be called first")

        X = X.reset_index(drop=True)
        key = self._make_key(X)

        score = key.map(self.group_stats_["smoothed"])
        support = key.map(self.group_stats_["count"])

        score = score.fillna(self.global_mean_).to_numpy()
        support = support.fillna(0).to_numpy()
        bins = np.digitize(score, self.bin_edges_[1:-1], right=True)

        out = X.copy()
        out["TAFC_score"] = score
        out["TAFC_bin"] = bins.astype(int)
        out["TAFC_support"] = support.astype(int)

        if self.target_type == "multiclass":
            for k, cls in enumerate(self.classes_):
                s = key.map(self.group_stats_per_class_[k])
                out[f"TAFC_score_class_{cls}"] = (
                    s.fillna(self.global_class_means_[k]).to_numpy()
                )
        out.index = X_index
        return out


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(
            valid_raw_types=[R_OBJECT, R_CATEGORY, R_BOOL, R_INT, R_FLOAT],
            invalid_special_types=[S_DATETIME_AS_OBJECT, S_IMAGE_PATH, S_IMAGE_BYTEARRAY],
            # required_raw_special_pairs=[
            #     (R_BOOL, None),
            #     (R_OBJECT, None),
            #     (R_CATEGORY, None),
            #     # (R_INT, S_BOOL),
            #     # (R_FLOAT, S_BOOL),
            # ],
        )


class RandomSubsetTAFC:
    """
    TargetAwareFeatureCompression on:
      - the full feature set (always included)
      - additional random subsets of features

    Output columns:
      RSTAF_0_*  -> full-feature TAFC
      RSTAF_1_*  -> first random subset
      ...
    """

    def __init__(
        self,
        tafc_params=None,
        n_subsets=10,          # total TAFCs INCLUDING full set
        subset_size=None,
        min_subset_size=2,
        max_subset_size=None,
        random_state=42,
    ):
        self.tafc_params = tafc_params or {}
        self.n_subsets = int(n_subsets)
        self.subset_size = subset_size
        self.min_subset_size = int(min_subset_size)
        self.max_subset_size = max_subset_size
        self.random_state = int(random_state)

        self.subsets_ = None
        self.tafcs_ = None
        self.class_labels_ = None   # <-- NEW

    def _sample_subset(self, features, rng):
        if self.subset_size is not None:
            k = self.subset_size
        else:
            k_max = self.max_subset_size or len(features) - 1
            k = rng.integers(self.min_subset_size, k_max + 1)

        return tuple(sorted(rng.choice(features, size=k, replace=False)))

    def _extract_class_labels(self, df):
        """
        Detect class labels from TAFC output columns:
          TAFC_score_class_<label>
        """
        labels = []
        for c in df.columns:
            if c.startswith("TAFC_score_class_"):
                labels.append(c.replace("TAFC_score_class_", ""))
        return labels

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X_index = X.index
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        features = np.array(X.columns)
        rng = np.random.default_rng(self.random_state)

        self.subsets_ = []
        self.tafcs_ = []

        out = X.copy()

        # ---- 0) Full-feature TAFC (always) ----
        full_subset = tuple(features.tolist())
        self.subsets_.append(full_subset)

        tafc_full = TargetAwareFeatureCompression(**self.tafc_params)
        sdf = tafc_full.fit_transform(X, y).reset_index(drop=True)

        # core outputs
        out["RSTAF_0_score"] = sdf["TAFC_score"].to_numpy()
        out["RSTAF_0_bin"] = sdf["TAFC_bin"].to_numpy()
        out["RSTAF_0_support"] = sdf["TAFC_support"].to_numpy()

        # --- NEW: detect multiclass channels once ---
        if self.tafc_params.get("target_type") == "multiclass":
            self.class_labels_ = self._extract_class_labels(sdf)
        else:
            self.class_labels_ = []


        # --- NEW: propagate class-conditional TAFC ---
        for lbl in self.class_labels_:
            out[f"RSTAF_0_score_class_{lbl}"] = sdf[
                f"TAFC_score_class_{lbl}"
            ].to_numpy()

        self.tafcs_.append(tafc_full)

        # ---- 1..n) Random subsets ----
        n_random = self.n_subsets - 1
        for i in range(n_random):
            subset = self._sample_subset(features, rng)
            while subset in self.subsets_:
                subset = self._sample_subset(features, rng)

            self.subsets_.append(subset)

            tafc = TargetAwareFeatureCompression(**self.tafc_params)
            sdf = tafc.fit_transform(X[list(subset)], y).reset_index(drop=True)

            col_idx = i + 1

            # core outputs
            out[f"RSTAF_{col_idx}_score"] = sdf["TAFC_score"].to_numpy()
            out[f"RSTAF_{col_idx}_bin"] = sdf["TAFC_bin"].to_numpy()
            out[f"RSTAF_{col_idx}_support"] = sdf["TAFC_support"].to_numpy()

            # --- NEW: propagate class-conditional TAFC ---
            for lbl in self.class_labels_:
                out[f"RSTAF_{col_idx}_score_class_{lbl}"] = sdf[
                    f"TAFC_score_class_{lbl}"
                ].to_numpy()

            self.tafcs_.append(tafc)

        out.index = X_index
        return out#, dict()

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_index = X.index
        if self.subsets_ is None or self.tafcs_ is None:
            raise RuntimeError("Call fit_transform first.")

        X = X.reset_index(drop=True)
        out = X.copy()

        for i, (subset, tafc) in enumerate(zip(self.subsets_, self.tafcs_)):
            if len(subset) == len(X.columns):
                sdf = tafc.transform(X).reset_index(drop=True)
            else:
                sdf = tafc.transform(X[list(subset)]).reset_index(drop=True)

            out[f"RSTAF_{i}_score"] = sdf["TAFC_score"].to_numpy()
            out[f"RSTAF_{i}_bin"] = sdf["TAFC_bin"].to_numpy()
            out[f"RSTAF_{i}_support"] = sdf["TAFC_support"].to_numpy()

            # --- NEW: propagate class-conditional TAFC ---
            for lbl in self.class_labels_:
                out[f"RSTAF_{i}_score_class_{lbl}"] = sdf[
                    f"TAFC_score_class_{lbl}"
                ].to_numpy()
      
        out.index = X_index
        return out

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(
            valid_raw_types=[R_OBJECT, R_CATEGORY, R_BOOL, R_INT, R_FLOAT],
            invalid_special_types=[S_DATETIME_AS_OBJECT, S_IMAGE_PATH, S_IMAGE_BYTEARRAY],
            # required_raw_special_pairs=[
            #     (R_BOOL, None),
            #     (R_OBJECT, None),
            #     (R_CATEGORY, None),
            #     # (R_INT, S_BOOL),
            #     # (R_FLOAT, S_BOOL),
            # ],
        )
