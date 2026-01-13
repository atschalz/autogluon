import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from itertools import combinations


from .abstract import AbstractFeatureGenerator
from typing import Literal

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
        only_cat=False,
        max_cardinality=None,
        add_bins=False,
        add_support=False,
        round_numerical: int = 2,   # <-- NEW
        return_oof: bool = True,   # <-- NEW        
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_bins = int(n_bins)
        self.n_splits = int(n_splits)
        self.smoothing = float(smoothing)
        self.random_state = int(random_state)
        self.stratify = bool(stratify)
        self.target_type = target_type
        self.only_cat = bool(only_cat)
        self.max_cardinality = (
            int(max_cardinality) if max_cardinality is not None else None
        )
        self.add_bins = bool(add_bins)
        self.add_support = bool(add_support)

        self.round_numerical = int(round_numerical)  # <-- NEW
        self.return_oof = return_oof


    def _make_key(self, X: pd.DataFrame) -> pd.Series:
        cols = X.columns

        # filter by categorical dtype if requested
        if self.only_cat:
            cols = X.select_dtypes(include=["object", "category"]).columns

        # filter by cardinality if requested
        if self.max_cardinality is not None and len(cols) > 0:
            nunique = X[cols].nunique(dropna=False)
            cols = nunique[nunique < self.max_cardinality].index

        # fallback: avoid empty key
        if len(cols) == 0:
            cols = X.columns

        X_key = X[cols].copy()

        # --- NEW: round numerical columns ---
        num_cols = X_key.select_dtypes(include=["float", "int"]).columns
        if len(num_cols) > 0:
            X_key[num_cols] = X_key[num_cols].round(self.round_numerical)

        return X_key.astype(str).agg("|".join, axis=1)



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
        
        full_score = key.map(self.group_stats_["smoothed"]).fillna(self.global_mean_).to_numpy()
        full_support = key.map(self.group_stats_["count"]).fillna(0).to_numpy()

        bins = np.digitize(oof_score, self.bin_edges_[1:-1], right=True)

        # ----- output -----
        out = X.copy()
        if self.return_oof:
            out["TAFC_score"] = oof_score
        else:
            out["TAFC_score"] = full_score
        if self.add_bins:
            out["TAFC_bin"] = bins.astype(int)
        if self.add_support:
            if self.return_oof:
                out["TAFC_support"] = oof_support.astype(int)
            else:
                out["TAFC_support"] = full_support.astype(int)

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
        if self.add_bins:
            out["TAFC_bin"] = bins.astype(int)
        if self.add_support:
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


class RandomSubsetTAFC(AbstractFeatureGenerator):
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
        target_type: Literal["binary", "multiclass", "regression"],  # "binary", "multiclass", "regression"
        base_tafc_params=None,
        subset_tafc_params=None,
        n_subsets=50,          # total TAFCs INCLUDING full set
        subset_size=5,
        min_subset_size=2,
        max_subset_size=None,
        random_state=42,
        use_meta_features=False,
        drop_raw_rstaf=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_tafc_params = base_tafc_params or {}
        self.subset_tafc_params = subset_tafc_params or {}
        if 'target_type' not in self.base_tafc_params:
            self.base_tafc_params['target_type'] = target_type
        if 'target_type' not in self.subset_tafc_params:
            self.subset_tafc_params['target_type'] = target_type
        self.n_subsets = int(n_subsets)
        self.subset_size = subset_size
        self.min_subset_size = int(min_subset_size)
        self.max_subset_size = max_subset_size
        self.random_state = int(random_state)
        self.target_type = target_type

        self.subsets_ = None
        self.tafcs_ = None
        self.class_labels_ = None   # <-- NEW

        self.use_meta_features = use_meta_features      # toggle if you want
        self.drop_raw_rstaf = drop_raw_rstaf         # strongly recommended
        self.meta_model_ = None
        self.meta_feature_names_ = None

    def _collect_rstaf_numeric_features(self, df):
        """
        Collect all numeric RSTAF features in a stable order.
        """
        cols = [c for c in df.columns if c.startswith("RSTAF_")]
        cols = sorted(cols)
        return df[cols], cols

    def _meta_predict(self, X):
        """
        Returns meta-features as 2D array.
        """
        preds = self.meta_model_.predict(X)

        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        return preds

    def _all_valid_subsets(self, features):
        n_features = len(features)

        if self.subset_size is not None:
            sizes = [self.subset_size]
        else:
            min_k = self.min_subset_size
            max_k = self.max_subset_size or (n_features - 1)
            sizes = range(min_k, max_k + 1)

        all_subsets = []
        for k in sizes:
            if 0 < k < n_features:
                all_subsets.extend(combinations(features, k))

        return all_subsets

    def _sample_subset(self, features, rng):
        if self.subset_size is not None:
            k = self.subset_size
        else:
            k_max = self.max_subset_size or len(features) - 1
            k = rng.integers(self.min_subset_size, k_max + 1)

        return tuple(sorted(rng.choice(features, size=k, replace=False)))
    
    def _sample_unique_subsets(
        self,
        features,
        rng,
        n: int,
        subset_size=None,
        min_k: int = 2,
        max_k=None,
        max_tries_multiplier: int = 50,
    ):
        """
        Sample up to n unique feature-subsets without enumerating all combinations.
        Returns a list[tuple[str]] of sorted feature names.

        - If subset_size is set: fixed-size subsets.
        - Else: random size in [min_k, max_k] each draw.
        """
        features = np.asarray(features)
        p = len(features)
        if p <= 1 or n <= 0:
            return []

        if max_k is None:
            max_k = p - 1
        max_k = min(max_k, p - 1)
        min_k = max(min_k, 1)

        if subset_size is not None:
            k_min = k_max = int(subset_size)
            if not (1 <= k_min <= p - 1):
                return []
        else:
            k_min, k_max = min_k, max_k
            if k_min > k_max:
                return []

        selected = []
        seen = set()

        # Prevent infinite loops when the space is small or constraints tight.
        max_tries = max_tries_multiplier * n

        tries = 0
        while len(selected) < n and tries < max_tries:
            tries += 1

            if subset_size is not None:
                k = k_min
            else:
                k = int(rng.integers(k_min, k_max + 1))

            subset = tuple(sorted(rng.choice(features, size=k, replace=False).tolist()))
            if subset in seen:
                continue

            seen.add(subset)
            selected.append(subset)

        return selected


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

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        from tabarena.benchmark.models.prep_ag.prep_lgb.linear_init import CustomLinearModel
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

        tafc_full = TargetAwareFeatureCompressionFeatureGenerator(**self.base_tafc_params)
        sdf = tafc_full.fit_transform(X, y).reset_index(drop=True)

        # core outputs
        out["RSTAF_0_score"] = sdf["TAFC_score"].to_numpy()
        if self.base_tafc_params.get("add_bins", False):
            out["RSTAF_0_bin"] = sdf["TAFC_bin"].to_numpy()
        if self.base_tafc_params.get("add_support", False):
            out["RSTAF_0_support"] = sdf["TAFC_support"].to_numpy()

        # --- NEW: detect multiclass channels once ---
        if self.base_tafc_params.get("target_type") == "multiclass":
            self.class_labels_ = self._extract_class_labels(sdf)
            # --- NEW: propagate class-conditional TAFC ---
            for lbl in self.class_labels_:
                out[f"RSTAF_0_score_class_{lbl}"] = sdf[
                    f"TAFC_score_class_{lbl}"
                ].to_numpy()

        else:
            self.class_labels_ = []


        self.tafcs_.append(tafc_full)

        # ---- 1..n) Random subsets ----
        if out["RSTAF_0_score"].nunique() == 1:
            # Full-feature TAFC is constant -> no point in adding more
            n_random = 0
        else:
            n_random = self.n_subsets - 1

        selected_subsets = self._sample_unique_subsets(
            features=features,
            rng=rng,
            n=n_random,
            subset_size=self.subset_size,
            min_k=self.min_subset_size,
            max_k=self.max_subset_size,
        )


        # all_subsets = self._all_valid_subsets(features)

        # if len(all_subsets) == 0:
        #     # No valid subsets possible (e.g. single feature)
        #     selected_subsets = []
        # else:
        #     rng.shuffle(all_subsets)
        #     selected_subsets = all_subsets[: min(n_random, len(all_subsets))]

        new_cols = {}
        for i, subset in enumerate(selected_subsets):
            subset = tuple(subset)
            self.subsets_.append(subset)

            tafc = TargetAwareFeatureCompressionFeatureGenerator(**self.subset_tafc_params)
            sdf = tafc.fit_transform(X[list(subset)], y).reset_index(drop=True)

            col_idx = i + 1

            new_cols[f"RSTAF_{col_idx}_score"] = sdf["TAFC_score"].to_numpy()

            if self.subset_tafc_params.get("add_bins", False):
                new_cols[f"RSTAF_{col_idx}_bin"] = sdf["TAFC_bin"].to_numpy()

            if self.subset_tafc_params.get("add_support", False):
                new_cols[f"RSTAF_{col_idx}_support"] = sdf["TAFC_support"].to_numpy()

            if self.base_tafc_params.get("target_type") == "multiclass":
                for lbl in self.class_labels_:
                    new_cols[f"RSTAF_{col_idx}_score_class_{lbl}"] = sdf[
                        f"TAFC_score_class_{lbl}"
                    ].to_numpy()

            self.tafcs_.append(tafc)

        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)


        # ============================
        # META AGGREGATION (GENERAL)
        # ============================
        if self.use_meta_features:

            X_meta, meta_cols = self._collect_rstaf_numeric_features(out)

            self.meta_model_ = CustomLinearModel(target_type=self.base_tafc_params.get("target_type"), random_state=self.random_state)
            self.meta_model_.fit(X_meta, y)

            meta_preds = self._meta_predict(X_meta)

            self.meta_feature_names_ = [
                f"RSTAF_META_{i}" for i in range(meta_preds.shape[1])
            ]

            for i, name in enumerate(self.meta_feature_names_):
                out[name] = meta_preds[:, i]

            if self.drop_raw_rstaf:
                out = out.drop(columns=meta_cols)


        out.index = X_index
        return out, dict()

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_index = X.index
        if self.subsets_ is None or self.tafcs_ is None:
            raise RuntimeError("Call fit_transform first.")

        X = X.reset_index(drop=True)
        out = X.copy()

        new_cols = {}
        for i, (subset, tafc) in enumerate(zip(self.subsets_, self.tafcs_)):
            if len(subset) == len(X.columns):
                sdf = tafc.transform(X).reset_index(drop=True)
            else:
                sdf = tafc.transform(X[list(subset)]).reset_index(drop=True)

            if i == 0:
                tafc_params = self.base_tafc_params
            else:
                tafc_params = self.subset_tafc_params
           
            new_cols[f"RSTAF_{i}_score"] = sdf["TAFC_score"].to_numpy()

            if self.subset_tafc_params.get("add_bins", False):
                new_cols[f"RSTAF_{i}_bin"] = sdf["TAFC_bin"].to_numpy()

            if self.subset_tafc_params.get("add_support", False):
                new_cols[f"RSTAF_{i}_support"] = sdf["TAFC_support"].to_numpy()

            if self.base_tafc_params.get("target_type") == "multiclass":
                for lbl in self.class_labels_:
                    new_cols[f"RSTAF_{i}_score_class_{lbl}"] = sdf[
                        f"TAFC_score_class_{lbl}"
                    ].to_numpy()
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

        # ============================
        # META AGGREGATION (GENERAL)
        # ============================
        if self.use_meta_features:

            X_meta, meta_cols = self._collect_rstaf_numeric_features(out)

            meta_preds = self._meta_predict(X_meta)

            for i, name in enumerate(self.meta_feature_names_):
                out[name] = meta_preds[:, i]

            if self.drop_raw_rstaf:
                out = out.drop(columns=meta_cols)

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
