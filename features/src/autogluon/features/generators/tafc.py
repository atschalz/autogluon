import numpy as np
import pandas as pd
from itertools import combinations


from .abstract import AbstractFeatureGenerator
from .oof_target_encoder import OOFTargetEncodingFeatureGenerator
from typing import Literal


class TargetAwareFeatureCompressionFeatureGenerator(AbstractFeatureGenerator):
    """
    Target-Aware Feature Compression (TAFC)
    """
    def __init__(
        self,
        target_type:str,  # "binary", "multiclass", "regression"
        n_splits=5,
        alpha=0,
        only_cat=False,
        max_cardinality=None,
        round_numerical: int = 2,   
        multi_class_as_regression: bool = False,
        random_state=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_type = target_type
        self.random_state = random_state
        self.n_splits = n_splits
        self.alpha = alpha

        self.only_cat = bool(only_cat)
        self.max_cardinality = (
            int(max_cardinality) if max_cardinality is not None else None
        )
        self.round_numerical = int(round_numerical)  
        self.multi_class_as_reg = multi_class_as_regression

        self.tafc_useless = False


    def _make_key(self, X: pd.DataFrame) -> pd.Series:
        # pick columns
        if self.only_cat:
            cols = X.select_dtypes(include=["object", "category"]).columns
        else:
            cols = X.columns

        if self.max_cardinality is not None and len(cols) > 0:
            nunique = X[cols].nunique(dropna=False)
            cols = nunique[nunique < self.max_cardinality].index

        if len(cols) == 0:
            cols = X.columns

        X_key = X.loc[:, cols]

        # round numeric cols without copying entire frame
        num_cols = X_key.select_dtypes(include=["float", "int"]).columns
        if len(num_cols) > 0:
            X_key = X_key.copy()
            X_key.loc[:, num_cols] = X_key.loc[:, num_cols].round(self.round_numerical)

        # hash each row -> deterministic uint64, then cast to string (OOF TE expects a single col)
        key = pd.util.hash_pandas_object(X_key, index=False).astype("uint64").astype(str)
        return key

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        key = self._make_key(X)
        if key.nunique()==X.shape[0]:
            self.tafc_useless = True
            out = pd.DataFrame(index=X.index)
            return out, dict()

        self.oof_te = OOFTargetEncodingFeatureGenerator(target_type=self.target_type,random_state=self.random_state, n_splits=self.n_splits, alpha=self.alpha)

        out = pd.DataFrame(index=X.index)
        if self.target_type == "multiclass":
            self.n_classes_ = len(np.unique(y))
            if self.multi_class_as_reg:
                self.oof_te_multi_as_reg = OOFTargetEncodingFeatureGenerator(target_type="regression", random_state=self.random_state, n_splits=self.n_splits, alpha=self.alpha)
                out['TAFC_score'] = self.oof_te_multi_as_reg.fit_transform(key.to_frame(), y)
            out[[f"TAFC_score_class_{i}" for i in range(self.n_classes_)]] = self.oof_te.fit_transform(key.to_frame(), y)
        else:
            out['TAFC_score'] = self.oof_te.fit_transform(key.to_frame(), y)    
        
        return out, dict()

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.tafc_useless:
            out = pd.DataFrame(index=X.index)
            return out
        key = self._make_key(X)
        out = pd.DataFrame(index=X.index)
        if self.target_type == "multiclass":
            if self.multi_class_as_reg:
                out['TAFC_score']  = self.oof_te_multi_as_reg.transform(key.to_frame())
            out[[f"TAFC_score_class_{i}" for i in range(self.n_classes_)]] = self.oof_te.transform(key.to_frame())
        else:
            out['TAFC_score'] = self.oof_te.transform(key.to_frame())
    
        return out

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(
            # valid_raw_types=[R_OBJECT, R_CATEGORY, R_BOOL, R_INT, R_FLOAT],
            # invalid_special_types=[S_DATETIME_AS_OBJECT, S_IMAGE_PATH, S_IMAGE_BYTEARRAY],
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
        multiclass_as_regression: bool = False,
        n_subsets=50,          # 50
        subset_size=None, # 5
        min_subset_size=2, # 2
        max_subset_size=None,
        max_base_feats_to_consider=50,  
        stop_at_tafc_if_empty=True,
        random_state=42,
        use_meta_features=False, # False
        drop_raw_rstaf=False, # False
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_tafc_params = base_tafc_params or {}
        self.subset_tafc_params = subset_tafc_params or {}
        if 'target_type' not in self.base_tafc_params:
            self.base_tafc_params['target_type'] = target_type
        if 'target_type' not in self.subset_tafc_params:
            self.subset_tafc_params['target_type'] = target_type
        self.multiclass_as_regression = multiclass_as_regression
        self.n_subsets = int(n_subsets)
        self.subset_size = subset_size
        self.min_subset_size = int(min_subset_size)
        self.max_subset_size = max_subset_size
        self.random_state = int(random_state)
        self.target_type = target_type
        self.max_base_feats_to_consider = (
            int(max_base_feats_to_consider) if max_base_feats_to_consider is not None else None
        )
        self.stop_at_tafc_if_empty = bool(stop_at_tafc_if_empty)
        self.base_features_ = None
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

    def _select_top_mode_features(self, X: pd.DataFrame, k: int):
        """
        Pick top-k columns by mode strength:
          max(value_count(col))/n_rows  (with dropna=False)
        Returns list[str] in descending order.
        """
        if k is None or k <= 0:
            return list(X.columns)

        k = min(k, X.shape[1])
        n = len(X)
        if n == 0:
            return list(X.columns)[:k]

        scores = {}
        for c in X.columns:
            vc = X[c].value_counts(dropna=False)
            # if all NaN or empty, vc can be empty; guard:
            top = int(vc.iloc[0]) if len(vc) else 0
            scores[c] = top / n

        # Highest mode strength first; stable tie-break by column name
        ordered = sorted(scores.keys(), key=lambda c: (-scores[c], str(c)))
        return ordered[:k]

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        from tabarena.benchmark.models.prep_ag.prep_lgb.linear_init import CustomLinearModel
        X_index = X.index
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        # restrict features by highest mode strength ---
        if self.max_base_feats_to_consider is not None and X.shape[1] > 0:
            selected = self._select_top_mode_features(X, self.max_base_feats_to_consider)
            X = X[selected]
            self.base_features_ = tuple(selected)
        else:
            self.base_features_ = tuple(X.columns)

        features = np.array(X.columns)
        rng = np.random.default_rng(self.random_state)

        self.subsets_ = []
        self.tafcs_ = []

        out = pd.DataFrame(index=X.index)

        # ---- 0) Full-feature TAFC (always) ----
        full_subset = tuple(features.tolist())

        tafc_full = TargetAwareFeatureCompressionFeatureGenerator(**self.base_tafc_params, verbosity=0)
        sdf = tafc_full.fit_transform(X, y).reset_index(drop=True)

        # core outputs
        if 'TAFC_score' in sdf.columns:
            out["RSTAF_0_score"] = sdf["TAFC_score"].to_numpy()
            self.subsets_.append(full_subset)

        # --- NEW: detect multiclass channels once ---
        if self.base_tafc_params.get("target_type") == "multiclass":
            self.class_labels_ = self._extract_class_labels(sdf)
            if sdf.shape[1]>1:
                # --- NEW: propagate class-conditional TAFC ---
                for lbl in self.class_labels_:
                    out[f"RSTAF_0_score_class_{lbl}"] = sdf[
                        f"TAFC_score_class_{lbl}"
                    ].to_numpy()

                if len(self.subsets_) == 0:
                    self.subsets_.append(full_subset)

        else:
            self.class_labels_ = []

        if self.stop_at_tafc_if_empty and len(self.subsets_) == 0:
            # no useful TAFC could be computed
            return out, dict()
    
        self.tafcs_.append(tafc_full)

        # ---- 1..n) Random subsets ----
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
        col_idx = 1

        for i, subset in enumerate(selected_subsets):
            subset = tuple(subset)

            tafc = TargetAwareFeatureCompressionFeatureGenerator(**self.subset_tafc_params, verbosity=0)
            sdf = tafc.fit_transform(X[list(subset)], y).reset_index(drop=True)
            if sdf.shape[1]>0:
                self.subsets_.append(subset)

                if 'TAFC_score' in sdf.columns:
                    new_cols[f"RSTAF_{col_idx}_score"] = sdf["TAFC_score"].to_numpy()

                if self.base_tafc_params.get("target_type") == "multiclass":
                    if sdf.shape[1]>1:
                        for lbl in self.class_labels_:
                            new_cols[f"RSTAF_{col_idx}_score_class_{lbl}"] = sdf[
                                f"TAFC_score_class_{lbl}"
                            ].to_numpy()

                self.tafcs_.append(tafc)
                col_idx += 1

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
        out = pd.DataFrame(index=X.index)

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
           
            if 'TAFC_score' in sdf.columns:
                new_cols[f"RSTAF_{i}_score"] = sdf["TAFC_score"].to_numpy()

            if self.base_tafc_params.get("target_type") == "multiclass" and sdf.shape[1]>1:
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
        return dict()
