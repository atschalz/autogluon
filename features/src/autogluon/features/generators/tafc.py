from __future__ import annotations
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from .abstract import AbstractFeatureGenerator
from .oof_target_encoder import OOFTargetEncodingFeatureGenerator

from contextlib import contextmanager
from time import perf_counter


class TimerLog:
    # TODO: Mainly used for debugging and tracking runtimes during development. Not needed for preprocessing logic. Better remove?
    def __init__(self):
        self.times = {}

    @contextmanager
    def block(self, name: str):
        t0 = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - t0
            self.times[name] = self.times.get(name, 0) + dt

    def summary(self, verbose: bool = False) -> dict:
        if verbose:
            print("\n--- Timing Summary (in order) ---")
            for name, total in self.times.items():
                print(f"{name:<20} {total:.3f}s")
        return dict(self.times)


class TargetAwareFeatureCompressionFeatureGenerator(AbstractFeatureGenerator):
    """
    Target-Aware Feature Compression (TAFC)

    Creates a deterministic per-row key from selected columns (optionally rounded),
    then applies out-of-fold target encoding on that key to produce a compressed,
    target-aware feature.

    Returns:
      - For binary/regression: TAFC_score
      - For multiclass: TAFC_score_class_<k> (+ optional TAFC_score if multi_class_as_regression)
    """

    def __init__(
        self,
        target_type: str,  # "binary", "multiclass", "regression"
        n_splits: int = 5,
        alpha: float = 0.0,
        only_cat: bool = False,
        max_cardinality: Optional[int] = None,
        round_numerical: int = 2,
        multi_class_as_regression: bool = False,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if target_type not in {"binary", "multiclass", "regression"}:
            raise ValueError(f"target_type must be one of {{binary,multiclass,regression}}, got {target_type}")

        self.target_type = target_type
        self.random_state = int(random_state)
        self.n_splits = int(n_splits)
        self.alpha = float(alpha)

        self.only_cat = bool(only_cat)
        self.max_cardinality = int(max_cardinality) if max_cardinality is not None else None
        self.round_numerical = int(round_numerical)
        self.multi_class_as_reg = bool(multi_class_as_regression)

        # fitted state
        self.tafc_useless_: bool = False
        self.n_classes_: int = 0
        self.oof_te_: Optional[OOFTargetEncodingFeatureGenerator] = None
        self.oof_te_multi_as_reg_: Optional[OOFTargetEncodingFeatureGenerator] = None

        self.timelog = TimerLog()

    def _make_key(self, X: pd.DataFrame) -> pd.Series:
        """
        Build a deterministic key per row by hashing selected columns.
        """
        if X.shape[1] == 0:
            # Degenerate: no columns; hash empty rows
            return pd.Series(pd.util.hash_pandas_object(X, index=False).astype("uint64").astype(str), index=X.index)

        # Select columns
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

        # Round numeric columns (copy only when needed)
        num_cols = X_key.select_dtypes(include=["number"]).columns
        if len(num_cols) > 0 and self.round_numerical is not None:
            X_key = X_key.copy()
            X_key.loc[:, num_cols] = X_key.loc[:, num_cols].round(self.round_numerical)

        # hash rows -> uint64 -> str (OOF-TE expects single column input)
        return pd.util.hash_pandas_object(X_key, index=False).astype("uint64").astype(str)

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series):
        with self.timelog.block("make_key"):
            key = self._make_key(X)

        # If the key is unique per row, TE won't generalize; skip.
        if key.nunique(dropna=False) == X.shape[0]:
            self.tafc_useless_ = True
            return pd.DataFrame(index=X.index), {}

        self.tafc_useless_ = False

        self.oof_te_ = OOFTargetEncodingFeatureGenerator(
            target_type=self.target_type,
            random_state=self.random_state,
            n_splits=self.n_splits,
            alpha=self.alpha,
            verbosity=0,
        )
        with self.timelog.block("fit_oof_te"):
            out = pd.DataFrame(index=X.index)
            key_df = key.to_frame(name="TAFC_key")

            if self.target_type == "multiclass":
                self.n_classes_ = int(pd.Series(y).nunique(dropna=False))

                if self.multi_class_as_reg:
                    self.oof_te_multi_as_reg_ = OOFTargetEncodingFeatureGenerator(
                        target_type="regression",
                        random_state=self.random_state,
                        n_splits=self.n_splits,
                        alpha=self.alpha,
                        verbosity=0,
                    )
                    out["TAFC_score"] = self.oof_te_multi_as_reg_.fit_transform(key_df, y)

                class_cols = [f"TAFC_score_class_{i}" for i in range(self.n_classes_)]
                out[class_cols] = self.oof_te_.fit_transform(key_df, y)
            else:
                out["TAFC_score"] = self.oof_te_.fit_transform(key_df, y)

        return out, {}

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.tafc_useless_:
            return pd.DataFrame(index=X.index)

        if self.oof_te_ is None:
            raise RuntimeError(
                "TargetAwareFeatureCompressionFeatureGenerator is not fitted. Call fit_transform first."
            )

        key = self._make_key(X)
        key_df = key.to_frame(name="TAFC_key")

        out = pd.DataFrame(index=X.index)
        if self.target_type == "multiclass":
            if self.multi_class_as_reg:
                if self.oof_te_multi_as_reg_ is None:
                    raise RuntimeError("multi_class_as_regression=True but regression TE model was not fitted.")
                out["TAFC_score"] = self.oof_te_multi_as_reg_.transform(key_df)

            class_cols = [f"TAFC_score_class_{i}" for i in range(self.n_classes_)]
            out[class_cols] = self.oof_te_.transform(key_df)
        else:
            out["TAFC_score"] = self.oof_te_.transform(key_df)

        return out

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return {}


class RandomSubsetTAFC(AbstractFeatureGenerator):
    """
    Random Subset TAFC (RSTAF)

    Runs TAFC on:
      - full feature set (always attempted)
      - additional random subsets of features

    Output columns:
      RSTAF_0_*  -> full-feature TAFC
      RSTAF_1_*  -> first random subset
      ...
    """

    def __init__(
        self,
        target_type: Literal["binary", "multiclass", "regression"],
        only_cat: bool = False,
        binary_as_cat: bool = True,
        max_cardinality: Optional[int] = None,
        round_numerical: Optional[int] = 2,
        n_subsets: int = 50,
        subset_size: Optional[int] = None,
        min_subset_size: int = 2,
        max_subset_size: Optional[int] = None,
        max_base_feats_to_consider: Optional[int] = 150,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.target_type = target_type
        self.n_subsets = int(n_subsets)
        self.subset_size = subset_size
        self.min_subset_size = int(min_subset_size)
        self.max_subset_size = max_subset_size
        self.random_state = int(random_state)
        self.only_cat = bool(only_cat)
        self.binary_as_cat = bool(binary_as_cat)
        self.max_cardinality = int(max_cardinality) if max_cardinality is not None else None
        self.round_numerical = int(round_numerical) if round_numerical is not None else None

        self.max_base_feats_to_consider = (
            int(max_base_feats_to_consider) if max_base_feats_to_consider is not None else None
        )

        # fitted state
        self.base_features_: Optional[tuple[str, ...]] = None

        self.timelog = TimerLog()

    @staticmethod
    def _select_top_mode_features(X: pd.DataFrame, k: Optional[int]) -> list[str]:
        """
        Pick top-k columns by mode strength:
          max(value_count(col)) / n_rows   (dropna=False)
        """
        if k is None or k <= 0 or X.shape[1] == 0:
            return list(X.columns)

        k = min(k, X.shape[1])
        n = len(X)
        if n == 0:
            return list(X.columns)[:k]

        scores = {}
        for c in X.columns:
            vc = X[c].value_counts(dropna=False)
            top = int(vc.iloc[0]) if len(vc) else 0
            scores[c] = top / n

        return sorted(scores.keys(), key=lambda c: (-scores[c], str(c)))[:k]

    @staticmethod
    def _select_features_by_dtype_and_cardinality(X: pd.DataFrame, k: Optional[int]) -> list[str]:
        if k is not None and k <= 0:
            return []

        ordered = (
            list(X.select_dtypes(include="category").nunique().sort_values().index)
            + list(X.select_dtypes(include="object").nunique().sort_values().index)
            + list(X.columns[X.nunique() == 2])
            + list(X.select_dtypes(include="integer").nunique().sort_values().index)  # Note: Just int not captured
            + list(X.select_dtypes(include="float").nunique().sort_values().index)
        )

        # remove duplicates, keep order
        ordered = list(dict.fromkeys(ordered))

        return ordered if k is None else ordered[:k]

    @staticmethod
    def _sample_unique_subsets(
        features: Sequence[str],
        rng: np.random.Generator,
        n: int,
        subset_size: Optional[int],
        min_k: int,
        max_k: Optional[int],
        max_tries_multiplier: int = 50,
    ) -> list[tuple[str, ...]]:
        feats = np.asarray(list(features), dtype=object)
        p = len(feats)
        if p <= 1 or n <= 0:
            return []

        max_k_eff = min(max_k if max_k is not None else (p - 1), p - 1)
        min_k_eff = max(min_k, 1)

        if subset_size is not None:
            k_min = k_max = int(subset_size)
            if not (1 <= k_min <= p - 1):
                return []
        else:
            k_min, k_max = min_k_eff, max_k_eff
            if k_min > k_max:
                return []

        selected: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        max_tries = max_tries_multiplier * n

        for _ in range(max_tries):
            if len(selected) >= n:
                break

            k = k_min if subset_size is not None else int(rng.integers(k_min, k_max + 1))
            subset = tuple(sorted(rng.choice(feats, size=k, replace=False).tolist()))
            if subset in seen:
                continue
            seen.add(subset)
            selected.append(subset)

        return selected

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.shape[1] == 0:
            # Degenerate: no columns; hash empty rows
            return pd.Series(pd.util.hash_pandas_object(X, index=False).astype("uint64").astype(str), index=X.index)

        # Select columns
        if self.only_cat:
            cols = X.select_dtypes(include=["object", "category"]).columns
            numeric_cols = X.select_dtypes(include=["number"]).columns
            if self.binary_as_cat:
                binary_cols = X.select_dtypes(include=["number"]).columns[X[numeric_cols].nunique() <= 2] # NOTE: uniform may occur at test, hence <=, should generally make train/test prepare versions
                cols = cols.union(binary_cols)
        else:
            cols = X.columns

        if self.max_cardinality is not None and len(cols) > 0:
            nunique = X[cols].nunique(dropna=False)
            cols = nunique[nunique < self.max_cardinality].index

        if len(cols) == 0:
            cols = X.columns

        X_candidates = X.loc[:, cols]

        # Round numeric columns (copy only when needed)
        num_cols = X_candidates.select_dtypes(include=["number"]).columns
        if len(num_cols) > 0 and self.round_numerical is not None:
            X_candidates = X_candidates.copy()
            X_candidates.loc[:, num_cols] = X_candidates.loc[:, num_cols].round(self.round_numerical)

        return X_candidates

    def _make_key(self, X: pd.DataFrame) -> pd.Series:
        """
        Build a deterministic key per row by hashing selected columns.
        """
        # hash rows -> uint64 -> str (OOF-TE expects single column input)
        return pd.util.hash_pandas_object(X, index=False).astype("uint64").astype(str)

    @staticmethod
    def collapse_singletons(s, threshold=1, label="__single__"):
        vc = s.value_counts()
        return s.where(s.map(vc) > threshold, label)

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series):
        # Prepare X
        with self.timelog.block("prepare_input"):
            X_local = self._prepare_X(X)

        # Restrict base feature space if requested
        with self.timelog.block("select_base_features"):
            if self.max_base_feats_to_consider is not None and X.shape[1] > 0:
                selected = self._select_features_by_dtype_and_cardinality(X_local, self.max_base_feats_to_consider)
                X_local = X_local[selected]
                self.base_features_ = list(selected)
            else:
                self.base_features_ = list(X.columns)

        features = list(X_local.columns)
        rng = np.random.default_rng(self.random_state)

        # ---- 1..n) Random subsets ----
        with self.timelog.block("select_random_subsets"):
            self.selected_subsets = [tuple(features)]  # always include full set
            n_random = max(self.n_subsets - 1, 0)
            self.selected_subsets += self._sample_unique_subsets(
                features=features,
                rng=rng,
                n=n_random,
                subset_size=self.subset_size,
                min_k=self.min_subset_size,
                max_k=self.max_subset_size,
            )

        with self.timelog.block("make_key"):
            X_str = pd.concat([self._make_key(X_local[list(i)]) for i in self.selected_subsets], axis=1)

        # with self.timelog.block("filter_uninformative_keys"): # Improves efficiency but hurts performance
        #     X_str = X_str.apply(self.collapse_singletons)

        with self.timelog.block("oof-te"):
            self.subset_oof = OOFTargetEncodingFeatureGenerator(target_type=self.target_type, verbosity=0, alpha=0, random_state=self.random_state)
            X_oof = self.subset_oof.fit_transform(X_str, y)

        self.col_names = [f"RSTAF_{i}_{i}" for i in range(X_oof.shape[1])]
        X_oof.columns = self.col_names

        return X_oof, {}

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Prepare X
        with self.timelog.block("transform_prepare_input"):
            X_local = self._prepare_X(X[self.base_features_])
        with self.timelog.block("transform_make_key"):
            X_str = pd.concat([self._make_key(X_local[list(i)]) for i in self.selected_subsets], axis=1)
        with self.timelog.block("transform_oof-transform"):
            out = self.subset_oof.transform(X_str)
            out.columns = self.col_names
        return out

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return {}
