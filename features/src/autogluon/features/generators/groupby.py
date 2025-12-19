import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .abstract import AbstractFeatureGenerator

def q25(series):
    return series.quantile(0.25)

def q75(series):
    return series.quantile(0.75)

def q10(series):
    return series.quantile(0.10)

def q90(series):
    return series.quantile(0.90)

AGGREGATION_REGISTRY = {
    "mean": "mean",
    "std": "std",
    "median": "median",
    "count": "count",
    "min": "min",
    "max": "max",
    "q10": q10,
    "q25": q25,
    "q75": q75,
    "q90": q90,
}

class GroupByFeatureGenerator(AbstractFeatureGenerator):
    """
    Groupby interaction features with flexible relative statistics
    against arbitrary group-level aggregations.

    Supports:
    - standard aggregations ("mean", "std", ...)
    - named callable aggregations ("q25", "q75", ...)
    """

    def __init__(
        self,
        # categorical_features,
        # numeric_features,
        target_type=None,
        aggregations=("mean",),
        relative_to_aggs=("mean",),
        relative_ops=("diff", "ratio"),
        add_zscore=False,
        add_pct_rank=False,
        add_outlier_flags=False,
        add_log_transforms=False,
        add_group_reliability=False,
        global_shrinkage_k=50,
        fill_value=0.0,
        eps=1e-8,
        return_dataframe=True,
        add_low_cardinality=None,
        max_features=None,
        random_state=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_type = target_type
        # self.categorical_features = categorical_features
        # self.numeric_features = numeric_features
        self.relative_to_aggs = relative_to_aggs
        self.relative_ops = relative_ops
        self.aggregations = aggregations
        self.add_zscore = add_zscore
        self.add_pct_rank = add_pct_rank
        self.add_outlier_flags = add_outlier_flags
        self.add_log_transforms = add_log_transforms
        self.add_group_reliability = add_group_reliability
        self.global_shrinkage_k = global_shrinkage_k
        self.max_features = max_features
        self.random_state = random_state

        self.fill_value = fill_value
        self.eps = eps
        self.return_dataframe = return_dataframe

        if add_low_cardinality is not None:
            self.max_cardinality = add_low_cardinality
        else:
            self.max_cardinality = 1e8

        # self.aggregations = {}
        # for agg_ in aggregations:
        #     if agg_ in [f"q{i}" for i in range(0, 101)]:
        #         self.aggregations[agg_] = lambda s: s.quantile(int(agg_[1:]) / 100)
        #     else:
        #         self.aggregations[agg_] = agg_

        unknown = set(self.aggregations) - set(AGGREGATION_REGISTRY)
        if unknown:
            raise ValueError(f"Unknown aggregations: {unknown}")

    def _features_per_pair(self):
        n = len(self.aggregations)  # base aggregations

        # relative features
        n += len(self.relative_to_aggs) * (
            ("diff" in self.relative_ops) +
            ("ratio" in self.relative_ops)
        )

        if self.add_log_transforms:
            n += len(self.relative_to_aggs)

        if self.add_zscore:
            n += 1

        if self.add_outlier_flags:
            n += 2

        if self.add_pct_rank:
            n += 1

        if self.add_group_reliability:
            n += 3

        return int(n)

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------
    def _fit(self, X, y=None):
        self.categorical_features = X.columns[X.nunique()<self.max_cardinality].tolist()
        self.categorical_features += X.select_dtypes(include='category').columns.tolist()
        self.categorical_features = np.unique(self.categorical_features).tolist()
        print(f'Categorical features: {self.categorical_features}')
        self.numeric_features = [col for col in X.columns if col not in self.categorical_features and X[col].dtype not in ['category', 'object']]

        if len(self.categorical_features) == 0 or len(self.numeric_features) == 0:
            return self


        X = self._to_dataframe(X)

        cardinalities = X.nunique(dropna=True).sort_values(ascending=False)
        ranked_features = cardinalities.index.tolist()

        self.group_stats_ = {}

        features_per_pair = self._features_per_pair()
        budget = self.max_features if self.max_features is not None else float("inf")
        used_features = 0

        # ranked by cardinality, independent of type
        ranked_cols = (
            X.nunique(dropna=True)
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        for cat in ranked_cols:
            if cat not in self.categorical_features:
                continue

            for num in ranked_cols:
                if num not in self.numeric_features:
                    continue

                if used_features + features_per_pair > budget:
                    return self  # STOP: budget reached

                named_aggs = {
                    name: pd.NamedAgg(
                        column=num,
                        aggfunc=AGGREGATION_REGISTRY[name]
                    )
                    for name in self.aggregations
                }

                stats = (
                    X.groupby(cat, observed=True)
                    .agg(**named_aggs)
                    .astype(float)
                )

                self.group_stats_[(cat, num)] = stats
                used_features += features_per_pair   

        self.global_stats_ = {
            num: X[num].mean() for num in self.numeric_features
        }

        for cat in self.categorical_features:
            for num in self.numeric_features:

                named_aggs = {
                    name: pd.NamedAgg(
                        column=num,
                        aggfunc=AGGREGATION_REGISTRY[name]
                    )
                    for name in self.aggregations
                }

                stats = (
                    X.groupby(cat, observed=True)
                    .agg(**named_aggs)
                    .astype(float)
                )

                self.group_stats_[(cat, num)] = stats

        return self

    # ------------------------------------------------------------------
    # TRANSFORM
    # ------------------------------------------------------------------
    def _transform(self, X):
        if len(self.categorical_features) == 0 or len(self.numeric_features) == 0:
            return X

        X = self._to_dataframe(X)
        features = []

        for (cat, num), stats in self.group_stats_.items():
            x = X[num].astype(float)

            # ----------------------------
            # Map group statistics
            # ----------------------------
            mapped = {
                agg: X[cat].map(stats[agg]).astype(float)
                for agg in stats.columns
            }

            for agg, values in mapped.items():
                features.append(
                    values.fillna(self.fill_value)
                          .rename(f"{num}__by__{cat}__{agg}")
                )

            # ----------------------------
            # Validate relative aggs
            # ----------------------------
            missing = set(self.relative_to_aggs) - set(mapped)
            if missing:
                raise ValueError(
                    f"Requested relative_to_aggs {missing} "
                    f"not present in computed aggregations {list(mapped)}"
                )

            # ----------------------------
            # Relative features (ANY agg)
            # ----------------------------
            for agg in self.relative_to_aggs:
                ref = mapped[agg]

                if "diff" in self.relative_ops:
                    features.append(
                        (x - ref)
                        .rename(f"{num}__minus__{cat}_{agg}")
                    )

                if "ratio" in self.relative_ops:
                    features.append(
                        (x / (ref + self.eps))
                        .rename(f"{num}__ratio__{cat}_{agg}")
                    )

                if self.add_log_transforms:
                    signed_log_diff = (
                        np.sign(x - ref) *
                        np.log1p(np.abs(x - ref))
                    )
                    features.append(
                        signed_log_diff.rename(
                            f"{num}__signed_log_diff__by__{cat}_{agg}"
                        )
                    )

            # ----------------------------
            # Distribution-aware features
            # ----------------------------
            if self.add_zscore and "mean" in mapped and "std" in mapped:
                z = (x - mapped["mean"]) / (mapped["std"] + self.eps)
                features.append(
                    z.rename(f"{num}__zscore__by__{cat}")
                )

            if self.add_outlier_flags and "mean" in mapped and "std" in mapped:
                features.append(
                    (x > mapped["mean"] + 2 * mapped["std"])
                    .astype(float)
                    .rename(f"{num}__high_outlier__by__{cat}")
                )
                features.append(
                    (x < mapped["mean"] - 2 * mapped["std"])
                    .astype(float)
                    .rename(f"{num}__low_outlier__by__{cat}")
                )

            # ----------------------------
            # Rank-based features
            # ----------------------------
            if self.add_pct_rank:
                pct_rank = (
                    X.groupby(cat, observed=True)[num]
                     .rank(method="average", pct=True)
                     .fillna(0.5)
                )
                features.append(
                    pct_rank.rename(f"{num}__pct_rank__by__{cat}")
                )

            # ----------------------------
            # Group reliability / shrinkage
            # ----------------------------
            if self.add_group_reliability and "mean" in mapped and "count" in mapped:
                global_mean = self.global_stats_[num]
                count = mapped["count"]
                k = self.global_shrinkage_k

                shrunk_mean = (
                    mapped["mean"] * count + global_mean * k
                ) / (count + k)

                features.append(
                    shrunk_mean.rename(
                        f"{num}__shrunk_mean__by__{cat}"
                    )
                )

                features.append(
                    (count / (count + k))
                    .rename(f"{num}__mean_confidence__by__{cat}")
                )

                features.append(
                    (mapped["mean"] - global_mean)
                    .rename(f"{num}__mean_minus_global__by__{cat}")
                )

        result = pd.concat(features, axis=1)
        return result if self.return_dataframe else result.values

    def _fit_transform(self, X, y):
        self._fit(X, y)
        return self._transform(X), dict()

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        raise ValueError("Input must be a pandas DataFrame")


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(
            # valid_raw_types=[R_OBJECT, R_CATEGORY, R_BOOL, R_INT, R_FLOAT],
            # invalid_special_types=[S_DATETIME_AS_OBJECT, S_IMAGE_PATH, S_IMAGE_BYTEARRAY],
            # required_raw_special_pairs=[
            #     (R_BOOL, None),
            #     (R_OBJECT, None),
            #     (R_CATEGORY, None),
            #     # (R_INT, S_BOOL),
            #     # (R_FLOAT, S_BOOL),
            # ],
        )
