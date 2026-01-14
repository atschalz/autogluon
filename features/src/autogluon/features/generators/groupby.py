import numpy as np
import pandas as pd

from .abstract import AbstractFeatureGenerator

def q25(series): return series.quantile(0.25)
def q75(series): return series.quantile(0.75)
def q10(series): return series.quantile(0.10)
def q90(series): return series.quantile(0.90)

AGGREGATION_REGISTRY = {
    "mean":   {"kind": "group", "agg": "mean"},
    "std":    {"kind": "group", "agg": "std"},
    "median": {"kind": "group", "agg": "median"},
    "count":  {"kind": "group", "agg": "count"},
    "nunique": {"kind": "group", "agg": pd.Series.nunique},
    "min":    {"kind": "group", "agg": "min"},
    "max":    {"kind": "group", "agg": "max"},
    "q10":    {"kind": "group", "agg": q10},
    "q25":    {"kind": "group", "agg": q25},
    "q75":    {"kind": "group", "agg": q75},
    "q90":    {"kind": "group", "agg": q90},
    "pct_rank": {"kind": "rowwise"},
}

import numpy as np
import pandas as pd

def rank_categoricals_by_small_counts(
    X: pd.DataFrame,
    categorical_cols,
    min_count: int = 1,
    top_k_smallest: int = 10,   # how many tie-breakers to use
    require_at_least_levels: int = 2,
    observed: bool = True,
):
    """
    Returns categorical_cols sorted best->worst by lexicographic comparison of the
    smallest group sizes (min, 2nd-min, ...).

    Score vector per cat:
      v = sorted(counts[counts >= min_count])[:top_k_smallest]
    Pad with +inf to fixed length so fewer levels doesn't get penalized.
    Sort by v descending lexicographically.
    """
    scores = {}
    for cat in categorical_cols:
        counts = X[cat].value_counts(dropna=True)
        counts = counts[counts >= min_count].sort_values()  # ascending

        if len(counts) < require_at_least_levels:
            # can't form meaningful group stats; rank it last
            v = np.full(top_k_smallest, -np.inf, dtype=float)
        else:
            v = counts.to_numpy(dtype=float)[:top_k_smallest]
            if v.size < top_k_smallest:
                v = np.pad(v, (0, top_k_smallest - v.size), constant_values=np.inf)

        scores[cat] = v

    # Sort by lexicographic DESC on v: maximize min, then 2nd min, ...
    ranked = sorted(scores.keys(), key=lambda c: tuple(scores[c]), reverse=True)
    return ranked, scores


class GroupByFeatureGenerator(AbstractFeatureGenerator):
    """
    Groupby interaction features with flexible relative statistics.

    - Group stats are learned at fit time and reused at transform time.
    - pct_rank (if requested) is computed vs. the TRAINING distribution per group
      to avoid depending on other rows in the transformed dataset.
    """

    def __init__(
        self,
        target_type=None,
        aggregations=("mean", "pct_rank",),
        relative_to_aggs=("mean",),
        relative_ops=("ratio",),
        drop_basic_groupby_when_relative=True,
        fill_value="nan",
        eps=1e-8,
        return_dataframe=True,
        add_low_cardinality=None,
        max_features=500,
        random_state=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_type = target_type
        self.relative_to_aggs = relative_to_aggs
        self.relative_ops = relative_ops
        self.aggregations = aggregations
        self.max_features = max_features
        self.random_state = random_state

        self.drop_basic_groupby_when_relative = drop_basic_groupby_when_relative  # <-- NEW

        self.fill_value = np.nan if fill_value == "nan" else fill_value
        self.eps = eps
        self.return_dataframe = return_dataframe

        self.max_cardinality = add_low_cardinality if add_low_cardinality is not None else 2

        unknown = set(self.aggregations) - set(AGGREGATION_REGISTRY)
        if unknown:
            raise ValueError(f"Unknown aggregations: {unknown}")

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        raise ValueError("Input must be a pandas DataFrame")

    def _split_aggs(self):
        group_aggs, rowwise_aggs = [], []
        for name in self.aggregations:
            entry = AGGREGATION_REGISTRY[name]
            if entry["kind"] == "group":
                group_aggs.append(name)
            elif entry["kind"] == "rowwise":
                rowwise_aggs.append(name)
            else:
                raise ValueError(f"Unknown agg kind for {name}: {entry}")
        return group_aggs, rowwise_aggs

    def _relative_enabled(self) -> bool:
        return ("diff" in self.relative_ops) or ("ratio" in self.relative_ops)

    def _drop_basic(self) -> bool:
        # Only drop when explicitly requested AND we are actually producing relative features
        return bool(self.drop_basic_groupby_when_relative and self._relative_enabled() and self.relative_to_aggs)

    def _features_per_pair(self):
        """
        How many OUTPUT features per (cat, num) pair, for budgeting.
        """
        group_aggs, rowwise_aggs = self._split_aggs()

        base = 0
        if not self._drop_basic():
            base += len(group_aggs)
            base += int("pct_rank" in rowwise_aggs)

        rel = 0
        if self._relative_enabled():
            rel += len(self.relative_to_aggs) * (
                int("diff" in self.relative_ops) + int("ratio" in self.relative_ops)
            )

        return int(base + rel)

    # ----------------------------
    # FIT
    # ----------------------------
    def _fit(self, X, y=None):
        X = self._to_dataframe(X)

        # infer types
        self.categorical_features = X.columns[X.nunique() < self.max_cardinality].tolist()
        self.categorical_features += X.select_dtypes(include="category").columns.tolist()
        self.categorical_features = np.unique(self.categorical_features).tolist()

        self.numeric_features = [
            col for col in X.columns
            if col not in self.categorical_features and X[col].dtype not in ["category", "object"]
        ]

        if len(self.categorical_features) == 0 or len(self.numeric_features) == 0:
            self.group_stats_ = {}
            self.pct_rank_values_ = {}
            return self

        group_aggs, rowwise_aggs = self._split_aggs()

        ranked_cats, _ = rank_categoricals_by_small_counts(
            X,
            categorical_cols=self.categorical_features,
            min_count=20,          # pick what "stable" means for you
            top_k_smallest=10,     # tie-break depth
        )

        # keep your numeric ordering separate (no need to use nunique order at all)
        ranked_nums = X[self.numeric_features].nunique().sort_values(ascending=False).index.to_list()

        self.group_stats_ = {}
        self.pct_rank_values_ = {}
        self.global_stats_ = {num: float(X[num].mean()) for num in self.numeric_features}

        features_per_pair = self._features_per_pair()
        budget = self.max_features if self.max_features is not None else float("inf")
        used_features = 0

        for cat in ranked_cats:
            if cat not in self.categorical_features:
                continue

            for num in ranked_nums:
                if num not in self.numeric_features:
                    continue

                if used_features + features_per_pair > budget:
                    return self

                # group-level stats (still needed even if we won't output them,
                # because relative features reference them)
                if group_aggs:
                    named_aggs = {
                        name: pd.NamedAgg(column=num, aggfunc=AGGREGATION_REGISTRY[name]["agg"])
                        for name in group_aggs
                    }
                    stats = (
                        X.groupby(cat, observed=True)
                         .agg(**named_aggs)
                         .astype(float)
                    )
                else:
                    stats = pd.DataFrame(index=X[cat].dropna().unique())

                self.group_stats_[(cat, num)] = stats

                # pct_rank training distribution (per group) (only necessary if requested,
                # but harmless to keep as-is)
                if "pct_rank" in rowwise_aggs:
                    g = X[[cat, num]].dropna()
                    sorted_per_group = g.groupby(cat, observed=True)[num].apply(
                        lambda s: np.sort(s.to_numpy(dtype=float, copy=False))
                    )
                    self.pct_rank_values_[(cat, num)] = sorted_per_group

                used_features += features_per_pair

        return self

    # ----------------------------
    # TRANSFORM
    # ----------------------------
    def _transform(self, X):
        if len(getattr(self, "categorical_features", [])) == 0 or len(getattr(self, "numeric_features", [])) == 0:
            return X

        X = self._to_dataframe(X)
        features = []

        group_aggs, rowwise_aggs = self._split_aggs()
        drop_basic = self._drop_basic()

        for (cat, num), stats in self.group_stats_.items():
            x = X[num].astype(float)

            # map precomputed group stats (always computed to support relatives)
            mapped = {}
            for agg in stats.columns:
                mapped[agg] = X[cat].map(stats[agg]).astype(float)

                # Only OUTPUT the basic groupby features if not dropping
                if not drop_basic:
                    features.append(
                        mapped[agg].fillna(self.fill_value).rename(f"{num}__by__{cat}__{agg}")
                    )

            # pct_rank output only if requested AND not dropping basics
            if ("pct_rank" in rowwise_aggs):
                pct_feat = pd.Series(index=X.index, dtype=float)

                dist = self.pct_rank_values_.get((cat, num), None)
                if dist is None or len(dist) == 0:
                    pct_feat[:] = 0.5
                else:
                    cats = X[cat]
                    vals = x.to_numpy()

                    out = np.full(len(X), 0.5, dtype=float)
                    for i, (c, v) in enumerate(zip(cats.to_numpy(), vals)):
                        if pd.isna(c) or pd.isna(v):
                            continue
                        arr = dist.get(c, None)
                        if arr is None or arr.size == 0:
                            continue
                        rank = np.searchsorted(arr, v, side="right")
                        out[i] = rank / arr.size
                    pct_feat[:] = out

                features.append(pct_feat.rename(f"{num}__by__{cat}__pct_rank"))

            # validate relative aggs exist among computed group stats
            missing = set(self.relative_to_aggs) - set(mapped)
            if missing:
                raise ValueError(
                    f"Requested relative_to_aggs {missing} not present in computed "
                    f"group aggregations {list(mapped)} for pair ({cat}, {num}). "
                    f"Make sure those aggs are included in `aggregations=`."
                )

            # relative features
            if self._relative_enabled():
                for agg in self.relative_to_aggs:
                    ref = mapped[agg]
                    if "diff" in self.relative_ops:
                        features.append((x - ref).rename(f"{num}__minus__{cat}_{agg}"))
                    if "ratio" in self.relative_ops:
                        features.append((x / (ref + self.eps)).rename(f"{num}__ratio__{cat}_{agg}"))

        result = pd.concat(features, axis=1)
        return result if self.return_dataframe else result.values

    def _fit_transform(self, X, y):
        self._fit(X, y)
        return self._transform(X), dict()


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()
