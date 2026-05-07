from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.util import hash_pandas_object

from autogluon.common.features.feature_metadata import FeatureMetadata

from .abstract import AbstractFeatureGenerator

_UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_UINT64_DENOMINATOR = float(1 << 64)
_SPLITMIX64_INCREMENT = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX64_MULTIPLIER_1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX64_MULTIPLIER_2 = np.uint64(0x94D049BB133111EB)


class RandomNoiseFeatureGenerator(AbstractFeatureGenerator):
    """
    Appends a fixed number of deterministic random noise features.

    The generated columns are derived from the row contents and ``random_state`` so repeated
    transforms of the same rows produce the same noise values.

    Parameters
    ----------
    num_noise_features : int
        Number of random noise features to generate.
    noise_prefix : str, default "__noise__."
        Prefix used when naming generated features.
    keep_original : bool, default True
        Whether to keep the original input features alongside the generated noise features.
    **kwargs
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """

    def __init__(
        self,
        num_noise_features: int = 1,
        noise_prefix: str = "__noise__.",
        keep_original: bool = True,
        **kwargs,
    ):
        # if isinstance(num_noise_features, bool) or not isinstance(num_noise_features, Integral):
        #     raise TypeError(
        #         f"num_noise_features must be an integer, but received {num_noise_features!r} "
        #         f"of type {type(num_noise_features).__name__}."
        #     )
        # if num_noise_features < 0:
        #     raise ValueError(f"num_noise_features must be >= 0, but received {num_noise_features}.")
        # if not isinstance(noise_prefix, str):
        #     raise TypeError(
        #         f"noise_prefix must be a string, but received {noise_prefix!r} "
        #         f"of type {type(noise_prefix).__name__}."
        #     )

        # passthrough = kwargs.pop("passthrough", keep_original)
        # if passthrough != keep_original:
        #     raise ValueError(
        #         f"keep_original={keep_original} conflicts with passthrough={passthrough}. "
        #         "Specify only keep_original when using RandomNoiseFeatureGenerator."
        #     )

        self.num_noise_features = int(num_noise_features)
        self.noise_prefix = noise_prefix
        self.keep_original = keep_original
        self._noise_feature_names: list[str] = []
        super().__init__(**kwargs)

    def estimate_no_of_new_features(self, X: pd.DataFrame, **kwargs) -> int:
        return self.num_noise_features

    def _fit_transform(self, X: DataFrame, **kwargs) -> tuple[DataFrame, dict]:
        self._noise_feature_names = self._infer_noise_feature_names(existing_features=list(X.columns))
        X_out = self._transform(X)
        return X_out, dict()

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._generate_noise_features(X=X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    # def is_valid_metadata_in(self, feature_metadata_in: FeatureMetadata):
    #     return self.num_noise_features > 0 or super().is_valid_metadata_in(feature_metadata_in)

    # def _more_tags(self):
    #     return {"feature_interactions": True}

    def _infer_noise_feature_names(self, existing_features: list[str]) -> list[str]:
        feature_names = []
        existing_features = set(existing_features)
        feature_index = 0
        while len(feature_names) < self.num_noise_features:
            feature_name = f"{self.noise_prefix}{feature_index}"
            if feature_name not in existing_features:
                feature_names.append(feature_name)
                existing_features.add(feature_name)
            feature_index += 1
        return feature_names

    def _generate_noise_features(self, X: DataFrame) -> DataFrame:
        if self.num_noise_features == 0:
            return DataFrame(index=X.index)

        row_hash = self._get_row_hash(X=X)
        noise_data = {}
        random_state = 0 if self.random_state is None else int(self.random_state)
        seed = random_state % (1 << 64)

        for feature_idx, feature_name in enumerate(self._noise_feature_names):
            feature_seed = np.uint64((seed + feature_idx * int(_SPLITMIX64_INCREMENT)) % (1 << 64))
            mixed_hash = self._splitmix64(row_hash ^ feature_seed)
            noise = (mixed_hash.astype(np.float64) / _UINT64_DENOMINATOR - 0.5).astype(np.float32)
            noise_data[feature_name] = noise

        return DataFrame(noise_data, index=X.index)

    @staticmethod
    def _get_row_hash(X: DataFrame) -> np.ndarray:
        if len(X.index) == 0:
            return np.array([], dtype=np.uint64)
        if len(X.columns) == 0:
            return hash_pandas_object(X.index.to_series(), index=False).to_numpy(dtype=np.uint64, copy=False)
        return hash_pandas_object(X, index=False).to_numpy(dtype=np.uint64, copy=False)

    @staticmethod
    def _splitmix64(values: np.ndarray) -> np.ndarray:
        z = (values + _SPLITMIX64_INCREMENT) & _UINT64_MASK
        z = ((z ^ (z >> np.uint64(30))) * _SPLITMIX64_MULTIPLIER_1) & _UINT64_MASK
        z = ((z ^ (z >> np.uint64(27))) * _SPLITMIX64_MULTIPLIER_2) & _UINT64_MASK
        return z ^ (z >> np.uint64(31))
