import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from autogluon.features.generators.abstract import AbstractFeatureGenerator


class NeighborInteractionFeatureGenerator(AbstractFeatureGenerator):
    """
    Leak-safe neighbor-based interaction features with:
      - binary passthrough
      - scaled numeric features
      - OOF target-encoded categoricals
    """

    def __init__(
        self,
        target_type='regression',
        use_svd=True,
        n_components="auto",   
        svd_variance_target=0.90,  
        svd_max_components=256, # safety cap
        n_neighbors=(1, 5, 10, 25, 50, 100, 150, 200, 250),
        metrics=("cosine", "euclidean"),
        n_splits=5,
        weighted=False,
        shrinkage_m=0.0,
        include_distance=True,
        random_state=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_type = target_type
        self.use_svd = use_svd
        self.n_components = n_components
        self.svd_variance_target = svd_variance_target
        self.svd_max_components = svd_max_components
        self.n_neighbors = n_neighbors
        self.metrics = metrics
        self.n_splits = n_splits
        self.weighted = weighted
        self.shrinkage_m = shrinkage_m
        self.include_distance = include_distance
        self.random_state = random_state
        self.na_means = {}

    def _geometry_quality(self, Z, y):
        rng = np.random.RandomState(self.random_state)
        n = len(Z)

        sample_idx = rng.choice(n, size=min(1000, n), replace=False)
        k = min(self.n_neighbors)

        nn = NearestNeighbors(
            n_neighbors=k,
            metric=self.metrics[0]
        )
        nn.fit(Z)

        _, indices = nn.kneighbors(Z[sample_idx])

        y_knn = y[indices].mean(axis=1)

        return np.var(y_knn) / (np.var(y) + 1e-12)


    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------
    def _fit(self, X, y):
        from autogluon.features import OOFTargetEncodingFeatureGenerator

        X = X.copy()
        y = np.asarray(y)
        self.global_mean_ = float(np.mean(y))

        self.feature_names_in_ = list(X.columns)
        self.n_train_ = X.shape[0]
        self.y_train_ = y
        self.train_index_ = X.index  # NEW: robust train/test detection
        self.train_index_set_ = set(self.train_index_)
        self.max_k_ = max(self.n_neighbors)  # NEW: cache max_k

        # ===============================
        # feature type detection
        # ===============================
        self.binary_cols_ = [
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c]) and X[c].dropna().isin([0, 1]).all()
        ]

        self.numeric_cols_ = [
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c]) and c not in self.binary_cols_
        ]

        self.categorical_cols_ = [
            c for c in X.columns
            if not pd.api.types.is_numeric_dtype(X[c])
        ]

        # ===============================
        # categorical OOF encoding
        # ===============================
        if self.categorical_cols_:
            self.cat_encoder_ = OOFTargetEncodingFeatureGenerator(
                self.target_type,
                n_splits=self.n_splits,
                random_state=self.random_state
            )
            X_cat = self.cat_encoder_.fit_transform(X[self.categorical_cols_], y)
        else:
            X_cat = pd.DataFrame(index=X.index)

        # ===============================
        # scale numeric non-binary
        # ===============================
        if self.numeric_cols_:
            self.scaler_ = StandardScaler()
            X_num = pd.DataFrame(
                self.scaler_.fit_transform(X[self.numeric_cols_]),
                columns=self.numeric_cols_,
                index=X.index
            )
        else:
            X_num = pd.DataFrame(index=X.index)

        # binary passthrough
        X_bin = X[self.binary_cols_] if self.binary_cols_ else pd.DataFrame(index=X.index)

        # final processed matrix
        X_proc = pd.concat([X_bin, X_num, X_cat], axis=1)
        for col in X_proc.columns:
            self.na_means[col] = X_proc[col].mean()
            if X_proc[col].isna().any():
                X_proc[col] = X_proc[col].fillna(self.na_means[col])
        self.proc_columns_ = X_proc.columns.tolist()
        X_np = X_proc.values

        # latent or raw space
        if self.use_svd:
            max_components = min(
                self.svd_max_components,
                X_np.shape[1] - 1,
                X_np.shape[0] - 1,
            )

            svd_full = TruncatedSVD(
                n_components=max_components,
                random_state=self.random_state
            )
            Z_full = svd_full.fit_transform(X_np)

            cum_var = np.cumsum(svd_full.explained_variance_ratio_)
            k = np.searchsorted(cum_var, self.svd_variance_target) + 1

            self.n_components_ = int(k)

            self.svd_ = TruncatedSVD(
                n_components=self.n_components_,
                random_state=self.random_state
            )
            Z = self.svd_.fit_transform(X_np)
        else:
            Z = X_np

        # geometry gating
        self.geometry_score_ = self._geometry_quality(Z, y)
        self.geometry_enabled_ = self.geometry_score_ > 0.01
        if not self.geometry_enabled_:
            return self

        self.Z_train_ = Z

        # NEW: cache fitted NearestNeighbors models on the full training reference set
        self.nn_models_ = {}
        for metric in self.metrics:
            nn = NearestNeighbors(n_neighbors=self.max_k_, metric=metric)
            nn.fit(self.Z_train_)
            self.nn_models_[metric] = nn

        # OOF feature generation (unchanged)
        self.nn_feature_names_ = self._make_nn_feature_names()
        self.oof_nn_features_ = np.zeros((self.n_train_, len(self.nn_feature_names_)))

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        col_offset = 0
        for metric in self.metrics:
            for tr_idx, val_idx in kf.split(Z):
                nn = NearestNeighbors(
                    n_neighbors=self.max_k_,
                    metric=metric
                )
                nn.fit(Z[tr_idx])

                distances, indices = nn.kneighbors(Z[val_idx])

                self._fill_nn_features(
                    self.oof_nn_features_,
                    val_idx,
                    distances,
                    indices,
                    y[tr_idx],
                    col_offset
                )

            col_offset += self._features_per_metric()

        return self

    def _fit_transform(self, X, y):
        self._fit(X, y)
        return self._transform(X), dict()

    def _transform(self, X):
        if not getattr(self, "geometry_enabled_", True):
            return X

        X = X.copy()
        is_train = X.index.map(self.train_index_set_.__contains__).all()

        # categorical
        if self.categorical_cols_:
            X_cat = self.cat_encoder_.transform(X[self.categorical_cols_])
        else:
            X_cat = pd.DataFrame(index=X.index)

        # numeric
        if self.numeric_cols_:
            X_num = pd.DataFrame(
                self.scaler_.transform(X[self.numeric_cols_]),
                columns=self.numeric_cols_,
                index=X.index
            )
        else:
            X_num = pd.DataFrame(index=X.index)

        # binary
        X_bin = X[self.binary_cols_] if self.binary_cols_ else pd.DataFrame(index=X.index)

        X_proc = pd.concat([X_bin, X_num, X_cat], axis=1)
        for col in X_proc.columns:
            if X_proc[col].isna().any():
                X_proc[col] = X_proc[col].fillna(self.na_means[col])

        if self.use_svd:
            Z = self.svd_.transform(X_proc.values)
        else:
            Z = X_proc.values

        nn_features = np.zeros((X.shape[0], len(self.nn_feature_names_)))

        col_offset = 0
        for metric in self.metrics:
            if is_train:
                # training → reuse OOF features (no kNN query needed)
                nn_features[:, col_offset:col_offset + self._features_per_metric()] = \
                    self.oof_nn_features_[:, col_offset:col_offset + self._features_per_metric()]
            else:
                # test/unseen → query cached NN model (no refit)
                nn = self.nn_models_[metric]
                distances, indices = nn.kneighbors(Z)

                self._fill_nn_features(
                    nn_features,
                    slice(None),
                    distances,
                    indices,
                    self.y_train_,
                    col_offset
                )

            col_offset += self._features_per_metric()

        nn_df = pd.DataFrame(
            nn_features,
            columns=self.nn_feature_names_,
            index=X.index
        )

        return pd.concat([X, nn_df], axis=1)


    # ------------------------------------------------------------------
    # INTERNAL HELPERS (UNCHANGED)
    # ------------------------------------------------------------------
    def _fill_nn_features(self, out, rows, distances, indices, y_ref, col_offset):
        col = col_offset
        for k in self.n_neighbors:
            knn_idx = indices[:, :k]
            knn_dist = distances[:, :k]

            # ---- SHRUNK mean target ----
            mean_knn = y_ref[knn_idx].mean(axis=1)
            alpha = k / (k + self.shrinkage_m)
            out[rows, col] = (
                alpha * mean_knn +
                (1.0 - alpha) * self.global_mean_
            )
            col += 1

            # kth distance
            if self.include_distance:
                out[rows, col] = knn_dist[:, -1]
            col += 1

            if self.weighted:
                w = 1.0 / (knn_dist + 1e-6)
                w_sum = np.sum(w, axis=1)

                weighted_mean = np.sum(w * y_ref[knn_idx], axis=1) / w_sum
                out[rows, col] = (
                    alpha * weighted_mean +
                    (1.0 - alpha) * self.global_mean_
                )
                col += 1

    def _features_per_metric(self):
        return len(self.n_neighbors) * (2 + int(self.weighted))

    def _make_nn_feature_names(self):
        names = []
        for metric in self.metrics:
            for k in self.n_neighbors:
                names.append(f"nn_{metric}_mean_target_k{k}")
                names.append(f"nn_{metric}_kth_distance_k{k}")
                if self.weighted:
                    names.append(f"nn_{metric}_weighted_target_k{k}")
        return names

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

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

from autogluon.features.generators.abstract import AbstractFeatureGenerator


class NeighborStructureFeatureGenerator(AbstractFeatureGenerator):
    """
    Leak-free neighborhood structure features:
      - kNN distance statistics
      - local geometric spread
      - prototype (cluster centroid) distances
    """

    def __init__(
        self,
        target_type='regression',
        use_svd=True,
        svd_variance_target=0.90,
        svd_max_components=256,
        n_neighbors=(5, 10, 25, 50),
        metrics=("cosine",),
        n_clusters=64,
        random_state=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_svd = use_svd
        self.svd_variance_target = svd_variance_target
        self.svd_max_components = svd_max_components
        self.n_neighbors = n_neighbors
        self.metrics = metrics
        self.n_clusters = n_clusters
        self.random_state = random_state

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------
    def _fit(self, X, y=None):
        X = X.copy()

        # ---------- numeric preprocessing ----------
        self.numeric_cols_ = [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]

        self.scaler_ = StandardScaler()
        X_num = self.scaler_.fit_transform(X[self.numeric_cols_])

        # ---------- optional SVD ----------
        if self.use_svd:
            max_components = min(
                self.svd_max_components,
                X_num.shape[1] - 1,
                X_num.shape[0] - 1,
            )

            svd_full = TruncatedSVD(
                n_components=max_components,
                random_state=self.random_state,
            )
            Z_full = svd_full.fit_transform(X_num)
            cum_var = np.cumsum(svd_full.explained_variance_ratio_)
            k = np.searchsorted(cum_var, self.svd_variance_target) + 1

            self.svd_ = TruncatedSVD(
                n_components=int(k),
                random_state=self.random_state,
            )
            Z = self.svd_.fit_transform(X_num)
        else:
            Z = X_num

        self.Z_train_ = Z
        self.max_k_ = max(self.n_neighbors)

        # ---------- kNN models ----------
        self.nn_models_ = {}
        for metric in self.metrics:
            nn = NearestNeighbors(
                n_neighbors=self.max_k_,
                metric=metric
            )
            nn.fit(Z)
            self.nn_models_[metric] = nn

        # ---------- prototypes (clusters) ----------
        self.kmeans_ = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=1024,
        )
        self.kmeans_.fit(Z)

        self.feature_names_ = self._make_feature_names()
        return self

    def _fit_transform(self, X, y=None):
        self._fit(X, y)
        return self._transform(X), dict()

    # ------------------------------------------------------------------
    # TRANSFORM
    # ------------------------------------------------------------------
    def _transform(self, X):
        X = X.copy()

        X_num = self.scaler_.transform(X[self.numeric_cols_])
        Z = self.svd_.transform(X_num) if self.use_svd else X_num

        features = np.zeros((X.shape[0], len(self.feature_names_)))
        col = 0

        # ---------- kNN geometry ----------
        for metric in self.metrics:
            nn = self.nn_models_[metric]
            distances, indices = nn.kneighbors(Z)

            for k in self.n_neighbors:
                d_k = distances[:, :k]

                # kth neighbor distance
                features[:, col] = d_k[:, -1]
                col += 1

                # mean distance
                features[:, col] = d_k.mean(axis=1)
                col += 1

                # local spread (std in latent space)
                spread = np.std(
                    self.Z_train_[indices[:, :k]],
                    axis=1
                ).mean(axis=1)
                features[:, col] = spread
                col += 1

        # ---------- prototype distances ----------
        centroids = self.kmeans_.cluster_centers_
        for i in range(self.n_clusters):
            diff = Z - centroids[i]
            features[:, col] = np.linalg.norm(diff, axis=1)
            col += 1

        return pd.concat(
            [
                X,
                pd.DataFrame(
                    features,
                    columns=self.feature_names_,
                    index=X.index,
                ),
            ],
            axis=1,
        )

    # ------------------------------------------------------------------
    # FEATURE NAMES
    # ------------------------------------------------------------------
    def _make_feature_names(self):
        names = []
        for metric in self.metrics:
            for k in self.n_neighbors:
                names.append(f"nn_{metric}_k{k}_kth_dist")
                names.append(f"nn_{metric}_k{k}_mean_dist")
                names.append(f"nn_{metric}_k{k}_local_spread")

        for i in range(self.n_clusters):
            names.append(f"proto_dist_{i}")

        return names


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