import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from itertools import combinations


from ..abstract import AbstractFeatureGenerator
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

from .linear_init import OOFCustomLinearModel

class LinearFeatureGenerator(AbstractFeatureGenerator):
    def __init__(
            self, 
            target_type: str, 
            random_state=None, 
            lin_kwargs=None, 
            **kwargs):
        super().__init__(**kwargs)
        if lin_kwargs is None:
            lin_kwargs = {'cat_method': 'oof-te'}
        self.target_type = target_type
        self.linear_model = OOFCustomLinearModel(target_type=target_type, random_state=random_state, **lin_kwargs)

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.linear_model.fit(X, y)
        return self._transform(X, is_train=True), dict()
    
    def _transform(self, X: pd.DataFrame, is_train=False) -> pd.DataFrame:
        X_transformed = X.copy()
        lin_preds = self.linear_model.predict(X, is_train=is_train)
        if self.target_type in ['binary', 'regression']:
            X_transformed[f'linear_score'] = lin_preds
        else:
            for i in range(lin_preds.shape[1]):
                if isinstance(lin_preds, np.ndarray):
                    X_transformed[f'linear_score_{i}'] = lin_preds[:, i]
                elif isinstance(lin_preds, pd.DataFrame):
                    X_transformed[f'linear_score_{i}'] = lin_preds.iloc[:, i]
                else:
                    raise ValueError(f"lin_preds is of unsupported type: {type(lin_preds)}")
        return X_transformed
    
    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()
    
