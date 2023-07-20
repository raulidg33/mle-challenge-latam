# ============================================================================================
# libraries
# ============================================================================================
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from typing import Tuple, Union, List
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
# ================================================================================================
# DelayModel class
# ================================================================================================
class DelayModel:
    # ============================================================================================
    # init method
    # ============================================================================================
    def __init__(
        self,
        from_file: bool = False
    ):
        self.__selected_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        self.__threshold_in_minutes = 15
        self.__scale_pos_weight = 4.4402380952380955
        self.__learning_rate = 0.01
        self.__random_state = 1
        if from_file:
            self._model = joblib.load('xgbc.sav')
        else:
            self._model = xgb.XGBClassifier(
                random_state=self.__random_state,
                learning_rate=self.__learning_rate, 
                scale_pos_weight=self.__scale_pos_weight
            )
    # ============================================================================================
    # public methods
    # ============================================================================================
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix = 'MES')],
            axis = 1
        )[self.__selected_features]
        if target_column:
            data['min_diff'] = data.apply(self.__get_min_diff, axis = 1)
            data['delay'] = np.where(data['min_diff'] > self.__threshold_in_minutes, 1, 0)
            return (features, pd.DataFrame({target_column: data[target_column]}))
        return features
    # -------------------------------------------------------------------------------
    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ):
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._model.fit(features, target)
        joblib.dump(self._model, "xgbc.sav")
    # -------------------------------------------------------------------------------
    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        y_pred = self._model.predict(features).tolist()
        return y_pred
    # ============================================================================================
    # private methods
    # ============================================================================================
    def __get_min_diff(
            self,
            data: pd.DataFrame
    ) -> float:
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff