import os
import unittest
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from challenge.model import DelayModel

class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel()
        # made sure it can find the data.csv no matter where the terminal is
        dirpath = os.path.dirname(os.path.realpath(__file__))
        data_filepath = os.path.join(dirpath, '../../data/data.csv')
        self.data = pd.read_csv(filepath_or_buffer=data_filepath, low_memory=False)
    def test_is_fitted(
            self
    ):
        curpath = os.path.dirname(os.path.realpath(__file__))
        savpath = os.path.join(curpath, '../../challenge/xgbc.sav')
        if os.path.exists(savpath):
            os.remove(savpath)
        self.model = DelayModel()
        assert self.model.is_fitted == False
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )
        self.model.fit(
            features=features,
            target=target
        )
        assert self.model.is_fitted == True

    def test_model_preprocess_for_training(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        assert isinstance(features, pd.DataFrame)
        # added all() because features.columns is pandas.core.indexes.base.Index
        assert all(features.columns == [
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
        ])
        assert isinstance(target, pd.DataFrame)
        assert target.columns == [
            "delay"
        ]

    def test_model_preprocess_for_serving(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        assert isinstance(features, pd.DataFrame)
        # added all() because features.columns is pandas.core.indexes.base.Index
        assert all(features.columns == [
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
        ])

    def test_model_fit(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        _, features_validation, _, target_validation = train_test_split(features, target, test_size = 0.33, random_state = 42)

        self.model.fit(
            features=features,
            target=target
        )

        predicted_target = self.model._model.predict(
            features_validation
        )

        report = classification_report(target_validation, predicted_target, output_dict=True)
        
        assert report["0"]["recall"] < 0.60
        assert report["0"]["f1-score"] < 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30

    def test_model_predict(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        predicted_targets = self.model.predict(
            features=features
        )

        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features.shape[0]
        assert all(isinstance(predicted_target, int) for predicted_target in predicted_targets)