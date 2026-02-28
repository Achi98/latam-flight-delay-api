import pandas as pd
import numpy as np

from typing import Tuple, Union, List
from datetime import datetime
import xgboost as xgb


class DelayModel:

    def __init__(self):
        self._model = None  # El modelo debe guardarse en este atributo.

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
        df = data.copy()

        def get_period_day(date_str: str) -> str:
            date_time = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").time()
            morning_min = datetime.strptime("05:00", "%H:%M").time()
            morning_max = datetime.strptime("11:59", "%H:%M").time()
            afternoon_min = datetime.strptime("12:00", "%H:%M").time()
            afternoon_max = datetime.strptime("18:59", "%H:%M").time()
            evening_min = datetime.strptime("19:00", "%H:%M").time()
            evening_max = datetime.strptime("23:59", "%H:%M").time()
            night_min = datetime.strptime("00:00", "%H:%M").time()
            night_max = datetime.strptime("04:59", "%H:%M").time()

            if morning_min <= date_time <= morning_max:
                return "morning"
            if afternoon_min <= date_time <= afternoon_max:
                return "afternoon"
            # noche: cubre desde 19:00 hasta 04:59 (cruza medianoche)
            return "night"

        def is_high_season(fecha: str) -> int:
            fecha_año = int(fecha.split("-")[0])
            fecha_dt = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
            range1_min = datetime.strptime("15-Dec", "%d-%b").replace(year=fecha_año)
            range1_max = datetime.strptime("31-Dec", "%d-%b").replace(year=fecha_año)
            range2_min = datetime.strptime("1-Jan", "%d-%b").replace(year=fecha_año)
            range2_max = datetime.strptime("3-Mar", "%d-%b").replace(year=fecha_año)
            range3_min = datetime.strptime("15-Jul", "%d-%b").replace(year=fecha_año)
            range3_max = datetime.strptime("31-Jul", "%d-%b").replace(year=fecha_año)
            range4_min = datetime.strptime("11-Sep", "%d-%b").replace(year=fecha_año)
            range4_max = datetime.strptime("30-Sep", "%d-%b").replace(year=fecha_año)

            if ((range1_min <= fecha_dt <= range1_max)
                    or (range2_min <= fecha_dt <= range2_max)
                    or (range3_min <= fecha_dt <= range3_max)
                    or (range4_min <= fecha_dt <= range4_max)):
                return 1
            return 0

        def get_min_diff(row: pd.Series) -> float:
            fecha_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
            fecha_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
            return (fecha_o - fecha_i).total_seconds() / 60.0

        # Ingeniería de features
        df["period_day"] = df["Fecha-I"].apply(get_period_day)
        df["high_season"] = df["Fecha-I"].apply(is_high_season)
        df["min_diff"] = df.apply(get_min_diff, axis=1)

        # Crear el target si no existe (umbral de 15 minutos)
        if "delay" not in df.columns:
            threshold_in_minutes = 15
            df["delay"] = np.where(df["min_diff"] > threshold_in_minutes, 1, 0)

        # Target
        target = None
        if target_column is not None and target_column in df.columns:
            target = df[[target_column]]

        # One-hot encoding para variables categóricas usadas en el notebook
        opera_dummies = pd.get_dummies(df["OPERA"], prefix="OPERA")
        tipovuelo_dummies = pd.get_dummies(df["TIPOVUELO"], prefix="TIPOVUELO")
        mes_dummies = pd.get_dummies(df["MES"], prefix="MES")

        features = pd.concat([opera_dummies, tipovuelo_dummies, mes_dummies], axis=1)

        # Asegurar que existan las top 10 features del notebook
        top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]

        for col in top_10_features:
            if col not in features.columns:
                features[col] = 0

        features = features[top_10_features].copy()

        if target is not None:
            return features, target
        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Asegurar array 1D para y
        y = target.squeeze()

        # Factor de balanceo
        n_y0 = int((y == 0).sum())
        n_y1 = int((y == 1).sum())
        scale = float(n_y0 / n_y1) if n_y1 != 0 else 1.0

        model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=scale,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(features, y)
        self._model = model

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            # Fallback: si no hay modelo entrenado, retornar ceros
            n = int(features.shape[0])
            return [0 for _ in range(n)]

        preds = self._model.predict(features)
        return [int(p) for p in preds]