import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from typing import List, Optional, Tuple, Dict


class SARIMAXTuner:
    """ A cross-validation class for tuning of SARIMAX model parameters. """

    def __init__(self,
                 data: pd.Series,
                 n_splits: int,
                 p: Optional[List[int]] = None,
                 d: Optional[List[int]] = None,
                 q: Optional[List[int]] = None,
                 P: Optional[List[int]] = None,
                 D: Optional[List[int]] = None,
                 Q: Optional[List[int]] = None,
                 s: Optional[int] = 4, ):
        self.data = data
        self.p = p if p is not None else [0, 1, 2]
        self.d = d if d is not None else [0, 1]
        self.q = q if q is not None else [0, 1, 2]
        self.P = P if P is not None else [0, 1]
        self.D = D if D is not None else [0, 1]
        self.Q = Q if Q is not None else [0, 1]
        self.s = s
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1.")
        self.n_splits = n_splits
        self.best_mse = np.inf
        self.best_params: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]] = None

    def rolling_window_cv(self, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int]) -> float:
        mse_scores = []
        train_size = int(len(self.data) / (self.n_splits + 1))
        for i in range(self.n_splits):
            train_end = train_size * (i + 1)
            if train_end >= len(self.data):
                break
            train_data = self.data[:train_end]
            test_data = self.data[train_end:train_end + train_size]

            if len(test_data) == 0:
                break

            try:
                model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
                fitted_model = model.fit(disp=False)
                predictions = fitted_model.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
                mse = mean_squared_error(test_data, predictions)
                mse_scores.append(mse)
            except Exception as e:
                print(f"Error with parameters {order} and {seasonal_order} in rolling window {i}: {e}")
                continue

        if mse_scores:
            return np.mean(mse_scores)
        return np.inf

    def grid_search(self) -> None:
        for param in product(self.p, self.d, self.q, self.P, self.D, self.Q):
            order = param[:3]
            seasonal_order = param[3:] + (self.s,)
            try:
                mean_mse = self.rolling_window_cv(order, seasonal_order)
                if mean_mse < self.best_mse:
                    self.best_mse = mean_mse
                    self.best_params = (order, seasonal_order)
            except Exception as e:
                print(f"Error with parameters {param}: {e}")

        print("Best SARIMAX parameters:", self.best_params)

    def get_best_params_dict(self) -> Dict:
        if self.best_params:
            best_order, best_seasonal_order = self.best_params
            return {"order": best_order, "seasonal_order": best_seasonal_order}
        else:
            print("No best parameters found. Please run the train method first.")

    def get_best_mean_mse_dict(self) -> Dict:
        if self.best_mse:
            return {"MSE": self.best_mse}
        else:
            print("No best mean MSE found. Please run the train method first.")

    def fit_best_model(self) -> Optional[SARIMAXResults]:
        if self.best_params:
            best_order, best_seasonal_order = self.best_params
            best_model = SARIMAX(self.data, order=best_order, seasonal_order=best_seasonal_order)
            best_results = best_model.fit(disp=False)
            print(best_results.summary())
            print(best_results.mle_retvals)
            return best_results
        else:
            print("No best parameters found. Provide them or run grid_search() first.")
            return None
