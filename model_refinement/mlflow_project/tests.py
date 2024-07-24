import pandas as pd
from unittest.mock import MagicMock
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from train.model_tuner import SARIMAXTuner
from preprocess.helpers import quarter_to_date


def test_fit_best_model_with_params(mocker):
    mock_sarimax = mocker.patch('train.model_tuner.SARIMAX')
    mock_model_instance = mock_sarimax.return_value
    mock_fit = mock_model_instance.fit
    mock_results = MagicMock(spec=SARIMAXResults)
    mock_fit.return_value = mock_results
    mock_results.summary.return_value = 'Model summary'
    mock_results.mle_retvals = {'convergence': True}

    tuner = SARIMAXTuner(data=MagicMock(), n_splits=3)
    tuner.best_params = ((1, 1, 1), (1, 1, 1, 4))
    result = tuner.fit_best_model()

    mock_sarimax.assert_called_once_with(tuner.data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    assert result == mock_results
    assert result.summary() == 'Model summary'
    assert 'convergence' in result.mle_retvals


def test_fit_best_model_without_params(capfd):
    tuner = SARIMAXTuner(data=MagicMock(), n_splits=3)
    tuner.best_params = None
    result = tuner.fit_best_model()

    assert result is None
    out, err = capfd.readouterr()
    assert "No best parameters found. Provide them or run grid_search() first." in out


def test_quarter_to_date():
    year = 2020
    quarter = 2
    expected_date = pd.Timestamp('2020-04-01')
    assert quarter_to_date(year, quarter) == expected_date



