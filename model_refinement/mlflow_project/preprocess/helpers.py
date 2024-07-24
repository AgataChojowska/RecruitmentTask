from typing import Literal
import pandas as pd


def quarter_to_date(year: int, quarter: int) -> pd.Timestamp:
    month = {1: '01', 2: '04', 3: '07', 4: '10'}[quarter]
    return pd.to_datetime(f'{year}-{month}-01')


def get_dates_as_index(data: pd.DataFrame, date_col: str, agg_col: str, year_col: str, quarter_col: str,
                       to_period: Literal['Q', 'M']) -> pd.Series:
    data[date_col] = data.apply(lambda row: quarter_to_date(int(row[year_col]), row[quarter_col]), axis=1)
    data = data.groupby(date_col)[agg_col].sum()
    data.index = pd.DatetimeIndex(data.index).to_period(to_period)
    return data


def convert_to_time_series(series: pd.Series, start_date: str, to_period: str) -> pd.Series:
    dates = pd.date_range(start=start_date, periods=len(series), freq=to_period)
    series = pd.Series(series.values, index=dates)
    series.index = pd.DatetimeIndex(series.index).to_period(to_period)
    return series
