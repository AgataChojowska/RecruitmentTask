import mlflow
import pandas as pd
import pickle

from helpers import get_dates_as_index, convert_to_time_series

if __name__ == "__main__":
    with mlflow.start_run(run_name="preprocess") as run:
        print("Preprocessing the data.")
        mlflow.set_tag("mlflow.runName", "preprocess")

        # Get raw data, narrow it down to C1 category only.
        new_data = pd.read_csv("data/raw/historical_sales_volume.csv")
        new_data = new_data[new_data["product"].isin(["P1", "P2", "P3"])]
        new_data = new_data.sort_values(by=['year', 'quarter'])
        new_data = new_data.reset_index(drop=True)

        # Turn data into series for the model.
        new_series = get_dates_as_index(data=new_data, date_col="dates", agg_col="volumeSales",
                                        year_col="year", quarter_col="quarter", to_period="Q")

        # Get data model was originally trained on.
        # I do that to be able to tune the model trained on both original and new data.
        with open('data/raw/arima_model.pkl', 'rb') as file:
            original_model = pickle.load(file)

            original_series = pd.Series(original_model.data.endog, name='volumeSales')

            # Original data isn't time-series, I am assuming quarterly intervals starting from Q1 2003.
            original_series = convert_to_time_series(series=original_series, start_date='2003-01-01', to_period="Q")

            # Combining original and new data. Now they both have datetime index.
            combined_series = pd.concat([original_series, new_series])
            df = combined_series.to_frame(name='volumeSales')
            df.to_csv(f"data/preprocessed/preprocessed_data.csv")




