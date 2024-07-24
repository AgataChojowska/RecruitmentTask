import warnings
import argparse
from model_tuner import SARIMAXTuner
import mlflow
import pickle
import pandas as pd
from mlflow.models.signature import infer_signature

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    with mlflow.start_run(run_name="train") as run:
        print("Training & tuning the model.")

        parser = argparse.ArgumentParser()
        parser.add_argument("--n_splits", type=int, default=3)
        args = parser.parse_args()

        mlflow.set_tag("mlflow.runName", "train")
        data = pd.read_csv("data/preprocessed/preprocessed_data.csv", index_col=0)
        data.index = pd.DatetimeIndex(data.index).to_period("Q-DEC")

        # Tuning and training the model.
        tuner = SARIMAXTuner(data=data["volumeSales"], n_splits=args.n_splits)
        tuner.grid_search()
        best_model_results = tuner.fit_best_model()

        # Logging model parameters.
        run_parameters = tuner.get_best_params_dict()
        mlflow.log_params(run_parameters)

        # Logging model metrics.
        metrics = tuner.get_best_mean_mse_dict()
        mlflow.log_metrics(metrics)

        signature = infer_signature(
            model_input=data,
            model_output=best_model_results.get_forecast().predicted_mean
        )

        # Logging model artifacts.
        mlflow.sklearn.log_model(best_model_results, "model", signature=signature)

        # Saving the model to use for inferring in the next pipeline step.
        with open(f"predict/models/model.pkl", 'wb') as file:
            pickle.dump(best_model_results, file)


