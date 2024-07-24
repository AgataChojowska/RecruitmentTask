import mlflow
import pickle
import argparse


if __name__ == "__main__":
    with mlflow.start_run(run_name="train") as run:
        parser = argparse.ArgumentParser()
        parser.add_argument("--predict_next_quarters", type=int, default=3)
        args = parser.parse_args()

        print(f"Getting predictions for the next {args.predict_next_quarters} quarters.")

        with open("predict/models/model.pkl", 'rb') as file:
            model = pickle.load(file)

            predictions = model.get_forecast(steps=args.predict_next_quarters).predicted_mean
            predictions.to_csv("data/predicted/predictions.csv")
            print(predictions)

