import logging
import traceback
import warnings

import mlflow
import click


def _run(entrypoint, parameters={}):
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return submitted_run


@click.command()
def workflow():
    with mlflow.start_run(run_name="sales_pipeline"):
        mlflow.set_tag("mlflow.runName", "sales_pipeline")
        _run("preprocess")
        _run("train", {"n_splits": 3})
        _run("predict",  {"predict_next_quarters": 5})


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(
        level=logging.INFO,
        filename="logs/pipeline.log",
        filemode="a",
        format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        workflow()
    except Exception as e:
        print("Exception occurred. Check logs.")
        logger.error(f"Failed to run workflow due to error:\n{e}")
        logger.error(traceback.format_exc())
