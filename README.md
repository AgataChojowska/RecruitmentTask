## Recruitment task

### Running the code

Install requirements.
To run tests, make sure you installed ``pytest pytest-mock``.

To run the pipeline, first run the command to start mlflow tracking uri.

``mlflow ui --host 0.0.0.0 --port 5000``
#### Data summaries



This part of the task is under data_summary.

Run main in ``data_summary/main.py`` to get data summaries.
Result files are saved under ``results/sales_diff`` and ``results/yearly_rank``.
Both result files are already there for easy access.


#### ML Model & Pipeline

This part of the task is under ``model_refinement``.

My initial analysis in Jupyter notebook.

Run main in ``mlflow_project/main.py`` to run the pipeline.

Preprocessed data will be saved under ``mlflow_project/data/preprocessed``.

Predictions will be printed out and saved under ``mlflow_project/data/predicted``.

Model artifacts will be saved in ``mlruns`` folder that's created by mlflow.
Logs will be saved in ``mlflow_project/logs`` folder.


### Comments
1. #### Data summaries

    I decided to go with SQLite3 instead of Pyspark as the dataset is only around 7 KB.

2. #### ML Model & Pipeline

    I started with exploratory analysis in ``model_refinement/exploratory_analysis.ipynb`` notebook.
    
    I decided to go with preprocessing of the data as SARIMAX model is intended for time-series, and business 
    problem requires quarterly sales predictions. My assumption is that the original data is quarterly intervals starting in Q1 2003.
    
    I decided not to append new training data to the model, but combine the original and new data.
    That approach lets me find parameters while retraining on all available data.
    
    I created a pipeline with mlflow that has 3 steps: 

        1. Preprocesing: of original and new data, then combining it.
        2. Training & tuning: using rolling windows cross-validation.
        3. Predicting: for chosen number of quarters.
    The model doesn't have good performance, but mindful of time, I focused on building reusable system that allows for continuous retraining and tuning of the model.



