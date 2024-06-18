import argparse
import importlib
from datetime import datetime

import joblib
import numpy as np
from flask import jsonify, request, g, Flask
from flask_apscheduler import APScheduler

from pipelines.BasePipeline import Pipeline
from database.tables import Base
from utilities.databes_utilities import get_db, get_local_session, save_request_response
from utilities.utilities import *

FILE_PATH = Path(__file__).resolve()
ROOT = Path(*FILE_PATH.parts[:-1])
DATA_PATH, DATABASE_PATH = ROOT / 'data', ROOT / 'database'

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--conf_path", type=str, default="configuration.yaml")
parsed_args = parser.parse_args()

configuration = read_yaml_file(parsed_args.conf_path)
web_service_configuration = configuration.pop("web_service", {})

# Sample data
MODELS_PIPELINE = {"linear": "pipelines.LinearRegressionPipeline",
                   "log_linear": "pipelines.LinearRegressionPipeline",
                   "xgb": "pipelines.XGBRegressionPipeline"}

app = Flask(__name__)

LocalSession = get_local_session(Path(DATABASE_PATH, "api_requests.db"))
Base.metadata.create_all(bind=LocalSession.kw["bind"])

# initialize scheduler
scheduler = APScheduler(app=app)

interval_minutes_for_task = web_service_configuration.get("interval_time_background_task", 5)


# cron examples
@scheduler.task("interval", id="execute_pipelines", minutes=interval_minutes_for_task, next_run_time=datetime.now())
def execute_pipelines():
    cfg = configuration
    for model in MODELS_PIPELINE:
        cfg["log_linear"] = model == "log_linear"
        model_pipeline = get_pipeline(model, **cfg)
        model_pipeline.execute()
        model_pipeline.find_best_model()


@app.before_request
def log_request():
    """
    Add to global context variable g information about database and request parameters
    :return:
    """
    g.db = next(get_db(LocalSession))
    g.method = request.method
    g.url = request.url
    g.headers = dict(request.headers)
    g.body = request.get_json() if request.is_json else request.data.decode()


@app.after_request
def log_response(response):
    """
    Save response request to database
    :param response:
    :return:
    """
    method = g.get('method')
    url = g.get('url')
    headers = g.get('headers')
    body = g.get('body')
    status = response.status_code
    response_headers = dict(response.headers)
    response_body = response.get_data(as_text=True)

    save_request_response(g.db, method, url, headers, body, status, response_headers, response_body)
    return response


def get_pipeline(model, **config) -> Pipeline:
    """
    Get model pipeline instance based on given model and configuration
    :param model: model pipeline to fetch
    :param config: configuration of pipeline to fetch
    :return: pipeline instance
    """
    class_name = MODELS_PIPELINE[model].split(".")[1]
    model_pipeline_class = getattr(importlib.import_module(MODELS_PIPELINE[model]), class_name)
    model_pipeline: Pipeline = model_pipeline_class(**config)
    return model_pipeline


# Endpoint to get all books
@app.route('/models', methods=['GET'])
def get_models():
    models_available = """Available Models:\n""" + \
                       """ linear: Linear Regression model
                       log_linear: Linear Regression model with logarithmic targets
                       xgb: Tree-based model with XGBoost"""
    return models_available


# Predict diamond price given features
@app.route('/predict', methods=['POST'], defaults={'model': 'xgb'})
@app.route('/predict/<string:model>', methods=['POST'])
def predict(model):
    if model not in ["linear", "log_linear", "xgb"]:
        return jsonify(f'Error Invalid model. Request {request.host_url}/models for see available models.'), 400
    # Retrieve model pipeline
    model_pipeline = get_pipeline(model)
    best_model_path = model_pipeline.find_best_model(echo=False)
    if best_model_path is not None:
        data = request.get_json() if request.content_type == "application/json" else request.args
        model_pipeline.model = joblib.load(best_model_path)
        formatted_data = model_pipeline.format_data(data, True)
        predicted_price = model_pipeline.model.predict(formatted_data)
        predicted_price = np.exp(predicted_price) if model == "log_linear" else predicted_price
        return jsonify(predicted_price.tolist())
    return jsonify(f'{str(model).capitalize()} Model not present. Wait background process to finish training phase.'
                   f'Request {request.host_url}/models for see available models.'), 400


@app.route('/samples', methods=['POST'], defaults={'number': 5})
@app.route('/samples/<int:number>', methods=['POST'])
def get_samples(number):
    data = request.get_json() if request.content_type == "application/json" else request.args
    mandatory_features = ["cut", "clarity", "color", "carat"]
    if not set(mandatory_features).issubset(set(data.keys())):
        return jsonify({'Error': 'Invalid features input'}), 400
    base_pipeline = Pipeline()
    train_set, _ = base_pipeline.get_cleaned_data()
    formatted_features = base_pipeline.format_data(data)
    # Get samples from train set with same cut, color and clarity
    for feature in mandatory_features[:-1]:
        train_set = train_set[train_set[feature].isin(formatted_features[feature])]

    # Get indexes of n nearest samples to given carat value
    given_carat = formatted_features["carat"][0]
    nearest_indexes = (np.abs(train_set["carat"].to_numpy() - given_carat)).argsort()[:number]
    train_set = train_set.iloc[nearest_indexes]
    return jsonify(train_set.to_dict(orient='records'))


if __name__ == '__main__':
    scheduler.init_app(app)
    scheduler.start()
    host = web_service_configuration.get("host", "127.0.0.1")
    port = web_service_configuration.get("port", 5000)
    app.run(debug=True, use_reloader=False, host=host, port=port)
