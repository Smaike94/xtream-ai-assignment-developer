import json
import shutil
import sys
from typing import Optional, Tuple
from datetime import datetime

import requests
import numpy as np
import validators
import pandas as pd
from sklearn.model_selection import train_test_split
from utilities.databes_utilities import *
from database.tables import Base, Request
from utilities.utilities import *
import matplotlib.pyplot as plt

FILE_PATH = Path(__file__).resolve()
ROOT = Path(*FILE_PATH.parts[:-2])
DATA_PATH, MODELS_PATH, DATABASE_PATH = ROOT / 'data', ROOT / 'models', ROOT / 'database'
DEFAULT_SEED, DEFAULT_TEST_SIZE = 42, 0.2


def check_new_data_from_url(url: str, model_name: str, db: Session) -> Tuple[bool, bool, list]:
    response = requests.get(url, json={"model_name": model_name})
    response_sha = get_sha256(response.text)
    dumped_sha = json.dumps(response_sha)

    get_model_name_from_req = lambda req: json.loads(str(req.request_body)).get("model_name", "")
    all_requests = db.query(Request).all()
    timestamp_requests = []
    previous_requests_same_sha = [prev_request for prev_request in all_requests if
                                  prev_request.response_body == dumped_sha]
    if previous_requests_same_sha:
        timestamp_requests.append(previous_requests_same_sha[0].timestamp.isoformat())

    previous_requests_by_model_body = [prev_request.response_body for prev_request in all_requests
                                       if get_model_name_from_req(prev_request) == model_name]

    for el in previous_requests_by_model_body:
        tmp = [prev_request for prev_request in all_requests if prev_request.response_body == el]
        if tmp:
            timestamp_requests.append(tmp[0].timestamp.isoformat())

    if_new_sha_for_model = [prev_request for prev_request in previous_requests_same_sha
                            if get_model_name_from_req(prev_request) == model_name]
    data_already_downloaded, new_data_for_model = len(previous_requests_same_sha) != 0, len(if_new_sha_for_model) == 0


    if not data_already_downloaded or new_data_for_model:
        tmp = save_request_response(db, response.request.method, response.request.url,
                                    dict(response.request.headers), json.loads(response.request.body.decode()),
                                    response.status_code, dict(response.headers), response_sha)
        if not data_already_downloaded:
            timestamp_requests.append(tmp.timestamp.isoformat())

    return data_already_downloaded, new_data_for_model, timestamp_requests


class Pipeline:
    model_name: Optional[str] = ""
    model_file_name: Optional[str]
    dataset: pd.DataFrame
    train_set: pd.DataFrame
    test_set: pd.DataFrame

    def __init__(self, **kwargs) -> None:
        self._init_fetch_db()
        self._set_save_path(kwargs.get("save_path") or MODELS_PATH)
        self.seed = kwargs.get("seed") or DEFAULT_SEED
        self.data_src = kwargs.get("data_src") or DATA_PATH
        self.test_size = kwargs.get("test_size") or DEFAULT_TEST_SIZE

    def _init_fetch_db(self):
        """
        Initialize database and relative session. This will be used to properly fetch new data.
        :return:
        """
        self.local_session = get_local_session(Path(DATABASE_PATH, "fetch_requests.db"), echo=False)
        Base.metadata.create_all(self.local_session.kw["bind"])

    def _set_save_path(self, save_path: Union[str, Path]):
        """
        Set run path for this pipeline.
        :param save_path: folder path under which run folders are saved
        :return:
        """
        timestamp = datetime.now().replace(microsecond=0).isoformat().replace(":", "-")
        self.save_path = Path(save_path) / self.model_name / timestamp

    def fetch_data(self, check_new_data: bool = True):
        new_data_found = False or (self.find_best_model(echo=False) is None)
        # Get local data csv files, both for train and test set

        csv_files = {"train": [el for el in DATA_PATH.iterdir() if el.suffix == ".csv" and "test" not in el.name],
                     "test": [el for el in DATA_PATH.iterdir() if el.suffix == ".csv" and "test" in el.name]}

        train_data, test_data = pd.DataFrame(), pd.DataFrame()
        if csv_files["train"]:
            train_data = pd.concat([pd.read_csv(train_files) for train_files in csv_files["train"]])
        if csv_files["test"]:
            test_data = pd.concat([pd.read_csv(test_files) for test_files in csv_files["test"]])
        else:
            # Pensa se fare a modo tuo o meno
            train_data, test_data = self.split_data(self.data_cleaning(train_data))
            test_data.to_csv(DATA_PATH / "diamonds_test.csv", index=False)
            # train_data.to_csv(DATA_PATH / f"diamonds.csv", index=False)

        if check_new_data:
            if validators.url(self.data_src):
                db_session = next(get_db(self.local_session))
                data_already_downloaded, new_data_found, timestamps = check_new_data_from_url(self.data_src,
                                                                                              self.model_name,
                                                                                              db_session)

                if data_already_downloaded:
                    train_data = train_data[train_data["Timestamp"].isin(timestamps)]
                if not data_already_downloaded:
                    new_data = self.data_cleaning(pd.read_csv(self.data_src))
                    local_data = pd.concat([train_data, test_data])
                    new_data = new_data.sort_values(by=new_data.columns.tolist()).reset_index(drop=True)
                    local_data = local_data.sort_values(by=new_data.columns.tolist()).reset_index(drop=True)
                    if not local_data.equals(new_data):
                        new_data_found = True
                        new_data_not_in_test_set = new_data[~new_data.apply(tuple, 1).isin(test_data.apply(tuple, 1))]
                        train_data = pd.concat([train_data, new_data_not_in_test_set])
                        train_data["Timestamp"] = train_data["Timestamp"].fillna(timestamps[-1])
                        train_data.to_csv(DATA_PATH / "diamonds.csv", index=False)

                if "Timestamp" not in train_data:
                    train_data["Timestamp"] = timestamps[-1]
                    train_data.to_csv(DATA_PATH / "diamonds.csv", index=False)

            # assert not train_data.apply(tuple, 1).isin(test_data.apply(tuple, 1)).any()

        train_data["split"], test_data["split"] = "train", "test"
        # self.dataset = pd.concat([train_data.reset_index(drop=True), test_data.reset_index(drop=True)])
        self.dataset = pd.concat([train_data.drop(columns=["Timestamp"], errors="ignore"), test_data])
        return self.dataset, new_data_found

    def data_cleaning(self, data_to_clean: pd.DataFrame = None) -> pd.DataFrame:
        """
        Clean data, remove negative prices and zeros diamonds dimensions.
        :param data_to_clean: dataframe to clean. If not passes as default is used class attribute self.dataset
        :return: cleaned dataframe
        """
        data_to_clean = self.dataset if data_to_clean is None else data_to_clean
        data_to_clean = data_to_clean[(data_to_clean.x * data_to_clean.y * data_to_clean.z != 0) &
                                      (data_to_clean.price > 0)]
        self.dataset = data_to_clean
        return self.dataset

    def data_preparation(self, data_to_prepare: pd.DataFrame = None) -> pd.DataFrame:
        """
        Preprocess data for training and testing. This method must be overridden by subclasses in order to reflect
        specific preprocessing procedure.
        :param data_to_prepare: dataframe to prepare. If not passes as default is used class attribute dataset
        :return: processed dataframe
        """
        raise NotImplementedError("Data preparation not implemented yet")

    def train(self, train_set: pd.DataFrame = None):
        """
        Train model using given train set. This method must be overridden by subclasses in order to reflect
        specific training procedure.
        :param train_set: dataframe on which to train model. If not passes as default is used class attribute train_set
        :return:
        """
        raise NotImplementedError("Training phase not implemented yet")

    def evaluate(self, test_set: pd.DataFrame = None):
        """
        Evaluate model performance on given test_set. This method must be overridden by subclasses in order to reflect
        specific evaluating procedure.
        :param test_set: dataframe on which to evaluate model. If not passes as default is used class attribute test_set
        :return:
        """
        raise NotImplementedError("Evaluating phase not implemented yet")

    def save_results(self, ground_truth: pd.Series, predictions: pd.Series, **metrics):
        """
        Save metrics to csv file. Moreover, a plot of predictions against ground truth is saved.
        :param ground_truth: ground truth values
        :param predictions: predicted values
        :param metrics: evaluation metrics
        :return:
        """
        timestamp, results_path = self.save_path.name, self.save_path.parent / 'results.csv'
        # Save plot actual vs predicted
        self.plot_gof(ground_truth, predictions, self.save_path)
        # Save results.csv file
        metrics.update({"Timestamp": [timestamp]})
        results_df = pd.DataFrame.from_dict(metrics)
        if results_path.exists():
            results_df = pd.concat([pd.read_csv(results_path), results_df])
        results_df.to_csv(results_path, index=False)

    def find_best_model(self, echo=True) -> Union[None, Path]:
        """
        Retrieve best model path for specified model.
        :param echo: If true print to standard output R2 Score and MAE of best model.
        :return:
        """
        results_csv = self.save_path.parent / 'results.csv'
        if results_csv.exists():
            best_result = pd.read_csv(results_csv).sort_values(by="MAE", ascending=True).iloc[0]
            best_model_path = self.save_path.parent / best_result["Timestamp"] / self.model_file_name
            best_model_path = best_model_path.resolve()
            if echo:
                print(f"\nBest model found for {self.model_name}:")
                print(f"Path: {best_model_path}\nR2 Score: {best_result['R2']}\nMAE: {best_result['MAE']}")
            return best_model_path

    def format_data(self, data_to_format: dict, for_prediction: bool = False) -> pd.DataFrame:
        """
        This method formats given data accordingly to dataset specification.
        Used to format json data from API requests.
        :param data_to_format: data to format
        :param for_prediction: If true, in addition to formatting, the data is also preprocessed to meet model
        requirements, such as feature types or order sequence.
        :return:
        """

        def get_default_value(col_dtype: str):
            if pd.api.types.is_integer_dtype(col_dtype):
                return 0
            elif pd.api.types.is_float_dtype(col_dtype):
                return 0.0
            elif pd.api.types.is_string_dtype(col_dtype):
                return ""

        list_elements = lambda elements: elements if isinstance(elements, list) else [elements]
        max_length = max([len(list_elements(data_values)) for data_values in data_to_format.values()])
        data_schema = pd.read_csv(DATA_PATH / "diamonds.csv", nrows=1)
        # Create DataFrame with same columns and data types of diamonds dataset, also setting properly default values.
        formatted_data = {col: pd.Series(data=data_to_format.get(col, get_default_value(col_dtype)), dtype=col_dtype)
                          for col, col_dtype in zip(data_schema.columns, data_schema.dtypes)}
        formatted_data = pd.DataFrame(formatted_data, index=np.arange(max_length))
        if for_prediction:
            return self.data_preparation(formatted_data.drop(columns=["price", "Timestamp"], errors="ignore"))
        return formatted_data

    def split_data(self, data_to_split: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        :param data_to_split: if not passes as default is used class attribute dataset
        :return: train and test sets dataframes
        """
        data_to_split = self.dataset if data_to_split is None else data_to_split
        if "split" in data_to_split.columns:
            self.train_set = data_to_split[data_to_split["split"] == "train"].drop("split", axis=1)
            self.test_set = data_to_split[data_to_split["split"] == "test"].drop("split", axis=1)
            self.train_set, self.test_set = self.train_set.reset_index(drop=True), self.test_set.reset_index(drop=True)
        else:
            self.train_set, self.test_set = train_test_split(data_to_split, test_size=self.test_size,
                                                             random_state=self.seed)
        return self.train_set, self.test_set

    def get_cleaned_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform data cleaning on train and test sets. This subpart of complete pipeline is useful in case it is needed
        to retrieve train and test some sorts of analysis.
        :return: train and test sets dataframes
        """
        self.fetch_data(check_new_data=False)
        self.data_cleaning()
        return self.split_data()

    @staticmethod
    def plot_gof(ground_truth: pd.Series, predictions: pd.Series, save_path: Path, save_fig: bool = True):
        """
        Produce graphical plot of predictions against ground truth.
        :param ground_truth: ground truth values
        :param predictions: predicted values
        :param save_path: path to save figure
        :param save_fig: If true save figure
        :return:
        """
        plt.plot(ground_truth, predictions, '.')
        plt.plot(ground_truth, ground_truth, linewidth=3, c='black')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        if save_fig:
            plt.savefig(save_path / "actual_vs_predicted.png")

    def execute(self):
        """
        Execute the whole pipeline from data fetching to saving results.
        :return:
        """
        print(f"Start executing pipeline for {self.model_name}", file=sys.stderr)
        try:
            _, new_data = self.fetch_data()
            if new_data:
                self.save_path.mkdir(parents=True)
                self.data_cleaning()
                self.data_preparation()
                self.split_data()
                self.train()
                y_true, preds, metrics = self.evaluate()
                self.save_results(y_true, preds, **metrics)
                print(f"Pipeline executed successfully", file=sys.stderr)
            else:
                print(f"No new data found. No new model has been trained.", file=sys.stderr)
        except Exception as e:
            shutil.rmtree(self.save_path)
            print(e)
