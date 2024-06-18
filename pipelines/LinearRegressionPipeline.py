import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from pipelines.BasePipeline import *


class LinearRegressionPipeline(Pipeline):
    model: LinearRegression
    model_name: str = "LinearRegression"
    model_file_name: str = "linear_model.pkl"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LinearRegression()
        self.log_linear = kwargs.get("log_linear", False)
        self.model_name = f"Log{self.model_name}" if self.log_linear else self.model_name
        self.model_file_name = f"log_{self.model_file_name}" if self.log_linear else self.model_file_name
        self._set_save_path(kwargs.get("save_path") or MODELS_PATH)

    def format_data(self, data_to_format: dict, for_prediction: bool = False) -> pd.DataFrame:
        """
        This method override format_data over super class BasePipeline, since for Linear regression models
        is needed to do specific procedure. In particular, is needed to return formatted data with features in same order
        has been passed in training phase.
        :param data_to_format:
        :param for_prediction:
        :return:
        """
        cat_columns = ['cut', 'color', 'clarity']
        formatted_data = super().format_data(data_to_format, for_prediction=False)
        formatted_data[cat_columns] = formatted_data[cat_columns].apply(lambda col: col.name + '_' + col.astype(str))
        # Get feature names of already trained model
        features_name = getattr(self.model, "feature_names_in_", [])
        features_data = pd.DataFrame(columns=features_name, index=range(len(formatted_data)))
        for col in features_data.columns:
            if col in formatted_data.columns:
                features_data[col] = formatted_data[col].astype(formatted_data[col].dtype)
            else:
                # In case of categorical feature name, formatted with column name + dummy value.
                col_prefix = col.split('_')[0]
                features_data[col] = formatted_data[col_prefix].isin([col])
        return features_data

    def data_preparation(self, data_to_prepare: pd.DataFrame = None) -> pd.DataFrame:
        data_to_prepare = self.dataset if data_to_prepare is None else data_to_prepare
        data_to_prepare = data_to_prepare.drop(columns=['depth', 'table', 'y', 'z'])
        data_to_prepare = pd.get_dummies(data_to_prepare, columns=['cut', 'color', 'clarity'], drop_first=True)
        self.dataset = data_to_prepare
        return self.dataset

    def train(self, train_set: pd.DataFrame = None) -> LinearRegression:
        train_set = self.train_set if train_set is None else train_set
        x_train, y_train = train_set.drop(columns=["price"]), train_set.price
        y_train = np.log(y_train) if self.log_linear else y_train
        self.model.fit(x_train, y_train)
        # Save trained model
        joblib.dump(self.model, Path(self.save_path, self.model_file_name))
        return self.model

    def evaluate(self, test_set: pd.DataFrame = None, model: LinearRegression = None) -> tuple:
        test_set = self.test_set if test_set is None else test_set
        model = model or self.model
        x_test, y_test = test_set.drop(columns=["price"]), test_set.price
        preds = model.predict(x_test)
        preds = np.exp(preds) if self.log_linear else preds
        # Compute R2 score and MAE
        r2, mae = round(r2_score(y_test, preds), 4), round(mean_absolute_error(y_test, preds), 2)
        metrics = {"R2": r2, "MAE": mae}
        print(f"Results reached:\nR2 Score:{r2}\nMAE:{mae}$")
        return y_test, preds, metrics

