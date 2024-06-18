import joblib
import optuna
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from pipelines.BasePipeline import *
from utilities.utilities import *


def objective(trial: optuna.trial.Trial, hyperparams: dict, x_train, y_train, **kwargs) -> float:
    # Define hyperparameters to tune
    val_size, seed = kwargs.get("val_size"), kwargs.get("seed")

    param = {"random_state": seed, "enable_categorical": True}
    for hyperparam, hyperparam_cfg in hyperparams.items():
        hyperparam_values = hyperparam_cfg["values"]
        if hyperparam_cfg["type"] == "float":
            param[hyperparam] = trial.suggest_float(hyperparam, hyperparam_values[0], hyperparam_values[1], log=True)
        elif hyperparam_cfg["type"] == "int":
            param[hyperparam] = trial.suggest_int(hyperparam, hyperparam_values[0], hyperparam_values[1])
        elif hyperparam_cfg["type"] == "categorical":
            param[hyperparam] = trial.suggest_categorical(hyperparam, hyperparam_values)

    # Split the training data into training and validation sets

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=seed)

    # Train the model
    model = XGBRegressor(**param)
    model.fit(x_train, y_train)

    # Make predictions
    preds = model.predict(x_val)

    # Calculate MAE
    mae = mean_absolute_error(y_val, preds)

    return mae


class XGBRegressionPipeline(Pipeline):
    model: XGBRegressor
    model_name: str = "XGBRegression"
    model_file_name: str = "xgr_model.pkl"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.val_size = kwargs.get("val_size") or self.test_size
        self.hyperparameters = kwargs.get("hyperparameters") or {}
        self._set_save_path(kwargs.get("save_path") or MODELS_PATH)

    def data_preparation(self, data_to_prepare: pd.DataFrame = None) -> pd.DataFrame:
        data_to_prepare = self.dataset if data_to_prepare is None else data_to_prepare
        categorical_variables = ["cut", "color", "clarity"]
        for categorical_var in categorical_variables:
            categories = data_to_prepare[categorical_var].unique().tolist()
            data_to_prepare[categorical_var] = pd.Categorical(data_to_prepare[categorical_var],
                                                              categories=categories, ordered=True)
        self.dataset = data_to_prepare
        return self.dataset

    def _format_hyperparameters(self) -> dict:
        """
        Format hyperparameters.  For each hyperparameter is it possible to pass a dictionary configuration passing
        the data type and range of admitted values. This formatting is needed to properly set up bayesian optimization.
        :return:
        """
        number_class = {"float": float, "int": int, "categorical": float}
        formatted_hyperparams = {}
        for hyperparam, hyper_cfg in self.hyperparameters.items():
            if isinstance(hyper_cfg, dict):
                cast_class = number_class.get(hyper_cfg["type"])
                hyperparam_values = hyper_cfg.get("low_high") or hyper_cfg.get("choices")
                hyperparam_values = sorted([cast_class(hyper) for hyper in hyperparam_values.split(",")])
                formatted_hyperparams[hyperparam] = {"type": hyper_cfg["type"], "values": hyperparam_values}
        return formatted_hyperparams

    def tuning_hyperparameters(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> dict:
        """
        Perform hyperparameters tuning using optuna, a Bayesian hyperparameter tuning library.
        :param x_train: features training data used for tuning
        :param y_train: target training data used for tuning
        :return: best set of hyperparameters
        """
        best_params = {}
        if self.hyperparameters:
            n_trials = self.hyperparameters.pop("n_trials", 100)
            hyperparameters = self._format_hyperparameters()
            # Instantiate tuna study
            study = optuna.create_study(direction='minimize', study_name='Diamonds XGBoost')
            study.optimize(lambda trial: objective(trial, hyperparameters, x_train, y_train,
                                                   val_size=self.val_size, seed=self.seed), n_trials=n_trials)
            best_params = study.best_params
        return best_params

    def train(self, train_set: pd.DataFrame = None) -> XGBRegressor:
        train_set = self.train_set if train_set is None else train_set
        x_train, y_train = train_set.drop(columns=["price"]), train_set.price
        best_hyperparameters = self.tuning_hyperparameters(x_train, y_train)
        self.model = XGBRegressor(**best_hyperparameters, enable_categorical=True, random_state=self.seed)
        self.model.fit(x_train, y_train)
        # Save trained model and best hyperparameters
        joblib.dump(self.model, Path(self.save_path, self.model_file_name))
        if best_hyperparameters:
            write_yaml_file(Path(self.save_path, "best_hyperparameters.yaml"), best_hyperparameters)
        return self.model

    def evaluate(self, test_set: pd.DataFrame = None, model: XGBRegressor = None) -> tuple:
        test_set = self.test_set if test_set is None else test_set
        model = model or self.model
        x_test, y_test = test_set.drop(columns=["price"]), test_set.price
        preds = model.predict(x_test)
        # Compute R2 score and MAE
        r2, mae = round(r2_score(y_test, preds), 4), round(mean_absolute_error(y_test, preds), 2)
        metrics = {"R2": r2, "MAE": mae}
        print(f"Results reached:\nR2 Score:{r2}\nMAE:{mae}$")
        return y_test, preds, metrics
