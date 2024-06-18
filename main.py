import argparse
from utilities.utilities import *

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, choices=["linear", "log_linear", "xgb"])
parser.add_argument("-c", "--conf_path", type=str, default="configuration.yaml")
parsed_args = parser.parse_args()


if parsed_args.model == "linear" or parsed_args.model == "log_linear":
    from pipelines.LinearRegressionPipeline import LinearRegressionPipeline as Pipeline
elif parsed_args.model == "xgb":
    from pipelines.XGBRegressionPipeline import XGBRegressionPipeline as Pipeline

if __name__ == '__main__':
    conf = read_yaml_file(parsed_args.conf_path)
    conf["log_linear"] = parsed_args.model == "log_linear"
    pipeline = Pipeline(**conf)
    pipeline.execute()
    pipeline.find_best_model()
