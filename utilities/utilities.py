from pathlib import Path
from typing import Union
import hashlib
import yaml


def read_yaml_file(file_path: Union[str, Path]) -> dict:
    with open(file_path, "r") as data_file:
        return yaml.load(data_file, Loader=yaml.FullLoader)


def write_yaml_file(file_path: Union[str, Path], data: dict):
    with open(file_path, "w") as data_file:
        yaml.dump(data, data_file, sort_keys=False)


def get_sha256(text: str) -> str:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(text.encode('utf-8'))
    return sha256_hash.hexdigest()

