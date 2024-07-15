import os
import pandas

from time import strftime, gmtime
from typing import Union, List

import torch
from typeguard import typechecked


class Logger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir  # .../log/v_2

    def log_txt(self, filename: str, message):
        path = os.path.join(self.log_dir, f'{filename}.txt')
        f = open(path, "a")
        f.write(strftime("%Y-%m-%d %H:%M:%S:: \t", gmtime()))
        f.write('\n')
        f.write(message)
        f.write('\n')
        f.close()

    @typechecked
    def log_csv(self,
                filename: str,
                key: Union[str, List[str]],
                value: Union[float, List[float], List[List[float]]],
                epoch: int):

        if isinstance(key, str) and isinstance(value, float):
            data = {"epoch": [epoch], "value": [value]}
        else:
            data = {"epoch": len(value[0]) * [epoch]}
            for k, v in zip(key, value):
                data[k] = v

        path = os.path.join(self.log_dir, f'{filename}.csv')

        try:
            df = pandas.read_csv(path)
        except FileNotFoundError:
            df = pandas.DataFrame(columns=data.keys())

        df = pandas.concat([df, pandas.DataFrame(data)], ignore_index=True)
        df.to_csv(path, index=False)
