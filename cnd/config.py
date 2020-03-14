import pathlib
import trafaret as t
from trafaret_config import read_and_validate

OCR_EXPERIMENTS_DIR = pathlib.Path('/workdir/data/exp/')
CONFIG_PATH = pathlib.Path('/workdir/cnd/config.json')

CONFIG_TRAFARET = t.Dict({
    t.Key('data_path'): t.String(),
    t.Key('ocr_image_size'): t.List(t.Int[1:], 2, 2),
})


class Config:
    def __init__(self, config_path):
        self.data = read_and_validate(config_path, CONFIG_TRAFARET)

    def _get(self, data, keys):
        next_key, keys = keys[0], keys[1:]

        if not keys:
            if next_key == "*":
                if isinstance(data, dict):
                    return [data[k] for k in data.keys()]
                else:
                    return data
            else:
                return data[next_key]

        if next_key == "*":
            if isinstance(data, dict):
                return [self._get(data[k], keys) for k in data.keys()]
            else:
                return [self._get(d, keys) for d in data]
        else:
            return self._get(data[next_key], keys)

    def get(self, *keys):
        return self._get(self.data, keys)
