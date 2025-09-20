###################################################################################
# ocr_translate-hugging_face - a plugin for ocr_translate                         #
# Copyright (C) 2023-present Davide Grassano                                      #
#                                                                                 #
# This program is free software: you can redistribute it and/or modify            #
# it under the terms of the GNU General Public License as published by            #
# the Free Software Foundation, either version 3 of the License.                  #
#                                                                                 #
# This program is distributed in the hope that it will be useful,                 #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                  #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                   #
# GNU General Public License for more details.                                    #
#                                                                                 #
# You should have received a copy of the GNU General Public License               #
# along with this program.  If not, see {http://www.gnu.org/licenses/}.           #
#                                                                                 #
# Home: https://github.com/Crivella/ocr_translate-hugging_face                    #
###################################################################################
"""Generic functions to work with HuggingFace's Classes."""

import logging
import os
from pathlib import Path

from transformers import (AutoImageProcessor, AutoModel, AutoModelForSeq2SeqLM,
                          AutoModelForZeroShotImageClassification,
                          AutoTokenizer, VisionEncoderDecoderModel)

from .tokenization_small100 import SMALL100Tokenizer

logger = logging.getLogger('plugin')

class Loaders():
    """Generic functions to load HuggingFace's Classes."""
    accept_device = ['ved_model', 'seq2seq', 'model']

    mapping = {
        'tokenizer': AutoTokenizer,
        'ved_model': VisionEncoderDecoderModel,
        'model': AutoModel,
        'image_processor': AutoImageProcessor,
        'seq2seq': AutoModelForSeq2SeqLM,
        'zsic_model': AutoModelForZeroShotImageClassification
    }

    model_overrides = {
        'alirezamsh/small100': {
            'tokenizer': SMALL100Tokenizer
        }
    }

    @staticmethod
    def _load(loader, model_id: str, root: Path):
        """Use the specified loader to load a transformers specific Class."""
        try:
            mid = root / model_id
            logger.debug(f'Attempt loading from store: "{loader}" "{mid}"')
            res = loader.from_pretrained(mid)
        except Exception:
            # Needed to catch some weird exception from transformers
            # eg: huggingface_hub.utils._validators.HFValidationError: Repo id must use alphanumeric chars or
            # '-', '_', '.', '--' and '..' are forbidden, '-' and '.'
            # cannot start or end the name, max length is 96: ...
            logger.debug(f'Attempt loading from cache: "{loader}" "{model_id}" "{root}"')
            res = loader.from_pretrained(model_id, cache_dir=root)
        return res

    @staticmethod
    def load(model_id: str, request: list[str], root: Path, dev: str = 'cpu') -> list:
        """Load the requested HuggingFace's Classes for the model into the memory of the globally specified device.

        Args:
            model_id (str): The HuggingFace model id to load, or a path to a local model.
            request (list[str]): A list of HuggingFace's Classes to load.
            root (Path): The root path to use for the cache.

        Raises:
            ValueError: If the model_id is not found or if the requested Class is not supported.

        Returns:
            _type_: A list of the requested Classes.
        """    """"""
        res = {}
        for r in request:
            if r not in Loaders.mapping:
                raise ValueError(f'Unknown request: {r}')
            model_cls = Loaders.mapping[r]
            model_cls = Loaders.model_overrides.get(model_id, {}).get(r, model_cls)
            cls = Loaders._load(model_cls, model_id, root)
            if cls is None:
                raise ValueError(f'Could not load model: {model_id}')
            logger.debug(f'Loaded `{r}`: {type(cls)}')

            if r in Loaders.accept_device:
                cls = cls.to(dev)

            res[r] = cls

        return res

class EnvMixin():
    """Mixin to allow usage of environment variables."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev = os.environ.get('DEVICE', 'cpu')
        if 'TRANSFORMERS_CACHE' in os.environ:
            self.root = Path(os.environ.get('TRANSFORMERS_CACHE'))
        elif 'OCT_BASE_DIR' in os.environ:
            self.root = Path(os.environ.get('OCT_BASE_DIR')) / 'models' / 'huggingface'
        else:
            raise ValueError('No TRANSFORMERS_CACHE or OCT_BASE_DIR environment variable found.')
        self.root.mkdir(parents=True, exist_ok=True)
        logger.debug(f'HuggingFace model dir: {self.root}')
