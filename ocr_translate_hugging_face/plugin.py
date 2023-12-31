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
"""ocr_translate plugin to allow loading of hugginface models."""
import logging
import os
from pathlib import Path

import torch
from ocr_translate import models as m
from PIL import Image
from transformers import (AutoImageProcessor, AutoModel, AutoModelForSeq2SeqLM,
                          AutoTokenizer, M2M100Tokenizer,
                          VisionEncoderDecoderModel)

logger = logging.getLogger('plugin')

class Loaders():
    """Generic functions to load HuggingFace's Classes."""
    accept_device = ['ved_model', 'seq2seq', 'model']

    mapping = {
        'tokenizer': AutoTokenizer,
        'ved_model': VisionEncoderDecoderModel,
        'model': AutoModel,
        'image_processor': AutoImageProcessor,
        'seq2seq': AutoModelForSeq2SeqLM
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
            cls = Loaders._load(Loaders.mapping[r], model_id, root)
            if cls is None:
                raise ValueError(f'Could not load model: {model_id}')

            if r in Loaders.accept_device:
                cls = cls.to(dev)

            res[r] = cls

        return res


def get_mnt(ntok: int, options: dict) -> int:
    """Get the maximum number of new tokens to generate."""
    min_max_new_tokens = int(options.get('min_max_new_tokens', 20))
    max_max_new_tokens = int(options.get('max_max_new_tokens', 512))
    max_new_tokens_ratio = float(options.get('max_new_tokens_ratio', 3.0)
)
    if min_max_new_tokens > max_max_new_tokens:
        raise ValueError('min_max_new_tokens must be less than max_max_new_tokens')

    mnt = min(
        max_max_new_tokens,
        max(
            min_max_new_tokens,
            max_new_tokens_ratio * ntok
        )
    )
    return int(mnt)

class EnvMixin():
    """Mixin to allow usage of environment variables."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev = os.environ.get('DEVICE', 'cpu')
        self.root = Path(os.environ.get('TRANSFORMERS_CACHE', '.'))
        logger.debug(f'Cache dir: {self.root}')

class HugginfaceSeq2SeqModel(m.TSLModel, EnvMixin):
    """OCRtranslate plugin to allow loading of hugginface seq2seq model as translator."""
    ALLOWED_OPTIONS = {
        **m.TSLModel.ALLOWED_OPTIONS,
        'min_max_new_tokens': {
            'type': int,
            'default': 20,
            'description': 'Minimum number for the maximum number of tokens to generate.',
        },
        'max_max_new_tokens': {
            'type': int,
            'default': 512,
            'description': 'Maximum number for the maximum number of tokens to generate.',
        },
        'max_new_tokens_ratio': {
            'type': float,
            'default': 3,
            'description': 'Attempts to generate `ratio` * `#original_tokens` tokens during translation.',
        },
    }

    class Meta: # pylint: disable=missing-class-docstring
        proxy = True

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.model = None

    def load(self):
        """Load the model into memory."""
        logger.info(f'Loading TSL model: {self.name}')
        res = Loaders.load(self.name, request=['seq2seq', 'tokenizer'], root=self.root, dev=self.dev)
        self.model = res['seq2seq']
        self.tokenizer = res['tokenizer']

    def unload(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.dev == 'cuda':
            torch.cuda.empty_cache()


    def _translate(
            self,
            tokens: list[str] | list[list[str]],
            src_lang: str, dst_lang: str,
            options: dict = None
            ) -> str | list[str]:
        """Translate a text using a the loaded model.

        Args:
            tokens (list): list or list[list] of string tokens to be translated.
            lang_src (str): Source language.
            lang_dst (str): Destination language.
            options (dict, optional): Options for the translation. Defaults to {}.

        Raises:
            TypeError: If text is not a string or a list of strings.

        Returns:
            Union[str,list[str]]: Translated text. If text is a list, returns a list of translated strings.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError('Model not loaded')
        if options is None:
            options = {}
        if not isinstance(tokens, list):
            raise TypeError('tokens must be a list of strings or a list of list of strings')

        logger.debug(f'TSL: {tokens}')
        if len(tokens) == 0:
            return ''

        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(
            tokens,
            return_tensors='pt',
            padding=True,
            truncation=True,
            is_split_into_words=True
            )
        ntok = encoded['input_ids'].shape[1]
        encoded.to(self.dev)

        mnt = get_mnt(ntok, options)

        kwargs = {
            'max_new_tokens': mnt,
        }
        if isinstance(self.tokenizer, M2M100Tokenizer):
            kwargs['forced_bos_token_id'] = self.tokenizer.get_lang_id(dst_lang)

        logger.debug(f'TSL ENCODED: {encoded}')
        logger.debug(f'TSL KWARGS: {kwargs}')
        generated_tokens = self.model.generate(
            **encoded,
            **kwargs,
            )

        tsl = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        logger.debug(f'TSL: {tsl}')

        if isinstance(tokens[0], str):
            tsl = tsl[0]

        if self.dev == 'cuda':
            torch.cuda.empty_cache()

        return tsl

    # def translate_batch(self, texts):
    #     """Translate a batch of texts."""
    #     raise NotImplementedError

class HugginfaceVEDModel(m.OCRModel, EnvMixin):
    """OCRtranslate plugin to allow loading of hugginface VisionEncoderDecoder model as text OCR."""
    class Meta: # pylint: disable=missing-class-docstring
        proxy = True

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.model = None
        self.image_processor = None

    def load(self):
        """Load the model into memory."""
        logger.info(f'Loading OCR VED model: {self.name}')
        res = Loaders.load(
            self.name, request=['ved_model', 'tokenizer', 'image_processor'],
            root=self.root, dev=self.dev
            )
        self.model = res['ved_model']
        self.tokenizer = res['tokenizer']
        self.image_processor = res['image_processor']

    def unload(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.image_processor is not None:
            del self.image_processor
            self.image_processor = None

        if self.dev == 'cuda':
            torch.cuda.empty_cache()

    def _ocr(
            self,
            img: Image.Image, lang: str = None, options: dict = None
            ) -> str:
        """Perform OCR on an image.

        Args:
            img (Image.Image):  A Pillow image on which to perform OCR.
            lang (str, optional): The language to use for OCR. (Not every model will use this)
            bbox (tuple[int, int, int, int], optional): The bounding box of the text on the image in lbrt format.
            options (dict, optional): A dictionary of options to pass to the OCR model.

        Raises:
            TypeError: If img is not a Pillow image.

        Returns:
            str: The text extracted from the image.
        """
        if self.model is None or self.tokenizer is None or self.image_processor is None:
            raise RuntimeError('Model not loaded')

        if options is None:
            options = {}

        pixel_values = self.image_processor(img, return_tensors='pt').pixel_values
        if self.dev == 'cuda':
            pixel_values = pixel_values.cuda()
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if self.dev == 'cuda':
            torch.cuda.empty_cache()

        return generated_text
