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
"""Class to enable hugginface Seq2Seq models."""
import logging

import torch
from ocr_translate import models as m
from transformers import M2M100Tokenizer

from .tokenization_small100 import SMALL100Tokenizer
from .utils import EnvMixin, Loaders

logger = logging.getLogger('plugin')

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

        if isinstance(self.tokenizer, SMALL100Tokenizer):
            self.tokenizer.tgt_lang = dst_lang
        else:
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
