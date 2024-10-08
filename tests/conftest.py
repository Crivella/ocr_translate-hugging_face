###################################################################################
# ocr_translate - a django app to perform OCR and translation of images.          #
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
# Home: https://github.com/Crivella/ocr_translate                                 #
###################################################################################
"""Fixtures for tests."""
from pathlib import Path

import numpy as np
import pytest
from ocr_translate import models as m
from PIL import Image

from ocr_translate_hugging_face import plugin as hugginface
from ocr_translate_hugging_face.plugin.utils import EnvMixin, Loaders

strings = [
    'This is a test string.',
    'This is a test string.\nWith a newline.',
    'This is a test string.\nWith a newline.\nAnd another.',
    'This is a test string.? With a special break character.',
    'This is a test string.? With a special break character.\nAnd a newline.',
    'String with a dash-newline brok-\nen word.'
]
ids = [
    'simple',
    'newline',
    'newlines',
    'breakchar',
    'breakchar_newline',
    'dash_newline'
]

@pytest.fixture(params=strings, ids=ids)
def string(request):
    """String to perform TSL on."""
    return request.param

@pytest.fixture()
def batch_string(string):
    """Batched string to perform TSL on."""
    return [string, string, string]


@pytest.fixture()
def language_dict():
    """Dict defining a language"""
    return {
        'name': 'Japanese',
        'iso1': 'ja',
        'iso2b': 'jpn',
        'iso2t': 'jpn',
        'iso3': 'jpn',
    }

@pytest.fixture(scope='session')
def image_pillow():
    """Random Pillow image."""
    npimg = np.random.randint(0,255,(25,25,3), dtype=np.uint8)
    return Image.fromarray(npimg)

@pytest.fixture()
def language(language_dict):
    """Language database object."""
    return m.Language(**language_dict)

@pytest.fixture()
def ved_model(language):
    """OCRModel database object."""
    model_dict = {
        'name': 'test_ved',
        'language_format': 'iso1',
        'entrypoint': 'hugginface.ved'
    }

    return hugginface.HugginfaceVEDModel(**model_dict)

@pytest.fixture()
def s2s_model(language):
    """OCRModel database object."""
    model_dict = {
        'name': 'test_seq2seq',
        'language_format': 'iso1',
        'entrypoint': 'hugginface.seq2seq'
    }

    return hugginface.HugginfaceSeq2SeqModel(**model_dict)



@pytest.fixture
def mock_loader(monkeypatch):
    """Mock hugging face class with `from_pretrained` method."""
    class Loader():
        """Mocked class."""
        def from_pretrained(self, model_id: Path | str, cache_dir=None):
            """Mocked method."""
            if isinstance(model_id, Path):
                if not model_id.is_dir():
                    raise FileNotFoundError('Not in dir')
            elif isinstance(model_id, str):
                if cache_dir is None:
                    cache_dir = EnvMixin().root
                if not (cache_dir / f'models--{model_id.replace("/", "--")}').is_dir():
                    raise FileNotFoundError('Not in cache')

            class A(): # pylint: disable=invalid-name
                """Mocked huggingface class with `to` method."""
                def to(self, dev): # pylint: disable=invalid-name,unused-argument,missing-function-docstring
                    pass
            return A()

    monkeypatch.setattr(Loaders, 'mapping', {
        'tokenizer': Loader(),
        'seq2seq': Loader(),
        'model': Loader(),
        'ved_model': Loader(),
        'image_processor': Loader(),
    })

@pytest.fixture(scope='function')
def mock_called(request):
    """Generic mock function to check if it was called."""
    def mock_call(*args, **kwargs): # pylint: disable=inconsistent-return-statements
        mock_call.called = True
        mock_call.args = args
        mock_call.kwargs = kwargs

        if hasattr(request, 'param'):
            return request.param

    if hasattr(request, 'param'):
        mock_call.expected = request.param

    return mock_call

@pytest.fixture()
def mock_ocr_preprocessor():
    """Mock preprocessor for OCR."""
    class RES():
        """Mock result"""
        def __init__(self):
            class PV(list):
                """Mock pixel values"""
                def cuda(self):
                    """Mock cuda"""
                    self.cuda_called = True # pylint: disable=attribute-defined-outside-init
                    return self
            self.pixel_values = PV([1,2,3,4,5])

    class _MockPreprocessor():
        def __init__(self, model_id):
            self.model_id = model_id
            self.options = {}

        def __call__(self, img, **options):
            self.options = options
            res = RES()
            return res

    return _MockPreprocessor

@pytest.fixture()
def mock_ocr_tokenizer():
    """Mock tokenizer for OCR."""
    class _MockTokenizer():
        def __init__(self, model_id):
            self.model_id = model_id
            self.options = {}

        def batch_decode(self, tokens, **options):
            """Mock batch decode."""
            self.options = options
            offset = ord('a') - 1
            return [''.join(chr(int(_)+offset) for _ in tokens)]

    return _MockTokenizer

@pytest.fixture()
def mock_ocr_model():
    """Mock model for OCR."""
    class _MockModel():
        def __init__(self, model_id):
            self.model_id = model_id
            self.options = {}

        def generate(self, pixel_values=None, **options):
            """Mock generate."""
            self.options = options
            return pixel_values

    return _MockModel

@pytest.fixture()
def mock_box_reader():
    """Mock box reader."""
    class _MockReader():
        def __init__(self, model_id):
            self.model_id = model_id
            self.options = {}

        def detect(self, img, **options): # pylint: disable=unused-argument
            """Mock recognize."""
            self.options = options
            return (([(10,10,30,30), (40,40,50,50)],),)

    return _MockReader

@pytest.fixture()
def mock_tsl_tokenizer():
    """Mock tokenizer for TSL."""
    import torch  # pylint: disable=import-outside-toplevel
    class _MockTokenizer():
        def __init__(self, model_id):
            self.model_id = model_id
            self.other_options = {}
            self.tok_to_word = {0: 0}
            self.word_to_tok = {0: 0}
            self.ntoks = 1
            self.called_get_lang_id = False

        def __call__(self, text, **options):
            issplit = options.pop('is_split_into_words', False)
            padding = options.pop('padding', False)
            truncation = options.pop('truncation', False) # pylint: disable=unused-variable

            self.other_options = options

            if isinstance(text, list):
                if isinstance(text[0], str):
                    text = [text]
                if issplit:
                    app = []
                    for line in text:
                        app2 = []
                        for seg in line:
                            app2.extend(seg.split(' '))
                        app.append(app2)
                else:
                    app = [_.split(' ') for _ in text]

                if padding:
                    app2 = []
                    for lst in app:
                        app3 = []
                        for word in lst:
                            if word not in self.word_to_tok:
                                self.word_to_tok[word] = self.ntoks
                                self.tok_to_word[self.ntoks] = word
                                self.ntoks += 1
                            app3.append(self.word_to_tok[word])
                        app2.append(app3)

                    max_len = max(len(_) for _ in app2)
                    res = [(_ + [0] * max_len)[:max_len] for _ in app2]
                else:
                    res = app
                class Dict(dict):
                    """Dict class with added .to method"""
                    def to(self, device): # pylint: disable=unused-argument,invalid-name
                        """Move the dict to a device."""
                        return None

                dct = Dict([('input_ids', torch.Tensor(res))])
                return dct

            raise TypeError(f'Expected list of strings, but got {type(text)}')

        def batch_decode(self, tokens, **options): # pylint: disable=unused-argument
            """Decode a batch of tokens."""
            res = [' '.join(filter(None, [self.tok_to_word[int(_)] for _ in lst])) for lst in tokens]
            return res

        def get_lang_id(self, lang): # pylint: disable=unused-argument
            """Get the language id."""
            self.called_get_lang_id = True
            return 0

    return _MockTokenizer

@pytest.fixture()
def mock_tsl_model():
    """Mock model for TSL."""
    class _MockModel():
        def __init__(self, model_id):
            self.model_id = model_id
            self.options = {}

        def generate(self, input_ids=None, **options):
            """Mock generate translated tokens."""
            self.options = options
            return input_ids

    return _MockModel
