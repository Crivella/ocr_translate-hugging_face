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
"""Plugins to enable usage of HuggingFace Models in ocr_translate"""

__version__ = '0.4.0'

khawhite_ocr_model_data = {
    'name': 'kha-white/manga-ocr-base',
    'lang': ['ja'],
    'lang_code': 'iso1',
    'entrypoint': 'hugginface.ved'
}

lucid_small_korean_ocr_model_data = {
    'name': 'team-lucid/trocr-small-korean',
    'lang': ['ko'],
    'lang_code': 'iso1',
    'entrypoint': 'hugginface.ved',
    'ocr_mode': 'single'
}

# The microsoft trocr models seems pretty biased toward numbers eg IT -> 17
# Tested with english but should work with all languages with latin alphabet?
microsoft_small_trocr_printed_model_data = {
    'name': 'microsoft/trocr-small-printed',
    'lang': [
        'sq', 'bm', 'be', 'bi', 'br', 'ch', 'co', 'hr', 'eo', 'et', 'fo', 'gl', 'ha', 'is', 'ig', 'ga', 'it', 'kk',
        'lv', 'lt', 'mg', 'mt', 'gv', 'mh', 'oc', 'sm', 'sc', 'sr', 'sn', 'sl', 'so', 'sw', 'ty', 'tr', 'tk', 'uz',
        'vi', 'cy', 'wo', 'yo', 'es', 'en', 'fr', 'de'
        ],
    'lang_code': 'iso1',
    'entrypoint': 'hugginface.ved',
    'ocr_mode': 'single'
}

microsoft_base_trocr_printed_model_data  = {
    'name': 'microsoft/trocr-base-printed',
    'lang': [
        'sq', 'bm', 'be', 'bi', 'br', 'ch', 'co', 'hr', 'eo', 'et', 'fo', 'gl', 'ha', 'is', 'ig', 'ga', 'it', 'kk',
        'lv', 'lt', 'mg', 'mt', 'gv', 'mh', 'oc', 'sm', 'sc', 'sr', 'sn', 'sl', 'so', 'sw', 'ty', 'tr', 'tk', 'uz',
        'vi', 'cy', 'wo', 'yo', 'es', 'en', 'fr', 'de'
        ],
    'lang_code': 'iso1',
    'entrypoint': 'hugginface.ved',
    'ocr_mode': 'single'
}

# microsoft_small_trocr_stage1_model_data = {
#     "name": "microsoft/trocr-small-stage1",
#     "lang": [
#         "sq", "bm", "be", "bi", "br", "ch", "co", "hr", "eo", "et", "fo", "gl", "ha", "is", "ig", "ga", "it", "kk",
#         "lv", "lt", "mg", "mt", "gv", "mh", "oc", "sm", "sc", "sr", "sn", "sl", "so", "sw", "ty", "tr", "tk", "uz",
#         "vi", "cy", "wo", "yo", "es", "en", "fr", "de"
#         ],
#     "lang_code": "iso1",
#     "entrypoint": "hugginface.ved",
#     "ocr_mode": "single"
# }

helsinki_zh_en_tsl_model_data = {
    'name': 'Helsinki-NLP/opus-mt-zh-en',
    'lang_src': ['zh'],
    'lang_dst': ['en'],
    'lang_code': 'iso1',
    'default_options': {
        'break_newlines': False
    },
    'entrypoint': 'hugginface.seq2seq'
}

helsinki_ja_en_tsl_model_data = {
    'name': 'Helsinki-NLP/opus-mt-ja-en',
    'lang_src': ['ja'],
    'lang_dst': ['en'],
    'lang_code': 'iso1',
    'default_options': {
        'break_newlines': True
    },
    'entrypoint': 'hugginface.seq2seq'
}

helsinki_ko_en_tsl_model_data = {
    'name': 'Helsinki-NLP/opus-mt-ko-en',
    'lang_src': ['ko'],
    'lang_dst': ['en'],
    'lang_code': 'iso1',
    'default_options': {
        'break_newlines': False
    },
    'entrypoint': 'hugginface.seq2seq'
}

helsinki_zh_en_tsl_model_data = {
    'name': 'Helsinki-NLP/opus-mt-zh-en',
    'lang_src': ['zh'],
    'lang_dst': ['en'],
    'lang_code': 'iso1',
    'default_options': {
        'break_newlines': False
    },
    'entrypoint': 'hugginface.seq2seq'
}

# Removed due to https://github.com/huggingface/transformers/issues/24657#issuecomment-3303054186
# staka_fugumt_ja_en_tsl_model_data = {
#     'name': 'staka/fugumt-ja-en',
#     'lang_src': ['ja'],
#     'lang_dst': ['en'],
#     'lang_code': 'iso1',
#     'default_options': {
#         'break_newlines': True
#     },
#     'entrypoint': 'hugginface.seq2seq'
# }

facebook_m2m100_418m_tsl_model_data = {
    'name': 'facebook/m2m100_418M',
    'lang_src': [
        'af', 'am', 'ar', 'az', 'ba', 'be', 'bg', 'bn', 'br', 'bs', 'cs', 'cy', 'da', 'de', 'en', 'et', 'fa','ff',
        'fi', 'fr', 'fy', 'ga', 'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja', 'jv',
        'ka', 'kk', 'km', 'kn', 'ko', 'lg', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'ne',
        'no', 'oc', 'or', 'pl', 'pt', 'ru', 'sd', 'sk', 'sl', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'th',
        'tl', 'tn', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh', 'zht', 'zu'
        ],
    'lang_dst': [
        'af', 'am', 'ar', 'az', 'ba', 'be', 'bg', 'bn', 'br', 'bs', 'cs', 'cy', 'da', 'de', 'en', 'et', 'fa','ff',
        'fi', 'fr', 'fy', 'ga', 'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja', 'jv',
        'ka', 'kk', 'km', 'kn', 'ko', 'lg', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'ne',
        'no', 'oc', 'or', 'pl', 'pt', 'ru', 'sd', 'sk', 'sl', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'th',
        'tl', 'tn', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh', 'zht', 'zu'
        ],
    'lang_code': 'iso1',
    'default_options': {
        'break_newlines': False
    },
    'entrypoint': 'hugginface.seq2seq',
    'iso1_map': {
        'zht': 'zh'
    }
}

facebook_m2m100_1_2b_tsl_model_data = {
    'name': 'facebook/m2m100_1.2B',
    'lang_src': [
        'af', 'am', 'ar', 'az', 'ba', 'be', 'bg', 'bn', 'br', 'bs', 'cs', 'cy', 'da', 'de', 'en', 'et', 'fa','ff',
        'fi', 'fr', 'fy', 'ga', 'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja', 'jv',
        'ka', 'kk', 'km', 'kn', 'ko', 'lg', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'ne',
        'no', 'oc', 'or', 'pl', 'pt', 'ru', 'sd', 'sk', 'sl', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'th',
        'tl', 'tn', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh', 'zht', 'zu'
        ],
    'lang_dst': [
        'af', 'am', 'ar', 'az', 'ba', 'be', 'bg', 'bn', 'br', 'bs', 'cs', 'cy', 'da', 'de', 'en', 'et', 'fa','ff',
        'fi', 'fr', 'fy', 'ga', 'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja', 'jv',
        'ka', 'kk', 'km', 'kn', 'ko', 'lg', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'ne',
        'no', 'oc', 'or', 'pl', 'pt', 'ru', 'sd', 'sk', 'sl', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'th',
        'tl', 'tn', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh', 'zht', 'zu'
        ],
    'lang_code': 'iso1',
    'default_options': {
        'break_newlines': False
    },
    'entrypoint': 'hugginface.seq2seq',
    'iso1_map': {
        'zht': 'zh'
    }
}
