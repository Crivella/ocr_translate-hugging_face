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

# Tested with transformers 4.56.1 but not great results. When it translates it is decent, but often
# if just outputs the romanization (or something close) of the japanese text.
# mitsua_ja_en_tsl_model_data = {
#     'name': 'Mitsua/elan-mt-bt-ja-en',
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

# facebook/nllb-200-distilled-600M
# It is missing some languages that i might not have matched with my script feel free to improve this as needed
facebook_nllb_600m_tsl_model_data = {
    'name': 'facebook/nllb-200-distilled-600M',
    'lang_src': [
        'af', 'ak', 'am', 'as', 'ba', 'bm', 'be', 'bn', 'bs', 'bg', 'cs', 'cy', 'da', 'de', 'dz', 'en', 'eo', 'et',
        'eu', 'ee', 'fo', 'fj', 'fi', 'fr', 'ga', 'gl', 'gn', 'gu', 'ha', 'he', 'hi', 'hr', 'hu', 'hy', 'ig', 'id',
        'is', 'it', 'jv', 'ja', 'kn', 'ka', 'kk', 'rw', 'ko', 'lo', 'ln', 'lt', 'lg', 'ml', 'mr', 'mk', 'mt', 'mi',
        'my', 'nn', 'ne', 'oc', 'pl', 'pt', 'rn', 'ru', 'sg', 'sa', 'sk', 'sl', 'sm', 'sn', 'sd', 'so', 'st', 'sc',
        'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'tt', 'te', 'tg', 'tl', 'th', 'ti', 'tn', 'ts', 'tk', 'tr', 'tw', 'uk',
        'ur', 'vi', 'wo', 'xh', 'yo', 'zu', 'zh', 'zht'],
    'lang_dst': [
        'af', 'ak', 'am', 'as', 'ba', 'bm', 'be', 'bn', 'bs', 'bg', 'cs', 'cy', 'da', 'de', 'dz', 'en', 'eo', 'et',
        'eu', 'ee', 'fo', 'fj', 'fi', 'fr', 'ga', 'gl', 'gn', 'gu', 'ha', 'he', 'hi', 'hr', 'hu', 'hy', 'ig', 'id',
        'is', 'it', 'jv', 'ja', 'kn', 'ka', 'kk', 'rw', 'ko', 'lo', 'ln', 'lt', 'lg', 'ml', 'mr', 'mk', 'mt', 'mi',
        'my', 'nn', 'ne', 'oc', 'pl', 'pt', 'rn', 'ru', 'sg', 'sa', 'sk', 'sl', 'sm', 'sn', 'sd', 'so', 'st', 'sc',
        'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'tt', 'te', 'tg', 'tl', 'th', 'ti', 'tn', 'ts', 'tk', 'tr', 'tw', 'uk',
        'ur', 'vi', 'wo', 'xh', 'yo', 'zu', 'zh', 'zht'],
    'lang_code': 'iso1',
    'entrypoint': 'hugginface.seq2seq',
    'iso1_map': {
        'af': 'afr_Latn', 'ak': 'aka_Latn', 'am': 'amh_Ethi', 'as': 'asm_Beng', 'ba': 'bak_Cyrl', 'bm': 'bam_Latn',
        'be': 'bel_Cyrl', 'bn': 'ben_Beng', 'bs': 'bos_Latn', 'bg': 'bul_Cyrl', 'cs': 'ces_Latn', 'cy': 'cym_Latn',
        'da': 'dan_Latn', 'de': 'deu_Latn', 'dz': 'dzo_Tibt', 'en': 'eng_Latn', 'eo': 'epo_Latn', 'et': 'est_Latn',
        'eu': 'eus_Latn', 'ee': 'ewe_Latn', 'fo': 'fao_Latn', 'fj': 'fij_Latn', 'fi': 'fin_Latn', 'fr': 'fra_Latn',
        'ga': 'gle_Latn', 'gl': 'glg_Latn', 'gn': 'grn_Latn', 'gu': 'guj_Gujr', 'ha': 'hau_Latn', 'he': 'heb_Hebr',
        'hi': 'hin_Deva', 'hr': 'hrv_Latn', 'hu': 'hun_Latn', 'hy': 'hye_Armn', 'ig': 'ibo_Latn', 'id': 'ind_Latn',
        'is': 'isl_Latn', 'it': 'ita_Latn', 'jv': 'jav_Latn', 'ja': 'jpn_Jpan', 'kn': 'kan_Knda', 'ka': 'kat_Geor',
        'kk': 'kaz_Cyrl', 'rw': 'kin_Latn', 'ko': 'kor_Hang', 'lo': 'lao_Laoo', 'ln': 'lin_Latn', 'lt': 'lit_Latn',
        'lg': 'lug_Latn', 'ml': 'mal_Mlym', 'mr': 'mar_Deva', 'mk': 'mkd_Cyrl', 'mt': 'mlt_Latn', 'mi': 'mri_Latn',
        'my': 'mya_Mymr', 'nn': 'nno_Latn', 'ne': 'npi_Deva', 'oc': 'oci_Latn', 'pl': 'pol_Latn', 'pt': 'por_Latn',
        'rn': 'run_Latn', 'ru': 'rus_Cyrl', 'sg': 'sag_Latn', 'sa': 'san_Deva', 'sk': 'slk_Latn', 'sl': 'slv_Latn',
        'sm': 'smo_Latn', 'sn': 'sna_Latn', 'sd': 'snd_Arab', 'so': 'som_Latn', 'st': 'sot_Latn', 'sc': 'srd_Latn',
        'sr': 'srp_Cyrl', 'ss': 'ssw_Latn', 'su': 'sun_Latn', 'sv': 'swe_Latn', 'sw': 'swh_Latn', 'ta': 'tam_Taml',
        'tt': 'tat_Cyrl', 'te': 'tel_Telu', 'tg': 'tgk_Cyrl', 'tl': 'tgl_Latn', 'th': 'tha_Thai', 'ti': 'tir_Ethi',
        'tn': 'tsn_Latn', 'ts': 'tso_Latn', 'tk': 'tuk_Latn', 'tr': 'tur_Latn', 'tw': 'twi_Latn', 'uk': 'ukr_Cyrl',
        'ur': 'urd_Arab', 'vi': 'vie_Latn', 'wo': 'wol_Latn', 'xh': 'xho_Latn', 'yo': 'yor_Latn', 'zu': 'zul_Latn',
        'zh': 'zho_Hans', 'zht': 'zho_Hant'
    }
}

# https://huggingface.co/alirezamsh/small100
small100_model_data = {
    'name': 'alirezamsh/small100',
    'lang_src': [
        'af', 'am', 'ar', 'az', 'ba', 'be', 'bg', 'bn', 'br', 'bs', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et',
        'fa', 'ff', 'fi', 'fr', 'fy', 'ga', 'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'ig', 'is', 'it',
        'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'lg', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms',
        'my', 'ne', 'no', 'or', 'pl', 'pt', 'ru', 'sd', 'sk', 'sl', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta',
        'th', 'tl', 'tn', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh', 'zht', 'zu'],
    'lang_dst': [
        'af', 'am', 'ar', 'az', 'ba', 'be', 'bg', 'bn', 'br', 'bs', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et',
        'fa', 'ff', 'fi', 'fr', 'fy', 'ga', 'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'ig', 'is', 'it',
        'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'lg', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms',
        'my', 'ne', 'no', 'or', 'pl', 'pt', 'ru', 'sd', 'sk', 'sl', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta',
        'th', 'tl', 'tn', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh', 'zht', 'zu'],
    'lang_code': 'iso1',
    'entrypoint': 'hugginface.seq2seq',
    'iso1_map': {
        'zht': 'zh'
    },
}
