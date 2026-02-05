# https://github.com/adi-gov-tw/Taiwan-Tongues-ASR-CE/blob/main/asr_core.py
import re
import unicodedata
from datetime import datetime, timedelta

import opencc
import cn2an


def replace_words(article):
    mappings = {
        "百分之十五": "15%",
        "百分之五": "5%",
        "百分之十二點五": "12.5%",
        "百分之七": "7%",
        "零八零零零九五九八": "080009598",
    }
    replaced_article = article
    for old, new in mappings.items():
        replaced_article = replaced_article.replace(old, new)
    return replaced_article


def full_to_half(text):
    half_width_text = ""
    for char in text:
        half_char = unicodedata.normalize("NFKC", char)
        if half_char.isalpha():
            half_char = half_char
        half_width_text += half_char
    return half_width_text


def remove_special_characters_by_dataset_name(text):
    # 移除特殊字符
    chars_to_ignore_regex_base = r'[,"\'。，^¿¡；「」《》:：＄$\[\]〜～·・‧―─–－⋯、＼【】=<>{}_〈〉　）（—『』«»→„…(),`&＆﹁﹂#＃\\!?！;]'

    sentence = re.sub(chars_to_ignore_regex_base, "", text)
    return sentence


def split_sentence_to_words(text: str, is_split: bool):
    if is_split is False:
        return text
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF%]|\d+\.\d+|\d+)"
    )
    chars = pattern.split(text.strip().lower())
    return " ".join([w.strip() for w in chars if w is not None and w.strip()])


def convert_time(time):
    time_str = f"{time:.3f}"
    if "." in time_str:
        seconds, millisecond = time_str.split(".")
    else:
        time = time_str
        millisecond = "000"

    delta = timedelta(seconds=int(seconds))
    time_str = (datetime.min + delta).strftime("%H:%M:%S")

    t = str(time_str).split(":")
    return f"{':'.join([x.zfill(2) for x in t])}.{millisecond}"


def num_to_cn(text, mode=0):
    method = "an2cn" if mode == 0 else "cn2an"
    text = cn2an.transform(text, method)
    return text


class ADITWNormalizer:
    def __init__(
        self,
        replace_words: bool = True,
        convert_to_zh_tw: bool = True,
        remove_special_characters: bool = True,
        to_banjiao: bool = True,
        to_lower: bool = False,
        to_upper: bool = True,
        convert_numbers: bool = False,
    ):

        self.replace_words = replace_words
        self.convert_to_zh_tw = convert_to_zh_tw
        self.remove_special_characters = remove_special_characters
        self.to_banjiao = to_banjiao
        self.to_lower = to_lower
        self.to_upper = to_upper
        self.convert_numbers = convert_numbers

        if convert_to_zh_tw:
            self.cc = opencc.OpenCC("s2tw")


    def __call__(self, text):
        if self.replace_words:
            text = replace_words(text)

        if self.convert_to_zh_tw:
            text = self.cc.convert(text)

        if self.remove_special_characters:
            text = remove_special_characters_by_dataset_name(text)

        if self.to_banjiao:
            text = full_to_half(text)

        if self.to_lower:
            text = text.lower()

        if self.to_upper:
            text = text.upper()

        if self.convert_numbers:
            text = num_to_cn(text, mode=0)

        return text


if __name__ == '__main__':
    text = '该  网站  根据  雇员  的  反馈'
    print(text)

    normalizer = ADITWNormalizer()

    text_norm = normalizer(text)
    print(text_norm)
