# romanization + tone-ization preprocessing functions
import re

from pypinyin import lazy_pinyin, Style
from ToJyutping import get_jyutping_text
from taibun import Converter


def keep_only_chinese(text: str) -> str:
    # Matches all CJK Unified Ideographs + extensions
    pattern = re.compile(r'[^\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]')
    return pattern.sub('', text)


def keep_roman_letters_numbers(text: str) -> str:
    return re.sub(r'[^a-z0-9\s]', '', text.lower())


def get_tone_number(pinyin_syl):
    """
    Extract tone number from Style.TONE3 format
    'ni3' -> '3', 'ni' -> '5' (neutral)
    """
    if pinyin_syl and pinyin_syl[-1].isdigit():
        return pinyin_syl[-1]
    return '5'  # neutral tone (no number)


def get_base(pinyin_syl):
    """
    Extract base without tone from Style.TONE3 format
    'ni3' -> 'ni', 'hao3' -> 'hao'
    """
    if pinyin_syl and pinyin_syl[-1].isdigit():
        return pinyin_syl[:-1]
    return pinyin_syl


class Pinyinizer:
    def __init__(
        self,
        language='zh',
        style = 'TONE3',
        neutral_tone_with_five: bool = True,
        keep_only_chinese=True,
        keep_only_romanized=True,
        break_into_list=True,
    ):

        self.language = language

        if self.language == 'nan':
            self.converter = Converter(system='TLPA')

        self.style = getattr(Style, style, 'TONE3')
        self.neutral_tone_with_five = neutral_tone_with_five        

        self.keep_only_chinese = keep_only_chinese
        self.keep_only_romanized = keep_only_romanized
        self.break_into_list = break_into_list

    def __call__(self, text):
        if self.keep_only_chinese:
            text = keep_only_chinese(text)

        if self.language == 'zh':
            pinyin = lazy_pinyin(
                text,
                style=self.style,
                neutral_tone_with_five=self.neutral_tone_with_five,
            )
            text = ' '.join(pinyin)

        elif self.language == 'yue':
            # ultraeval does additional preprocessing of cantonese
            # using zhconv, unknown purpose (standarize to zh-cn?)
            # https://github.com/OpenBMB/UltraEval-Audio/blob/7028c39e4209163159cccf02c097a6e771b844ac/audio_evals/lib/wer.py#L47
            text = get_jyutping_text(text)

        # minnan -> nan-tw
        elif self.language == 'nan':
            # https://github.com/andreihar/taibun
            text = self.converter.get(text)

        # hakka -> hak
        elif self.language == 'hak':
            # may have to crawl from here
            # https://hakkadict.moe.edu.tw/
            raise NotImplementedError('Hakka romanization not implemented')

        # thailandese
        elif self.language == 'th':
            #  pythainlp.transliterate with engine=tltk_g2p, which converts Thai script to
            # Latin characters with explicit tone marking (numbers 0–4).
            raise NotImplementedError('Thai romanization not implemented')

        # vietnamese
        elif self.language == 'vi':
            # https://github.com/kirbyj/vPhon
            raise NotImplementedError('Vietnamese romanization not implemented')

        if self.keep_only_romanized:
            text = keep_roman_letters_numbers(text)

        text = text.split(' ')

        return text



if __name__ == '__main__':
    results = {}

    results['references'] = ["this is the reference", "there is another one", "能吞虾玻璃而不霜身体啦", "我发现了问题", "你好世界"]
    results['predictions'] = ["this is the prediction", "there is an other sample", "我能吞下玻璃而不伤身体", "我发线了问题", "你号世界"]
    print(results)

    for ref, pred in zip(results['references'], results['predictions']):
        pass
        # token_pairs = align_sequences(ref, pred)
        # print(token_pairs)
