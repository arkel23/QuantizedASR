# https://github.com/andreihar/taibun

import re

import jiwer
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


def align_sequences(ref_syls, hyp_syls):
    """
    Syllable-level alignment using jiwer.process_words()

    Returns: List of tuples (ref_syl, hyp_syl)
    - (syl, syl) for matches/substitutions
    - (syl, None) for deletions
    - (None, syl) for insertions
    """
    # Join syllables into space-separated strings
    ref_str = ' '.join(ref_syls)
    hyp_str = ' '.join(hyp_syls)

    # Get alignment using process_words
    output = jiwer.process_words(ref_str, hyp_str)

    # output.alignments is: List[List[AlignmentChunk]]
    # For single sentence pair, use [0]
    alignment_chunks = output.alignments[0]

    # Convert AlignmentChunk objects to (ref, hyp) tuples
    token_pairs = []

    for chunk in alignment_chunks:
        # chunk has: type, ref_start_idx, ref_end_idx, hyp_start_idx, hyp_end_idx
        # type is: 'equal', 'substitute', 'delete', or 'insert'

        r_span = ref_syls[chunk.ref_start_idx:chunk.ref_end_idx]
        h_span = hyp_syls[chunk.hyp_start_idx:chunk.hyp_end_idx]

        if chunk.type == 'equal' or chunk.type == 'substitute':
            # Both sequences have elements
            token_pairs.extend(zip(r_span, h_span))

        # elif chunk.type == "substitute":
        #     for i in range(max(len(r_span), len(h_span))):
        #         r = r_span[i] if i < len(r_span) else None
        #         h = h_span[i] if i < len(h_span) else None
        #         token_pairs.append((r, h))

        elif chunk.type == "delete":
            # Only ref has element
            for r in r_span:
                token_pairs.append((r, None))

        elif chunk.type == "insert":
            # Only hyp has element
            for h in h_span:
                token_pairs.append((None, h))

    return token_pairs


def compute_ter(references, predictions):
    """
    Tone Error Rate: % of syllables with correct base but wrong tone
    """
    total_matched_base = 0
    total_tone_errors = 0

    for ref_text, pred_text in zip(references, predictions):
        # Convert to pinyin with tones
        # ref_pinyin = lazy_pinyin(ref_text, style=Style.TONE3)
        # pred_pinyin = lazy_pinyin(pred_text, style=Style.TONE3)

        # Align syllables
        aligned = align_sequences(ref_text, pred_text)

        for ref_syl, pred_syl in aligned:
            if ref_syl is None or pred_syl is None:
                continue  # Skip insertions/deletions

            # Split base and tone
            ref_base = get_base(ref_syl)
            ref_tone = get_tone_number(ref_syl)
            pred_base = get_base(pred_syl)
            pred_tone = get_tone_number(pred_syl)

            # Count if bases match
            if ref_base == pred_base:
                total_matched_base += 1
                if ref_tone != pred_tone:
                    total_tone_errors += 1

    return total_tone_errors / max(total_matched_base, 1)


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
