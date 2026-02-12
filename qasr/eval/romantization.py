# romanization + tone-ization preprocessing functions
import re

from pypinyin import lazy_pinyin, Style
from ToJyutping import get_jyutping_text
from taibun import Converter

try:
    from .alignment import align_sequences_pinyin_hanzi
except:
    from alignment import align_sequences_pinyin_hanzi


# not a valid character string, mostly for taiwanese
NAVC = 'NAVC1'


def keep_only_chinese(text: str) -> str:
    # Matches all CJK Unified Ideographs + extensions
    pattern = re.compile(r'[^\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]')
    return pattern.sub('', text)


def keep_roman_letters_numbers(text: str) -> str:
    return re.sub(r'[^a-z0-9\s]', NAVC, text.lower())


def get_tone_number(pinyin_syl):
    """
    Extract tone number from Style.TONE3 format
    'ni3' -> '3', 'ni' -> '5' (neutral)
    """
    if pinyin_syl and pinyin_syl[-1].isdigit():
        return int(pinyin_syl[-1])
    return 5  # neutral tone (no number)


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
        return_base=True,
    ):

        self.language = language

        if self.language == 'nan':
            self.converter = Converter(system='TLPA')

        self.style = getattr(Style, style, 'TONE3')
        self.neutral_tone_with_five = neutral_tone_with_five        

        self.keep_only_chinese = keep_only_chinese
        self.keep_only_romanized = keep_only_romanized
        self.break_into_list = break_into_list
        self.return_base = return_base

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

        if self.return_base:
            bases = []
            for syllable in text:
                base = get_base(syllable)
                bases.append(base)
            return text, bases

        return text


def pinyinize_results(references, predictions, pinyinizer: Pinyinizer, language='zh'):
    # results is a dictionary with fields ['references', 'predictions']
    # need to add new fields for pinyin and aligned sequences
    results_temp = {
        'references_pinyin': [],
        'predictions_pinyin': [],
        'references_base': [],
        'predictions_base': [],
        'pinyin_hanzi_ref_pred_pairs': [],
    }

    if language == 'zh':
        results_temp.update({
            'references_initials': [],
            'predictions_initials': [],
            'references_finals': [],
            'predictions_finals': [],
        })


    for ref, pred in zip(references, predictions):
        pinyin_ref, base_ref = pinyinizer(ref)
        pinyin_pred, base_pred = pinyinizer(pred)

        token_pairs = align_sequences_pinyin_hanzi(pinyin_ref, pinyin_pred, ref, pred)

        pinyin_ref_str = ' '.join(pinyin_ref)
        pinyin_pred_str = ' '.join(pinyin_pred)

        base_ref = ' '.join(base_ref)
        base_pred = ' '.join(base_pred)

        results_temp['references_pinyin'].append(pinyin_ref_str)
        results_temp['predictions_pinyin'].append(pinyin_pred_str)

        results_temp['references_base'].append(pinyin_ref_str)
        results_temp['predictions_base'].append(pinyin_pred_str)

        results_temp['pinyin_hanzi_ref_pred_pairs'].append(token_pairs)

        #  initials (声母) and finals (韵母) only applies to mandarin
        if language == 'zh':
            initials_ref = ' '.join(lazy_pinyin(ref, style=Style.INITIALS))
            finals_ref = ' '.join(lazy_pinyin(ref, style=Style.FINALS))

            initials_pred = ' '.join(lazy_pinyin(pred, style=Style.INITIALS))
            finals_pred = ' '.join(lazy_pinyin(pred, style=Style.FINALS))

            results_temp['references_initials'].append(initials_ref)
            results_temp['references_finals'].append(finals_ref)

            results_temp['predictions_initials'].append(initials_pred)
            results_temp['predictions_finals'].append(finals_pred)


    return results_temp


if __name__ == '__main__':
    results = {}

    test_texts = [
        '该  网站  根据  雇员  的  反馈',
        '参与风筝表演的都是世界级顶尖高手',
        '咱们的第八层是属于租的人家的',
        '我有的时候说不清楚你们知道吗',
        '以下是关于农学的单项选择题，请直接给出正确答案的选项。\n题目: 农业税的计税标准，按（）来进行\nA. 当年产量\nB. 三年平均产量\nC. 常年产量\nD. 去年产量\n答案是: '
        '以下是单项选择题，请直接给出正确答案的选项。题目： 明天是二月二十五日,星期三。明天是星期几? 选项： a: 星期二 b: 星期三 c: 星期五 答案是',
        '以下是单项选择题，请直接给出正确答案的选项。题目： 女：下星期我们要去上海旅游，你去吗？ 男：太好了！我也去。 问：男的是什么意思？ 选项： a: 他也去 b: 他不去 c: 他去过了 答案是',
        '以下是单项选择题，请直接给出正确答案的选项。题目： 女：你的耳朵和鼻子都红了。 男：是啊，太冷了。我要买个帽子。 问：男的怎么了？ 选项： a: 口渴 b: 生病了 c: 觉得很冷 答案是',
        '以下是单项选择题，请直接给出正确答案的选项。题目： 男：我上午发的那份传真你收到了吧？ 女：没收到。等等，我看一下，抱歉，没纸了，麻烦您再发一遍吧。 问：女的为什么没收到传真？ 选项： a: 没纸了 b: 男的没发 c: 打印机坏了 d: 传真机坏了 答案是',
        '以下是单项选择题，请直接给出正确答案的选项。题目： 女：明天上午9点我准时到。 男：我觉得还是提前几分钟吧。 问：男的主要是什么意思？ 选项： a: 9点太早了 b: 他不会迟到 c: 可能不参加 d: 应该早点儿来 答案是',
        '以下是单项选择题，请选出跟题目内容一样的选项。题目： 一个年轻人坐在公园的长椅上休息，有个小孩儿站在他旁边很久，一直不走。年轻人问：“小朋友，你为什么站在这里不走，有什么事儿吗？”小孩儿说：“这个椅子刚刷了油漆，我想看看你站起来会是什么样子。” 选项： a: 年轻人在睡觉 b: 椅子是小孩的 c: 小孩想坐那个椅子 d: 椅子上刚刷了油漆 答案是',
        '你報名了嗎',
        '在家也可以刷卡',
        '就買好一點的給孩子吃吧',
        '遲遲未定的原因',
        '大屯橋（T?a-t?n-ki?）',
        '我沒咧驚（gu? b? leh kiann）',
        '你知道嗎？其實每天早上花十分鐘做瑜伽，整天的心情都會變得很不一樣喔！',
        '有一次我跑到半路突然下大雨，雖然全身濕透，但那感覺反而讓我更覺得活著。',
        '杞人嘅朋友嘆咗一口氣',
        '黑身准裂腹鱼为辐鳍鱼纲鲤形目鲤科的其中一种。',
        'なに喜んでるんだ、しもべのくせに…これはもっと、お仕置きが必要だな…\n',
        '''
        賴總統表示，上週第6屆台美經濟繁榮夥伴對話（EPPD），在美國華府舉行。這個台美經濟對話機制是在美國總統川普第一任任期內所創立，上週的會議則是 川普在第二任期，第一次召開EPPD會議，並且是2020年首屆EPPD以來，雙方主談人首度實體面對面對談，對台美關係來說，具有重大意義。
        賴總統談到台美未來合作的三大戰略方向，包括，第一，強化「經濟安全」。面對地緣政治局勢變動，AI與半導體等高科技產業，越來越重視安全、可信與韌性，並加速建構具備韌性的非紅供應鏈。
        ''',
    ]

    pinyinizer_zh = Pinyinizer(language='zh', return_base=False)
    pinyinizer_yue = Pinyinizer(language='yue', return_base=False)
    pinyinizer_nan = Pinyinizer(language='nan', return_base=False)

    for text in test_texts:
        print('Original: ', text)

        pinyin_zh = pinyinizer_zh(text)
        print(f'Mandarin: {len(pinyin_zh)}', pinyin_zh)

        pinyin_yue = pinyinizer_yue(text)
        print(f'Cantonese: {len(pinyin_yue)}', pinyin_yue)

        pinyin_nan = pinyinizer_nan(text)
        print(f'Taiwanese: {len(pinyin_nan)}', pinyin_nan)

    import json

    with open('assets/transcript_qwen_aishell1.jsonl', 'r') as json_file:
        json_list = list(json_file)

    results = {'references': [], 'predictions': []}
    for json_str in json_list:
        result = json.loads(json_str)
        results['references'].append(result['text'])
        results['predictions'].append(result['pred_text'])
        # print(result)

    print(len(results), len(results['references']))

    language = 'zh'
    pinyinizer = Pinyinizer(language=language)
    results_pinyin = pinyinize_results(results['references'], results['predictions'],
                                       pinyinizer, language=language)
    print(results_pinyin.keys(), len(results_pinyin['pinyin_hanzi_ref_pred_pairs']))

