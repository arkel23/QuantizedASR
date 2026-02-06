import argparse

import opencc
from pypinyin import lazy_pinyin, Style

try:
    from tn.chinese.normalizer import Normalizer
except:
    print('WeTextProcessing not installed')

try:
    from .normalizer_zh_speechio import TextNorm
    from .normalizer_zh_ultraeval import TextNormUltraEval
    from .normalizer_zh_ntnu import normalize_corpus
    from .normalizer_zh_aditw import ADITWNormalizer
except:
    from normalizer_zh_speechio import TextNorm
    from normalizer_zh_ultraeval import TextNormUltraEval
    from normalizer_zh_ntnu import normalize_corpus
    from normalizer_zh_aditw import ADITWNormalizer


class Pinyinizer:
    def __init__(
        self,
        style = 'TONE3',
        neutral_tone_with_five: bool = False,
    ):

        self.style = getattr(Style, style, 'TONE3')
        self.neutral_tone_with_five = neutral_tone_with_five        

    def __call__(self, text):
        pinyin = lazy_pinyin(
            text,
            style=self.style,
            neutral_tone_with_five=self.neutral_tone_with_five,
        )
        text = ' '.join(pinyin)
        return text


class ChineseNormalizer:
    def __init__(
        self,
        traditional = False,
        pinyin = False,
        style = 'TONE3',
        neutral_tone_with_five: bool = False,
    ):

        self.normalizer = TextNormUltraEval(
            to_banjiao=True,
            to_upper=True,
            to_lower=False,
            remove_fillers=False,
            remove_erhua=False,
            check_chars=False,
            remove_space=True,
            cc_mode='s2t' if traditional else 't2s',
        )

        self.pinyin = pinyin
        self.style = getattr(Style, style, 'TONE3')
        self.neutral_tone_with_five = neutral_tone_with_five        

    def __call__(self, text):
        text = self.normalizer(text)

        if self.pinyin:
            text = lazy_pinyin(
                text,
                style=self.style,
                neutral_tone_with_five=self.neutral_tone_with_five,
            )
            text = ' '.join(text)

        return text

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # normalizer options
    p.add_argument('--to_banjiao', action='store_false', help='convert quanjiao chars to banjiao')
    p.add_argument('--to_upper', action='store_false', help='convert to upper case')
    p.add_argument('--to_lower', action='store_true', help='convert to lower case')
    p.add_argument('--remove_fillers', action='store_true', help='remove filler chars such as "呃, 啊"')
    p.add_argument('--remove_erhua', action='store_true', help='remove erhua chars such as "他女儿在那边儿 -> 他女儿在那边"')
    p.add_argument('--check_chars', action='store_true' , help='skip sentences containing illegal chars')
    p.add_argument('--remove_space', action='store_false' , help='remove whitespace')
    p.add_argument('--cc_mode', choices=['', 't2s', 's2t'], default='s2t', help='convert between traditional to simplified')

    args = p.parse_args()

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


    normalizer = TextNorm(
        to_banjiao = args.to_banjiao,
        to_upper = args.to_upper,
        to_lower = args.to_lower,
        remove_fillers = args.remove_fillers,
        remove_erhua = args.remove_erhua,
        check_chars = args.check_chars,
        remove_space = args.remove_space,
        cc_mode = args.cc_mode,
    )

    normalizer_ue = TextNormUltraEval(
        to_banjiao = args.to_banjiao,
        to_upper = args.to_upper,
        to_lower = args.to_lower,
        remove_fillers = args.remove_fillers,
        remove_erhua = args.remove_erhua,
        check_chars = args.check_chars,
        remove_space = args.remove_space,
        cc_mode = args.cc_mode,
    )

    normalizer_aditw = ADITWNormalizer()

    try:
        normalizer_wetext = Normalizer(
            remove_interjections=False,
            remove_erhua=False,
            traditional_to_simple=True,
            remove_puncts=True,
            full_to_half=True,
        )
    except:
        normalizer_wetext = normalizer

    cc_mode = f'{args.cc_mode}.json' if args.cc_mode else 's2t.json'
    converter = opencc.OpenCC(cc_mode)

    pinyinizer = Pinyinizer()

    # ultraeval-audio applies another step after the chinese normalizer if cantonese (yue)
    # https://github.com/OpenBMB/UltraEval-Audio/blob/7028c39e4209163159cccf02c097a6e771b844ac/audio_evals/lib/wer.py#L10

    for text in test_texts:
        print('Original text: ')
        print(text)

        text_conv = converter.convert(text)
        print('Converted text: ', text_conv)

        text_norm_ntnu = normalize_corpus(text, is_remove_numbers=False, is_remove_alphabets=False)
        print('NTNU normalization: ')
        print(text_norm_ntnu)

        text_norm_speechio = normalizer(text)
        print('SpeechIO normalization: ')
        print(text_norm_speechio)

        text_norm_ue = normalizer_ue(text)
        print('UltraEval normalization: ')
        print(text_norm_ue)

        text_norm_aditw = normalizer_aditw(text)
        print('ADI-TW normalization: ')
        print(text_norm_aditw)

        text_norm_wetext = normalizer_wetext.normalize(text)
        print('WeText normalization: ')
        print(text_norm_wetext)

        # https://github.com/mozillazg/python-pinyin
        # text_norm_pinyin = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
        # 不考虑多音字的情况
        text_pinyin = pinyinizer(text)
        print('Pinyin without normalization: ')
        print(text_pinyin)

        text_pinyin_norm_ntnu =pinyinizer(text_norm_ntnu)
        print('Pinyin with NTNU normalization: ')
        print(text_pinyin_norm_ntnu)

        text_pinyin_norm_speechio = pinyinizer(text_norm_speechio)
        print('Pinyin with SpeechIO normalization: ')
        print(text_pinyin_norm_speechio)

        text_pinyin_norm_ue = pinyinizer(text_norm_ue)
        print('Pinyin with UltraEval normalization: ')
        print(text_pinyin_norm_ue)

        text_pinyin_norm_aditw = pinyinizer(text_norm_aditw)
        print('Pinyin with ADI-TW normalization: ')
        print(text_pinyin_norm_aditw)

        text_pinyin_norm_wetext = pinyinizer(text_norm_wetext)
        print('Pinyin with WeText normalization: ')
        print(text_pinyin_norm_wetext)
