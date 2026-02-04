# https://github.com/mozillazg/python-pinyin
from pypinyin import lazy_pinyin, Style
import opencc

from .normalizer_zh_speechio import TextNorm
from .normalizer_zh_ntnu import normalize_corpus


if __name__ == '__main__':

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
        to_banjiao=True,
        to_lower=True,
        remove_fillers=False,
        remove_erhua=False,
        check_chars=True,
        remove_space=True,
        cc_mode='', # also s2t or t2s
    )

    # converter = opencc.OpenCC('s2t.json')
    # converter.convert('汉字')  # 漢字

    for text in test_texts:
        print('Original text: ')
        print(text)

        norm_ntnu = normalize_corpus(text, is_remove_numbers=False, is_remove_alphabets=False)
        print('NTNU normalization: ')
        print(norm_ntnu)

        norm_speechio = normalizer(text)
        print('SpeechIO normalization: ')
        print(norm_speechio)

        # norm_pinyin = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
        # 不考虑多音字的情况
        text_pinyin = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
        print('Pinyin without normalization: ')
        print(text_pinyin)

        pinyin_norm_ntnu = ' '.join(lazy_pinyin(norm_ntnu, style=Style.TONE3, neutral_tone_with_five=True))
        print('Pinyin with NTNU normalization: ')
        print(pinyin_norm_ntnu)

        pinyin_norm_speechio = ' '.join(lazy_pinyin(norm_speechio, style=Style.TONE3, neutral_tone_with_five=True))
        print('Pinyin with SpeechIO normalization: ')
        print(pinyin_norm_speechio)
