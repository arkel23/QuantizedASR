import argparse

import opencc
from pypinyin import lazy_pinyin, Style

try:
    from .normalizer_zh_ultraeval import TextNormUltraEval
except:
    from normalizer_zh_ultraeval import TextNormUltraEval


class Pinyinizer:
    def __init__(
        self,
        style = 'TONE3',
        neutral_tone_with_five: bool = True,
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
            to_upper=False,
            to_lower=True,
            remove_fillers=False,
            remove_erhua=True,
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
    # import wetext when working with hf results in a lot of debugging logs while reading data
    from tn.chinese.normalizer import Normalizer

    from normalizer_zh_speechio import TextNorm
    # from normalizer_zh_ultraeval import TextNormUltraEval
    from normalizer_zh_ntnu import normalize_corpus
    from normalizer_zh_aditw import ADITWNormalizer

    p = argparse.ArgumentParser()

    # normalizer options
    p.add_argument('--main_normalizer_only', action='store_false')
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

        '以下是单项选择题 请选出跟题目内容一样的选项 题目 很多父母习惯只给婴儿吃少数几种食物 但专家表示 大多数婴儿六个月大时就可以安全进食多种食物 他们认为给孩子提供多样化食物有好处 可以帮助他们长大后适应不同种类的食品 选项 a 婴儿应该少吃b 婴儿应多喝牛奶c 婴儿食物应该多样化d 许多婴儿爱吃一种食物答案是',
        '以下是单项选择题 请选出跟题目内容一样的选项 题目 一个年轻人坐在公园的长椅上休息 有个小孩儿站在他旁边很久 一直不走 年轻人 小朋友 你为什么站在这里不走 有什么事儿吗 小孩儿说 这个椅子刚刷了油漆 我想看看你站起来会是什么样子 选项 a 年轻人在睡觉b 椅子是小孩的c 小孩想坐那个椅子d 椅子上刚刷了油漆答案是',

        # https://github.com/OpenBMB/UltraEval-Audio/blob/main/audio_evals/lib/index-tts2/indextts/utils/front.py
        "IndexTTS 正式发布1.0版本了，效果666",
        "晕XUAN4是一种GAN3觉",
        "我爱你！",
        "I love you!",
        "“我爱你”的英语是“I love you”",
        "2.5平方电线",
        "共465篇，约315万字",
        "2002年的第一场雪，下在了2003年",
        "速度是10km/h",
        "现在是北京时间2025年01月11日 20:00",
        "他这条裤子是2012年买的，花了200块钱",
        "电话：135-4567-8900",
        "1键3连",
        "他这条视频点赞3000+，评论1000+，收藏500+",
        "这是1024元的手机，你要吗？",
        "受不liao3你了",
        "“衣裳”不读衣chang2，而是读衣shang5",
        "最zhong4要的是：不要chong2蹈覆辙",
        "不zuo1死就不会死",
        "See you at 8:00 AM",
        "8:00 AM 开会",
        "Couting down 3, 2, 1, go!",
        "数到3就开始：1、2、3",
        "This sales for 2.5% off, only $12.5.",
        "5G网络是4G网络的升级版，2G网络是3G网络的前身",
        "苹果于2030/1/2发布新 iPhone 2X 系列手机，最低售价仅 ¥12999",
        "这酒...里...有毒...",
        # 异常case
        "只有,,,才是最好的",
        "babala2是什么？",  # babala二是什么?
        "用beta1测试",  # 用beta一测试
        "have you ever been to beta2?",  # have you ever been to beta two?
        "such as XTTS, CosyVoice2, Fish-Speech, and F5-TTS",  # such as xtts,cosyvoice two,fish-speech,and f five-tts
        "where's the money?",  # where is the money?
        "who's there?",  # who is there?
        "which's the best?",  # which is the best?
        "how's it going?",  # how is it going?
        "今天是个好日子 it's a good day",  # 今天是个好日子 it is a good day
        # 人名
        "约瑟夫·高登-莱维特（Joseph Gordon-Levitt is an American actor）",
        "蒂莫西·唐纳德·库克（英文名：Timothy Donald Cook），通称蒂姆·库克（Tim Cook），美国商业经理、工业工程师和工业开发商，现任苹果公司首席执行官。",
        # 长句子
        "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗·诺兰执导并编剧，莱昂纳多·迪卡普里奥、玛丽昂·歌迪亚、约瑟夫·高登-莱维特、艾利奥特·佩吉、汤姆·哈迪等联袂主演，2010年7月16日在美国上映，2010年9月1日在中国内地上映，2020年8月28日在中国内地重映。影片剧情游走于梦境与现实之间，被定义为“发生在意识结构内的当代动作科幻片”，讲述了由莱昂纳多·迪卡普里奥扮演的造梦师，带领特工团队进入他人梦境，从他人的潜意识中盗取机密，并重塑他人梦境的故事。",
        "清晨拉开窗帘，阳光洒在窗台的Bloomixy花艺礼盒上——薰衣草香薰蜡烛唤醒嗅觉，永生花束折射出晨露般光泽。设计师将“自然绽放美学”融入每个细节：手工陶瓷花瓶可作首饰收纳，香薰精油含依兰依兰舒缓配方。限量款附赠《365天插花灵感手册》，让每个平凡日子都有花开仪式感。\n宴会厅灯光暗下的刹那，Glimmeria星月系列耳坠开始发光——瑞士冷珐琅工艺让蓝宝石如银河流动，钛合金骨架仅3.2g无负重感。设计师秘密：内置微型重力感应器，随步伐产生0.01mm振幅，打造“行走的星光”。七夕限定礼盒含星座定制铭牌，让爱意如星辰永恒闪耀。",
        "电影1：“黑暗骑士”（演员：克里斯蒂安·贝尔、希斯·莱杰；导演：克里斯托弗·诺兰）；电影2：“盗梦空间”（演员：莱昂纳多·迪卡普里奥；导演：克里斯托弗·诺兰）；电影3：“钢琴家”（演员：艾德里安·布洛迪；导演：罗曼·波兰斯基）；电影4：“泰坦尼克号”（演员：莱昂纳多·迪卡普里奥；导演：詹姆斯·卡梅隆）；电影5：“阿凡达”（演员：萨姆·沃辛顿；导演：詹姆斯·卡梅隆）；电影6：“南方公园：大电影”（演员：马特·斯通、托马斯·艾恩格瑞；导演：特雷·帕克）",
    ]

    if args.main_normalizer_only:
        normalizer = ChineseNormalizer()

        for text in test_texts:
            print('Original text: ')
            print(text)

            text_norm = normalizer(text)
            print('Normalized text: ')
            print(text_norm)

    else:
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
