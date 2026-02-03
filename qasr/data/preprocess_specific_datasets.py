import re
import json


def clean_speech_hsk_asr(example):
    # 1. Store the original text as requested
    example['text_og'] = example['text']

    # 2. Extract the core text from the 'question' field
    # (The 'text' field contains prompt instructions we don't want for ASR)
    asr_text = example['text']

    # 3. Clean speaker labels (女：, 男：, 问：) and their variants (full/half-width colons)
    # This covers HSK 2, 3, 4, and 5
    asr_text = re.sub(r'(女|男|问)[:：]', '', asr_text)

    # 4. Handle HSK 6 and general punctuation cleaning
    # If it's pure ASR, we usually strip quotes and potentially ending punctuation
    # depending on your ASR target format.
    # asr_text = asr_text.replace('“', '').replace('”', '').replace('"', '')

    # 5. Final cleanup: Remove double spaces and strip leading/trailing whitespace
    asr_text = "".join(asr_text.split()).strip()

    # Update the text field
    example['text'] = asr_text

    return example


def clean_hokkien_asr(example):
    # 1. Remove text inside parentheses (handles both full-width and half-width)
    # This matches (brackets), （brackets）, and mixed pairs
    clean_text = re.sub(r'[\(\（].*?[\)\）]', '', example['txt'])

    # 2. Remove any trailing punctuation or whitespace
    # For Hokkien ASR, usually, we just want the characters
    clean_text = clean_text.strip()

    # Update the text field
    example['text'] = clean_text

    return example


def clean_air_chat_asr(example):
    try:
        # 1. Parse the string into a list of dictionaries
        segments = json.loads(example['meta_info'])

        # 2. Extract the 'transcription' from each segment
        # We use .get() as a safety measure
        transcriptions = [s.get('transcription', '') for s in segments]

        # 3. Concatenate with a space
        full_text = " ".join(transcriptions)

        # 4. Optional: Clean up filler tags like [laughter] or [noise]
        # commonly found in conversational datasets
        full_text = re.sub(r'\[.*?\]', '', full_text)

        # Clean up extra spaces created by tag removal
        example['text'] = " ".join(full_text.split()).strip()

    except (json.JSONDecodeError, TypeError):
        # Fallback if the meta_info is malformed
        example['text'] = ""

    return example


def preprocess_dataset(dataset, dataset_path):
    if dataset_path == 'TwinkStart/air-chat':
        dataset = dataset.map(clean_air_chat_asr)
    elif dataset_path == 'TwinkStart/speech-HSK':
        dataset = dataset.map(clean_speech_hsk_asr)
    elif dataset_path == 'adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hokkien':
        dataset = dataset.map(clean_hokkien_asr)
    return dataset


if __name__ == '__main__':
    # if chinese hokkien need to decide if chinese characters or pinyin
    # eg 我沒咧驚（gu? b? leh kian)
    # if hsk2/3/4/5 need to remove the "女：", "男：", "问：" from 'text' fields
    # if hsk6 need to eliminate the punctuation such as "样子。”" in the text

    # for air-chat need to concatenate "transcription" key from a list of dictionaries in meta_info
    # {'meta_info': '[{"speaker": "speaker 1", "speaking time": "(0.00,13.39)", "transcription": "so it\'s a lot more than the like two thousand dollar computers that like you or i can afford you know [laughter] so ah they so they put a lot of effort and this is the kind of stuff that gets sold to the government"}, {"speaker": "speaker 2", "speaking time": "(5.33,6.88)", "transcription": "right right"}, {"speaker": "speaker 2", "speaking time": "(13.01,13.91)", "transcription": "mhm"}, {"speaker": "speaker 1", "speaking time": "(13.30,20.20)", "transcription": "and that\'s why and our government has a massive sum of money and that\'s why i"}, {"speaker": "speaker 2", "speaking time": "(19.29,21.60)", "transcription": "isn\'t it amazing how it has"}, {"speaker": "speaker 1", "speaking time": "(21.30,26.59)", "transcription": "i mean they they ah how what\'s the national debt six trillion plus"}, {"speaker": "speaker 2", "speaking time": "(25.88,28.08)", "transcription": "[laughter]"}]'

    sample = {'meta_info': '[{"speaker": "speaker 1", "speaking time": "(0.00,13.39)", "transcription": "so it\'s a lot more than the like two thousand dollar computers that like you or i can afford you know [laughter] so ah they so they put a lot of effort and this is the kind of stuff that gets sold to the government"}, {"speaker": "speaker 2", "speaking time": "(5.33,6.88)", "transcription": "right right"}, {"speaker": "speaker 2", "speaking time": "(13.01,13.91)", "transcription": "mhm"}, {"speaker": "speaker 1", "speaking time": "(13.30,20.20)", "transcription": "and that\'s why and our government has a massive sum of money and that\'s why i"}, {"speaker": "speaker 2", "speaking time": "(19.29,21.60)", "transcription": "isn\'t it amazing how it has"}, {"speaker": "speaker 1", "speaking time": "(21.30,26.59)", "transcription": "i mean they they ah how what\'s the national debt six trillion plus"}, {"speaker": "speaker 2", "speaking time": "(25.88,28.08)", "transcription": "[laughter]"}]', 'question': 'What does the first speaker imply about the national debt?', 'answer_gt': "That it's more than six trillion dollars.", 'path': '568.38_596.46.wav', 'task_name': 'speech_dialogue_QA', 'dataset_name': 'fisher', 'uniq_id': 0, 'WavPath': 'speech_dialogue_QA_fisher/568.38_596.46.wav',}
    sample = clean_air_chat_asr(sample)
    print(sample)

    sample = {'question': '女：下星期我们要去上海旅游，你去吗？ 男：太好了！我也去。 问：男的是什么意思？', 'choices': ['他也去', '他不去', '他去过了'], 'ans_choice': 'A', 'ans': '他也去', 'WavPath': 'audio/2/21.wav', 'choices_path': ['audio/2/21_a.wav', 'audio/2/21_b.wav', 'audio/2/21_c.wav'], 'question_path': 'audio/2/21_question.wav', 'text': '以下是单项选择题，请直接给出正确答案的选项。题目： 女：下星期我们要去上海旅游，你去吗？ 男：太好了！我也去。 问：男的是什么意思？ 选项： a: 他也去 b: 他不去 c: 他去过了 答案是'}
    sample = clean_speech_hsk_asr(sample)
    print(sample)

    sample = {'question': '女：明天上午9点我准时到。 男：我觉得还是提前几分钟吧。 问：男的主要是什么意思？', 'choices': ['9点太早了', '他不会迟到', '可能不参加', '应该早点儿来'], 'ans_choice': 'D', 'ans': '应该早点儿来', 'WavPath': 'audio/5/1.wav', 'choices_path': ['audio/5/1_a.wav', 'audio/5/1_b.wav', 'audio/5/1_c.wav', 'audio/5/1_d.wav'], 'question_path': 'audio/5/1_question.wav', 'text': '以下是单项选择题，请直接给出正确答案的选项。题目： 女：明天上午9点我准时到。 男：我觉得还是提前几分钟吧。 问：男的主要是什么意思？ 选项： a: 9点太早了 b: 他不会迟到 c: 可能不参加 d: 应该早点儿来 答案是'}
    sample = clean_speech_hsk_asr(sample)
    print(sample)

    sample = {'txt': '我沒咧驚（gu? b? leh kiann）', 'json': {'age': 'twenties', 'audio_ext': '.mp3', 'client_id': '8c4a30530366c24080fe985b3a38959b60d594c29d822d780a8ff483f18f8bb8410b1149fa893d0c1df74c2b29ad19471caaf080dcea4ffa9a1bc4fea7e7625a', 'gender': 'male_masculine', 'locale': '', 'path': 'clips/common_voice_nan-tw_32158967.mp3', 'sentence': '我沒咧驚（gu? b? leh kiann）', 'sentence_id': '4743279fc8d77049056498d9aa6f68effea7824bbdbca066bf0026e5e522e4d3', 'split': 'test'}, '__key__': 'test-000000000', '__url__': 'hf://datasets/adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hokkien@ffcc36b9afe31ec50f7742c0c4c614f1a5b40151/test/test-000000.tar'}
    sample = clean_hokkien_asr(sample)
    print(sample)

