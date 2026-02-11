from typing import List, Tuple, Optional

import jiwer


def align_sequences(
        ref_tokens: List[str],
        hyp_tokens: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Syllable-level alignment using jiwer.process_words()

    Returns: List of tuples (ref_syl, hyp_syl)
    - (syl, syl) for matches/substitutions
    - (syl, None) for deletions
    - (None, syl) for insertions
    """
    # Join syllables into space-separated strings
    ref_str = ' '.join(ref_tokens)
    hyp_str = ' '.join(hyp_tokens)

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

        r_span = ref_tokens[chunk.ref_start_idx:chunk.ref_end_idx]
        h_span = hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx]

        if chunk.type == 'equal' or chunk.type == 'substitute':
            # Handle both same-length and different-length spans
            if len(r_span) == len(h_span):
                token_pairs.extend(zip(r_span, h_span))
            else:
                # Different lengths: pad with None
                for i in range(max(len(r_span), len(h_span))):
                    r = r_span[i] if i < len(r_span) else None
                    h = h_span[i] if i < len(h_span) else None
                    token_pairs.append((r, h))

        elif chunk.type == "delete":
            # Only ref has element
            token_pairs.extend(zip(r_span, [None] * len(r_span)))
            # for r in r_span:
            #     token_pairs.append((r, None))

        elif chunk.type == "insert":
            # Only hyp has element
            token_pairs.extend(zip([None] * len(h_span), h_span))
            # for h in h_span:
            #     token_pairs.append((None, h))

    return token_pairs


def align_sequences_pinyin_hanzi(
        ref_tokens: List[str],
        hyp_tokens: List[str],
        ref_hanzi: List[str],
        hyp_hanzi: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Syllable-level alignment using jiwer.process_words()

    Returns: List of tuples (ref_syl, hyp_syl)
    - (syl, syl) for matches/substitutions
    - (syl, None) for deletions
    - (None, syl) for insertions
    """
    # Join syllables into space-separated strings
    ref_str = ' '.join(ref_tokens)
    hyp_str = ' '.join(hyp_tokens)

    # Get alignment using process_words
    output = jiwer.process_words(ref_str, hyp_str)

    # output.alignments is: List[List[AlignmentChunk]]
    # For single sentence pair, use [0]
    alignment_chunks = output.alignments[0]

    # Convert AlignmentChunk objects to (ref, hyp) tuples
    token_pairs = []
    # hanzi_pairs = []

    for chunk in alignment_chunks:
        # chunk has: type, ref_start_idx, ref_end_idx, hyp_start_idx, hyp_end_idx
        # type is: 'equal', 'substitute', 'delete', or 'insert'

        r_span = ref_tokens[chunk.ref_start_idx:chunk.ref_end_idx]
        h_span = hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx]

        r_span_hanzi = ref_hanzi[chunk.ref_start_idx:chunk.ref_end_idx]
        h_span_hanzi = hyp_hanzi[chunk.hyp_start_idx:chunk.hyp_end_idx]

        if chunk.type == 'equal' or chunk.type == 'substitute':
            # Handle both same-length and different-length spans
            if len(r_span) == len(h_span):
                token_pairs.extend(zip(r_span, h_span, r_span_hanzi, h_span_hanzi))
                # token_pairs.extend(zip(r_span, h_span))
                # hanzi_pairs.extend(zip(r_span_hanzi, h_span_hanzi))
            else:
                # Different lengths: pad with None
                for i in range(max(len(r_span), len(h_span))):
                    r = r_span[i] if i < len(r_span) else None
                    h = h_span[i] if i < len(h_span) else None
                    r_hanzi = r_span_hanzi[i] if i < len(r_span_hanzi) else None
                    h_hanzi = h_span_hanzi[i] if i < len(h_span_hanzi) else None
                    token_pairs.append((r, h, r_hanzi, h_hanzi))
                    # token_pairs.append((r, h))
                    # hanzi_pairs.append((r_hanzi, h_hanzi))

        elif chunk.type == "delete":
            # Only ref has element
            token_pairs.extend(zip(r_span, [None] * len(r_span), r_span_hanzi, [None] * len(r_span_hanzi)))
            # token_pairs.extend(zip(r_span, [None] * len(r_span)))
            # hanzi_pairs.extend(zip(r_span_hanzi, [None] * len(r_span_hanzi)))
            # for r in r_span:
            #     token_pairs.append((r, None))

        elif chunk.type == "insert":
            # Only hyp has element
            token_pairs.extend(zip([None] * len(h_span), h_span, [None] * len(h_span_hanzi), h_span_hanzi))
            # token_pairs.extend(zip([None] * len(h_span), h_span))
            # hanzi_pairs.extend(zip([None] * len(h_span_hanzi), h_span_hanzi))
            # for h in h_span:
            #     token_pairs.append((None, h))

    # return token_pairs, hanzi_pairs
    return token_pairs


if __name__ == '__main__':
    from romantization import Pinyinizer

    results = {}

    results['references'] = ["this is the reference", "there is another one", "能吞虾玻璃而不霜身体啦", "我发现了问题", "你好世界"]
    results['predictions'] = ["this is the prediction", "there is an other sample", "我能吞下玻璃而不伤身体", "我发线了问题", "你号世界"]
    print(results)

    for ref, pred in zip(results['references'], results['predictions']):
        token_pairs = align_sequences(ref, pred)
        print(token_pairs)


    # focused on chinese
    pinyinizer_zh = Pinyinizer(language='zh')
    pinyinizer_yue = Pinyinizer(language='yue')
    pinyinizer_nan = Pinyinizer(language='nan')

    results['references'] = ["我沒咧驚", "我发现了问题", "你好世界", "並加速建構具備韌性的非紅供應鏈", "你再发一遍吧"]
    results['predictions'] = ["我沒驚", "我发线了问题", "你号世界", "加速建構具備韌性的供應", "麻烦您再发一遍吧"]
    print(results)

    for ref, pred in zip(results['references'], results['predictions']):
        print(f'Original: {len(ref)} {len(pred)}', ref, pred)

        pinyin_ref_zh = pinyinizer_zh(ref)
        print(f'Mandarin: {len(pinyin_ref_zh)}', pinyin_ref_zh)

        pinyin_pred_zh = pinyinizer_zh(pred)
        print(f'Mandarin: {len(pinyin_pred_zh)}', pinyin_pred_zh)

        token_pairs = align_sequences_pinyin_hanzi(pinyin_ref_zh, pinyin_pred_zh, ref, pred)
        print(len(token_pairs))
        print(token_pairs)


        pinyin_ref_yue = pinyinizer_yue(ref)
        print(f'Cantonese: {len(pinyin_ref_yue)}', pinyin_ref_yue)

        pinyin_pred_yue = pinyinizer_yue(pred)
        print(f'Cantonese: {len(pinyin_pred_yue)}', pinyin_pred_yue)

        token_pairs = align_sequences_pinyin_hanzi(pinyin_ref_yue, pinyin_pred_yue, ref, pred)
        print(len(token_pairs))
        print(token_pairs)


        pinyin_ref_nan = pinyinizer_nan(ref)
        print(f'Taiwanese: {len(pinyin_ref_nan)}', pinyin_ref_nan)

        pinyin_pred_nan = pinyinizer_nan(pred)
        print(f'Taiwanese: {len(pinyin_pred_nan)}', pinyin_pred_nan)

        token_pairs = align_sequences_pinyin_hanzi(pinyin_ref_nan, pinyin_pred_nan, ref, pred)
        print(len(token_pairs))
        print(token_pairs)
