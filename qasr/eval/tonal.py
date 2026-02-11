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

