import re


def pre_caption(caption, max_words = 64):
    """
    对字幕进行预处理，包括规范化和长度截断。

    规范化处理会将字幕转换为小写，去除特殊字符并替换某些字符为指定的格式。
    长度截断确保字幕不会超过指定的最大单词数。

    参数:
    - caption: 输入的字幕字符串。
    - max_words: 允许的最大单词数，默认为64。

    返回:
    - 经过处理的字幕字符串。

    异常:
    - 当处理后的字幕为空时，抛出ValueError。
    """
    # 原始字幕，用于异常时的提示
    caption_raw = caption

    # 规范化字幕文本：转换为小写，替换特殊字符为单个空格
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        ' ',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    # 合并多余空格为单个空格
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )

    # 去除尾部换行符和首尾空格
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # 截断字幕，确保不超过max_words个单词
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    # 确保字幕处理后不为空，否则抛出异常
    if not len(caption):
        raise ValueError(f"pre_caption yields invalid text (raw: {caption_raw})")

    return caption


def pre_caption_list(captionList, max_words = 64):
    """
    对列表中的每个字幕进行预处理，确保每个字幕不超过指定的最大字数。

    参数:
    - captionList: List[str], 需要进行预处理的字幕列表。
    - max_words: int, 每个字幕最多允许的字数，默认为64。

    返回:
    - newCaptionList: List[str], 预处理后的字幕列表。
    """
    # 使用列表推导式对字幕列表中的每个字幕进行预处理
    newCaptionList = [pre_caption(caption, max_words) for caption in captionList]

    return newCaptionList
