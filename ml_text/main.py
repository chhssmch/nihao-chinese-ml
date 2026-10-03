from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings

warnings.filterwarnings("ignore")

_MODEL_EN_ZH = "Helsinki-NLP/opus-mt-en-zh"
_MODEL_ZH_EN = "Helsinki-NLP/opus-mt-zh-en"

_model_en_zh = None
_tokenizer_en_zh = None
_model_zh_en = None
_tokenizer_zh_en = None


def detect_english_ratio(text: str) -> float:
    if not text:
        return 0.0

    english_chars = 0
    total_chars = 0

    for char in text:
        if char.isalpha():
            total_chars += 1
            if char.isascii() and char.isalpha():
                english_chars += 1

    if total_chars == 0:
        return 0.0

    return english_chars / total_chars


def validate_language(text: str, direction: str) -> None:
    english_ratio = detect_english_ratio(text)

    if direction == "en_to_zh":
        if english_ratio < 0.6:
            raise ValueError(
                "Текст не соответствует выбранному направлению перевода. "
                "Для перевода введите текст на английском языке."
            )
    elif direction == "zh_to_en":
        if english_ratio > 0.5:
            raise ValueError(
                "Текст не соответствует выбранному направлению перевода. "
                "Для перевода введите текст на китайском языке."
            )


def get_model_and_tokenizer_en_zh():
    global _model_en_zh, _tokenizer_en_zh
    if _model_en_zh is None:
        _model_en_zh = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_EN_ZH)
        _tokenizer_en_zh = AutoTokenizer.from_pretrained(_MODEL_EN_ZH)
    return _model_en_zh, _tokenizer_en_zh


def get_model_and_tokenizer_zh_en():
    global _model_zh_en, _tokenizer_zh_en
    if _model_zh_en is None:
        _model_zh_en = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_ZH_EN)
        _tokenizer_zh_en = AutoTokenizer.from_pretrained(_MODEL_ZH_EN)
    return _model_zh_en, _tokenizer_zh_en


def translate_en_to_zh(text: str) -> str:
    model, tokenizer = get_model_and_tokenizer_en_zh()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def translate_zh_to_en(text: str) -> str:
    model, tokenizer = get_model_and_tokenizer_zh_en()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

def translate(text: str, direction: str = "en_to_zh") -> str:
    validate_language(text, direction)

    if direction == "en_to_zh":
        return translate_en_to_zh(text)
    elif direction == "zh_to_en":
        return translate_zh_to_en(text)
    else:
        raise ValueError(f"Неправильное направление: {direction}")