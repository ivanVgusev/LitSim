import string
import re
import unicodedata


def complex_cleaner(sentence) -> str:
    sentence = punctuation_cleaning(sentence)
    sentence = arabic_numerals_cleaner(sentence)
    sentence = roman_numerals_cleaner(sentence)

    return str(sentence)


def punctuation_cleaning(sentence: str) -> str:
    cleaned = []

    for char in sentence:
        if char.isspace():
            cleaned.append(' ')
        elif unicodedata.category(char).startswith(('P', 'S')):
            continue
        elif char in {'\u200b', '\ufeff'}:
            cleaned.append(' ')
        else:
            cleaned.append(char)

    return ' '.join(''.join(cleaned).split())


def arabic_numerals_cleaner(sentence):
    sentence = sentence.translate(str.maketrans("", "", string.digits))

    return sentence


def roman_numerals_cleaner(text):
    # only matches standalone patterns
    roman_pattern = r'\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b'
    cleaned_text = re.sub(roman_pattern, '', text, flags=re.IGNORECASE)

    return cleaned_text
