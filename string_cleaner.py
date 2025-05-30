import string
import re


def complex_cleaner(sentence):
    sentence = punctuation_cleaning(sentence)
    sentence = arabic_numerals_cleaner(sentence)
    sentence = roman_numerals_cleaner(sentence)

    return sentence


def punctuation_cleaning(sentence):
    additional_punctuation = '«»–—[]'
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    sentence = sentence.translate(str.maketrans("", "", additional_punctuation))

    return sentence


def arabic_numerals_cleaner(sentence):
    sentence = sentence.translate(str.maketrans("", "", string.digits))

    return sentence


def roman_numerals_cleaner(text):
    # only matches standalone patterns
    roman_pattern = r'\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b'
    cleaned_text = re.sub(roman_pattern, '', text, flags=re.IGNORECASE)

    return cleaned_text
