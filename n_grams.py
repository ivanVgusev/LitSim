import nltk
import string
from collections import Counter
import math
from string_cleaner import punctuation_cleaning


def reader(file, encoding='utf-8'):
    with open(file, 'r', encoding=encoding) as f:
        lines = f.readlines()
    return lines


# splitting the input data into two parts (train data and test data)
def prepare_train_test_data(input_texts):
    train_data = input_texts[: len(input_texts) // 10 * 9]
    test_data = input_texts[len(input_texts) // 10 * 9:]
    return train_data, test_data


# transforming the input texts into sentences of "<s> ... </s>" form
def make_sentence_list(texts, N):
    N_1 = N - 1
    result_sentences = []
    for text in texts:
        sentences = nltk.tokenize.sent_tokenize(text)
        for sentence in sentences:
            sentence = str('<s> ' * N_1 + punctuation_cleaning(sentence) + ' </s>')
            sentence = sentence.lower().split()

            if len(sentence) == 0:
                continue

            result_sentences.append(sentence)
    return result_sentences


# transforming the input sentence into n_grams
def get_n_grams_for_sentence(sentence, N):
    ngrams = []
    for i in range(len(sentence) - N + 1):
        ngram = tuple(sentence[i: i + N])
        ngrams.append(ngram)
    return ngrams


# counting n_grams and n_1grams in the sentences. also returning the text's vocabulary
def get_ngram_dict(sentences, N):
    vocab = set()
    ngram_counter = Counter()
    n_1gram_counter = Counter()

    for sentence in sentences:
        vocab.update(sentence)
        ngrams = get_n_grams_for_sentence(sentence, N)
        n_1grams = get_n_grams_for_sentence(sentence, N - 1)
        ngram_counter.update(ngrams)
        n_1gram_counter.update(n_1grams)

    vocab.add('<unk>')
    return ngram_counter, n_1gram_counter, vocab


# counting n_gram probability for each sentence and applying Laplace smoothing
def get_sentence_probability(sentence, ngram_counter, n_1gram_counter, vocab):
    sentence_probability = 0
    for n_gram in sentence:
        n_1gram_prob = n_1gram_counter.get(n_gram[:-1], 0)
        n_gram_prob = ngram_counter.get(n_gram, 0)
        # probability -> log exp
        probability = math.log((n_gram_prob + 1) / (n_1gram_prob + len(vocab)))
        sentence_probability += probability
    return sentence_probability


def n_grams_main(data, N):
    sentences = make_sentence_list(data, N)
    ngram_counter, n_1gram_counter, vocab = get_ngram_dict(sentences, N)
    n_gram = set([item for i in sentences for item in get_n_grams_for_sentence(i, N)])

    return [ngram_counter, vocab, n_gram]
