from ml import training
from lit_parser import text_fetcher
from lit_parser import similarity_measurer
from n_grams import n_grams_main
from statistical_methods import jaccard, tanimoto
from readers import txt_linesreader
from lit_parser import text_normaliser
import os

book1 = txt_linesreader('/Users/ivanguseff/PycharmProjects/LitSim/literature_test/Булгаков Михаил. Собачье сердце.txt')
book2 = txt_linesreader('literature_test/Шукшин Василий. Калина красная.txt')

# book1 = text_normaliser(book1)
# book2 = text_normaliser(book2)

ngram_counter_1, vocab_1, n_gram_1 = n_grams_main(book1, 3)
ngram_counter_2, vocab_2, n_gram_2 = n_grams_main(book2, 3)

# jaccard ngram similarity measures unique ngrams
jaccard_ngram_similarity = jaccard(n_gram_1, n_gram_2)
# jaccard vocabulary similarity measures unique words
jaccard_vocab_similarity = jaccard(vocab_1, vocab_2)
# tanimoto ngram_counter similarity measures weighted vectors i.e. ngram counters
tanimoto_ngram_counter_similarity = tanimoto(ngram_counter_1, ngram_counter_2)

test_data = [[jaccard_ngram_similarity, jaccard_vocab_similarity, tanimoto_ngram_counter_similarity]]

path = '/Users/ivanguseff/PycharmProjects/LitSim/results/N=3/'
auth1_auth1 = []
auth1_auth2 = []

for root, _, files in os.walk(path):
    for file in files:
        if file.startswith('.'):
            continue
        stats = txt_linesreader(root + '/' + file)[0]
        stats = eval(stats)
        if 'auth1–auth1' in root:
            book_values = list(stats.values())
            book_values = [list(i.values()) for i in book_values]
            auth1_auth1.extend(book_values)
        elif 'auth1–auth2' in root:
            book_values = list(stats.values())
            book_values = [list(i.values()) for i in book_values]
            auth1_auth2.extend(book_values)

result = training(auth1_auth1, auth1_auth2, test_data)
if result == [0]:
    print('0 // same author')
else:
    print('1 // different author')
