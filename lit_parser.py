import os
from n_grams import n_grams_main
from readers import fb2reader, txt_reader
import time
from statistical_methods import jaccard, tanimoto
from progress_monitor import progress_bar


def txt_writer(data, filepath, encoding='utf-8'):
    if type(data) is str:
        pass
    else:
        data = str(data)
    with open(filepath, 'w', encoding=encoding) as f:
        f.write(data)


def text_fetcher(repo_path: str) -> dict:
    litcorpus = {}
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.startswith('.'):
                continue

            if file.endswith('.fb2'):
                try:
                    text = fb2reader(root + '/' + file)
                    litcorpus.update({file: text})
                except Exception as e:
                    # print(f'An error {e} occurred!')
                    continue

            if file.endswith('.txt'):
                try:
                    text = txt_reader(root + '/' + file)
                    litcorpus.update({file: text})
                except Exception as e:
                    # print(f'An error {e} occurred!')
                    continue

    return litcorpus


# Jaccard ngrams, Jaccard vocabulary and Tanimoto ngrams counters
def similarity_measurer(litcorpus1: dict, litcorpus2: dict, N) -> dict:
    total_iterations = len(litcorpus1) * len(litcorpus2)
    iteration_counter = 0

    stats_dict = {}
    for text1 in litcorpus1:
        loop_start = time.time()
        for text2 in litcorpus2:
            # progress bar updating
            iteration_counter += 1
            progress_bar(total_iterations, iteration_counter)

            ngram_counter_1, vocab_1, n_gram_1 = n_grams_main(litcorpus1.get(text1), N)
            ngram_counter_2, vocab_2, n_gram_2 = n_grams_main(litcorpus2.get(text2), N)

            # jaccard ngram similarity measures unique ngrams
            jaccard_ngram_similarity = jaccard(n_gram_1, n_gram_2)
            # jaccard vocabulary similarity measures unique words
            jaccard_vocab_similarity = jaccard(vocab_1, vocab_2)
            # tanimoto ngram_counter similarity measures weighted vectors i.e. ngram counters
            tanimoto_ngram_counter_similarity = tanimoto(ngram_counter_1, ngram_counter_2)

            # excluding text == text situations
            if jaccard_ngram_similarity and jaccard_vocab_similarity and tanimoto_ngram_counter_similarity == 1.0:
                continue

            texts_similarity = {f'{text1} – {text2}': {'jaccard_ngram': jaccard_ngram_similarity,
                                                       'jaccard_vocab': jaccard_vocab_similarity,
                                                       'tanimoto_ngram_counter': tanimoto_ngram_counter_similarity}}
            stats_dict.update(texts_similarity)
        loop_end = time.time()
        print(f' /// This iteration took {loop_end - loop_start} seconds.')
    return stats_dict


path1 = '/Users/ivanguseff/PycharmProjects/LitSim/literature/Толстой'
# path2 = '/Users/ivanguseff/PycharmProjects/LitSim/literature/Тургенев'
path2 = path1

name = 'tolstoy–tolstoy.txt'
output_path = '/Users/ivanguseff/PycharmProjects/LitSim/results/N=2/auth1–auth2'

start_time = time.time()

corpus1 = text_fetcher(path1)
corpus2 = text_fetcher(path2)
stats = similarity_measurer(corpus1, corpus2, 2)

txt_writer(stats, output_path + '/' + name)

end_time = time.time()
print(f'Achieved in: {end_time - start_time}')
