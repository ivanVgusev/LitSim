import os
from n_grams import n_grams_main
from readers import fb2reader, txt_linesreader, epub_reader
from statistical_methods import jaccard, tanimoto
from progress_monitor import progress_bar
import math
import pymorphy2
import nltk
from string_cleaner import complex_cleaner
from pathlib import Path
import itertools


morph = pymorphy2.MorphAnalyzer()


def txt_writer(data, filepath, encoding='utf-8'):
    if type(data) is not str:
        data = str(data)

    if not filepath.endswith('.txt'):
        filepath += '.txt'

    with open(filepath, 'w', encoding=encoding) as f:
        f.write(data)


def text_normaliser(text):
    normalised_text = ''
    for lines in text:
        sentences = nltk.tokenize.sent_tokenize(lines, "russian")
        for sentence in sentences:
            sentence = str(complex_cleaner(sentence))
            sentence = sentence.lower()
            sentence = nltk.tokenize.word_tokenize(sentence, "russian")

            if len(sentence) == 0:
                continue
            normalised_sentence = [morph.parse(i)[0].normal_form for i in sentence]
            normalised_text += ' '.join(normalised_sentence) + ' '

    return normalised_text


def corpus_normaliser(m_path, lit_folder_name):
    inp_path = str(os.path.join(m_path, lit_folder_name))
    normalised_literature_folder = str(os.path.join(m_path, f'{lit_folder_name}_normalised'))
    os.makedirs(os.path.join(normalised_literature_folder), exist_ok=True)

    already_normalised = []
    yet_to_normalise = 0
    for root, _, files in os.walk(inp_path):
        for file in files:
            if file.startswith('.'):
                continue
            path = Path(str(os.path.join(root, file)))
            path_parts = list(path.parts)
            lit_folder_index = path_parts.index(lit_folder_name)

            pre_index = path_parts[:lit_folder_index]
            post_index = path_parts[lit_folder_index + 1:]
            normalised_path = os.path.join(*pre_index, f'{lit_folder_name}_normalised', *post_index)
            if not normalised_path.endswith('.txt'):
                normalised_path = os.path.splitext(normalised_path)[0] + '.txt'

            if not os.path.exists(normalised_path):
                yet_to_normalise += 1
            else:
                # basename = os.path.basename(normalised_path)[1]
                # already_normalised.append(basename)[[
                already_normalised.append(file)

    if yet_to_normalise == 0:
        return print("[corpus_normaliser] /// All the texts have already been normalised, so we're good to go!")
    else:
        print(f'[corpus_normaliser] /// Looks like we need to normalise {yet_to_normalise} text(s)')
        iteration_counter = 0

        for root, _, files in os.walk(inp_path):
            for file in files:
                if file.startswith('.'):
                    continue
                if file in already_normalised:
                    continue

                # progress bar updating
                iteration_counter += 1
                progress_bar(yet_to_normalise, iteration_counter)

                full_non_normalised_path = os.path.join(root, file)
                path = Path(full_non_normalised_path)
                path_parts = list(path.parts)
                lit_folder_index = path_parts.index(lit_folder_name)

                pre_index = path_parts[:lit_folder_index]
                post_index = path_parts[lit_folder_index + 1:]

                full_normalised_path = os.path.join(*pre_index, f'{lit_folder_name}_normalised', *post_index)
                full_normalised_path = os.path.splitext(full_normalised_path)[0]
                path = Path(full_normalised_path)
                path_parts = list(path.parts)
                full_normalised_path_no_basename = path_parts[:-1]
                full_normalised_path_no_basename = os.path.join(*full_normalised_path_no_basename)

                if file.endswith('.fb2'):
                    try:
                        text = fb2reader(full_non_normalised_path)
                    except Exception as e:
                        # print(f'An error {e} occurred!')
                        continue
                elif file.endswith('.txt'):
                    try:
                        text = txt_linesreader(full_non_normalised_path)
                    except Exception as e:
                        # print(f'An error {e} occurred!')
                        continue
                else:
                    continue

                normalised_text = text_normaliser(text)

                os.makedirs(full_normalised_path_no_basename, exist_ok=True)

                txt_writer(normalised_text, full_normalised_path)


def text_fetcher(repo_path: str) -> dict:
    litcorpus = {}
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.startswith('.'):
                continue

            full_path = os.path.join(root, file)
            if file.endswith('.fb2'):
                try:
                    text = fb2reader(full_path)
                    litcorpus.update({file: text})
                except Exception as e:
                    # print(f'An error {e} occurred!')
                    continue

            elif file.endswith('.txt'):
                try:
                    text = txt_linesreader(full_path)
                    litcorpus.update({file: text})
                except Exception as e:
                    # print(f'An error {e} occurred!')
                    continue

            elif file.endswith('.epub'):
                try:
                    text = epub_reader(full_path)
                    litcorpus.update({file: text})
                except Exception as e:
                    # print(f'An error {e} occurred!')
                    continue

    return litcorpus


# Jaccard ngrams, Jaccard vocabulary and Tanimoto ngrams counters
def similarity_measurer(litcorpus1: dict, litcorpus2: dict, N) -> dict:
    k = 2
    all_combinations = list(itertools.product(litcorpus1, litcorpus2))
    unique_combinations = len([(a, b) for a, b in all_combinations if a != b])

    if litcorpus1 == litcorpus2:
        unique_combinations_condensed = int(unique_combinations / k)
    else:
        unique_combinations_condensed = int(unique_combinations)

    iteration_counter = 0
    stats_dict = {}
    analysed_books_pairs = []
    for text1 in litcorpus1:
        for text2 in litcorpus2:
            if text1 == text2:
                continue
            if sorted([text1, text2]) in analysed_books_pairs:
                continue

            # progress bar updating
            iteration_counter += 1
            progress_bar(unique_combinations_condensed, iteration_counter)

            ngram_counter_1, vocab_1, n_gram_1 = n_grams_main(litcorpus1.get(text1), N)
            ngram_counter_2, vocab_2, n_gram_2 = n_grams_main(litcorpus2.get(text2), N)

            # jaccard ngram similarity measures unique ngrams
            jaccard_ngram_similarity = jaccard(n_gram_1, n_gram_2)
            # jaccard vocabulary similarity measures unique words
            jaccard_vocab_similarity = jaccard(vocab_1, vocab_2)
            # tanimoto ngram_counter similarity measures weighted vectors i.e. ngram counters
            tanimoto_ngram_counter_similarity = tanimoto(ngram_counter_1, ngram_counter_2)

            texts_similarity = {f'{text1} – {text2}': {'jaccard_ngram': jaccard_ngram_similarity,
                                                       'jaccard_vocab': jaccard_vocab_similarity,
                                                       'tanimoto_ngram_counter': tanimoto_ngram_counter_similarity}}
            stats_dict.update(texts_similarity)
            analysed_books_pairs.append(sorted([text1, text2]))
    return stats_dict


def auth1_auth1(author_folder_filepath, N):
    corpus1 = text_fetcher(author_folder_filepath)
    corpus2 = corpus1
    stats = similarity_measurer(corpus1, corpus2, N)
    return stats


def auth1_auth2(author_folder_filepath1, author_folder_filepath2, N):
    corpus1 = text_fetcher(author_folder_filepath1)
    corpus2 = text_fetcher(author_folder_filepath2)
    stats = similarity_measurer(corpus1, corpus2, N)
    return stats


def wholesale_processing_auth1_auth1(m_path, lit_folder_name, N, normalised=True):
    inp_path = str(os.path.join(m_path, lit_folder_name))
    if normalised:
        corpus_normaliser(m_path, lit_folder_name)
        inp_path = str(os.path.join(m_path, f'{lit_folder_name}_normalised'))

    author_directories = os.listdir(inp_path)
    n = len(author_directories)
    if '.DS_Store' in author_directories:
        n -= 1
    print(f'[wholesale_processing_auth1_auth1] /// '
          f'There are {n} author directories => {n} possible non-repeated combination(s)')

    for folder in author_directories:
        if folder.startswith('.'):
            continue

        folder_full_path = os.path.join(inp_path, folder)
        basename = os.path.basename(folder_full_path)

        if normalised:
            output_path = os.path.join(m_path, 'values_normalised', f'N={N}', 'auth1–auth1')
        else:
            output_path = os.path.join(m_path, 'values', f'N={N}', 'auth1–auth1')

        os.makedirs(output_path, exist_ok=True)

        output_path = os.path.join(output_path, f'{basename}–{basename}.txt')

        if not os.path.exists(output_path):
            stats = auth1_auth1(folder_full_path, N)
            txt_writer(stats, output_path)


def wholesale_processing_auth1_auth2(m_path, lit_folder_name, N, normalised=True):
    inp_path = str(os.path.join(m_path, lit_folder_name))

    if normalised:
        corpus_normaliser(m_path, lit_folder_name)
        inp_path = str(os.path.join(m_path, f'{lit_folder_name}_normalised'))

    author_directories = os.listdir(inp_path)
    n = len(author_directories)
    if '.DS_Store' in author_directories:
        n -= 1
    k = 2
    combinations_amount = int(math.factorial(n) / (math.factorial(n - k) * math.factorial(k)))
    print(f'[wholesale_processing_auth1_auth2] /// '
          f'There are {n} author directories => {combinations_amount} possible non-repeated combination(s)')

    analysed_author_pairs = []
    for folder1 in author_directories:
        if folder1.startswith('.'):
            continue
        for folder2 in author_directories:
            if folder2.startswith('.'):
                continue
            if folder1 == folder2:
                continue
            # checking the (auth1–auth2, auth2–auth1) situations
            # i.e. when the combination appears two times but with different element placements
            if sorted([folder1, folder2]) in analysed_author_pairs:
                continue

            folder1_full_path = os.path.join(inp_path, folder1)
            folder2_full_path = os.path.join(inp_path, folder2)
            basename1 = os.path.basename(folder1_full_path)
            basename2 = os.path.basename(folder2_full_path)
            if normalised:
                output_path = os.path.join(m_path, 'values_normalised', f'N={N}', 'auth1–auth2')
            else:
                output_path = os.path.join(m_path, 'values', f'N={N}', 'auth1–auth2')

            os.makedirs(output_path, exist_ok=True)

            output_path = os.path.join(output_path, f'{basename1}–{basename2}.txt')

            if not os.path.exists(output_path):
                stats = auth1_auth2(folder1_full_path, folder2_full_path, N)
                txt_writer(stats, output_path)
                analysed_author_pairs.append(sorted([folder1, folder2]))


def corpus_processing(main_path, literature_folder_name):
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    n_grams_confing = [2, 3, 4]
    normalisation_confing = [True, False]
    config_combinations = itertools.product(n_grams_confing, normalisation_confing)
    for congfig in config_combinations:
        N_config = congfig[0]
        normalisation_config = congfig[1]
        print(f'[corpus_processing] /// N={N_config}, normalisation={normalisation_config}')

        wholesale_processing_auth1_auth1(main_path, literature_folder_name, N_config, normalisation_config)
        wholesale_processing_auth1_auth2(main_path, literature_folder_name, N_config, normalisation_config)


# path = os.path.join("/Users", "ivanguseff", "PycharmProjects", "LitSim")
# literature_folder = 'literature'
# corpus_processing(path, literature_folder)
