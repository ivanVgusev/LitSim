from writers_and_readers import txt_writer, fb2reader, txt_reader, epub_reader
from string_cleaner import complex_cleaner
from statistical_methods import jaccard, tanimoto
from n_grams import n_grams_main

import nltk
import pymorphy2

import os
import itertools
import math

from progress_monitor import progress_bar


morph = pymorphy2.MorphAnalyzer()


def text_fetcher(repo_path: str) -> dict:
    """
    This function reads and formats texts, normalises them and puts into a dictionary,
    so that it would be easily accessible.

    :param repo_path: path to the folder with books
    :return: dictionary of the form {name of the book: text}
    """

    litcorpus = {}
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.startswith('.'):
                continue

            full_path = os.path.join(root, file)
            if file.endswith('.fb2'):
                try:
                    text = fb2reader(full_path)
                    text = text_normalisation(text)
                    litcorpus.update({file: text})

                except Exception as e:
                    print(f'An error {e} occurred!')
                    continue

            elif file.endswith('.txt'):
                try:
                    text = txt_reader(full_path)
                    text = text_normalisation(text)
                    litcorpus.update({file: text})

                except Exception as e:
                    print(f'An error {e} occurred!')
                    continue

            elif file.endswith('.epub'):
                try:
                    text = epub_reader(full_path)
                    text = text_normalisation(text)
                    litcorpus.update({file: text})

                except Exception as e:
                    print(f'An error {e} occurred!')
                    continue

    return litcorpus


def text_lemmatisation(text: str) -> str:
    """
    Text normalisation and lemmatisation.
    ::param text: literature piece
    """

    if not isinstance(text, str):
        return ''

    lemmatised_text = ''
    sentences = nltk.tokenize.sent_tokenize(text, "russian")
    for sentence in sentences:
        sentence = nltk.tokenize.word_tokenize(sentence, "russian")
        if len(sentence) == 0:
            continue
        lemmatised_sentence = [morph.parse(i)[0].normal_form for i in sentence]
        lemmatised_text += ' '.join(lemmatised_sentence)

    return lemmatised_text


def text_normalisation(text: str) -> str:
    """
    This function cleans input texts from punctuation marks and other unnecessary elements (numbers, etc.),
    formats the text to lowercase.

    :param text: book text
    :return: normalised book text
    """

    text = complex_cleaner(text)
    text = text.lower()
    return text


def corpus_lemmatisation(base_path: str, lit_folder_name: str) -> None:
    """
    This function applies text_lemmatisation() to the user's literature corpus while also
    monitoring that the elements that had already been lemmatised would not be processed once again.

    :param base_path: path to the directory (without the folder that stores literature)
    :param lit_folder_name: name of the folder that stores literature
    :return: None
    """

    # Construct the full path to the input and output directories
    input_dir = os.path.join(base_path, lit_folder_name)
    output_dir = os.path.join(base_path, f"{lit_folder_name}_lemmatised")
    os.makedirs(output_dir, exist_ok=True)

    files_to_lemmatise = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.startswith('.'):
                continue  # Skip hidden/system files

            # Full path to the current input file
            full_input_path = os.path.join(root, file)

            # Get path relative to the literature folder (preserving subfolders)
            rel_path = os.path.relpath(full_input_path, input_dir)

            # Build corresponding output path in the lemmatised folder
            # Ensure the output has a .txt extension
            lemmatised_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.txt')

            # Only add to the processing list if not already lemmatised
            if not os.path.exists(lemmatised_path):
                files_to_lemmatise.append((full_input_path, lemmatised_path))

    if not files_to_lemmatise:
        print("[corpus_lemmatisation] /// All the texts have already been lemmatised, so we're good to go!")
        return

    print(f"[corpus_lemmatisation] /// Looks like we need to lemmatise {len(files_to_lemmatise)} text(s)")

    # Process each file that needs to be lemmatised
    for idx, (src, dest) in enumerate(files_to_lemmatise, start=1):
        progress_bar(len(files_to_lemmatise), idx)

        try:
            # Choose the appropriate reader based on file type
            if src.endswith('.fb2'):
                text = fb2reader(src)
            elif src.endswith('.epub'):
                text = epub_reader(src)
            elif src.endswith('.txt'):
                text = txt_reader(src)
            else:
                continue  # Skip unsupported file types
        except Exception:
            continue

        # Lemmatise the text content
        lemmatised_text = text_lemmatisation(text)

        # Make sure the output directory exists before writing
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        # Write the lemmatised content to the output file
        txt_writer(lemmatised_text, dest)


def similarity_measurer(litcorpus1: dict, litcorpus2: dict, N) -> dict:
    """
    Jaccard ngrams, Jaccard vocabulary and Tanimoto ngrams counters measuring in order to compute text similarity.

    :param litcorpus1: the output dict of text_fetcher() i.e. a processed literature piece
    :param litcorpus2: the output dict of text_fetcher() i.e. a processed literature piece
    :param N: parameter for N-grams; in this project (N = 2; N = 3; N = 4) are conventionally used
    :return: dict with measured similarity parameters for two literature pieces
    """

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


def auth1_auth1(author_folder_filepath: str, N: int) -> dict:
    """
    This function measures similarity of literature pieces written by the same author.

    :param author_folder_filepath: path to the folder with author's books
    :param N: parameter for N-grams; in this project (N = 2; N = 3; N = 4) are conventionally used
    :return: dictionary of dictionaries;
    {name of the book: {ngrams jaccard value: float, vocabulary jaccard value: float, ngrams tanimoto value: float}}
    """
    corpus1 = text_fetcher(author_folder_filepath)
    corpus2 = corpus1
    stats = similarity_measurer(corpus1, corpus2, N)
    return stats


def auth1_auth2(author_folder_filepath1: str, author_folder_filepath2: str, N: int) -> dict:
    """
    This function measures similarity of literature pieces written by two different author.

    :param author_folder_filepath1: path to the folder with the first author's books
    :param author_folder_filepath2: path to the folder with the second author's books
    :param N: parameter for N-grams; in this project (N = 2; N = 3; N = 4) are conventionally used
    :return: dictionary of dictionaries;
    {name of the book: {ngrams jaccard value: float, vocabulary jaccard value: float, ngrams tanimoto value: float}}
    """
    corpus1 = text_fetcher(author_folder_filepath1)
    corpus2 = text_fetcher(author_folder_filepath2)
    stats = similarity_measurer(corpus1, corpus2, N)
    return stats


def wholesale_processing_auth1_auth1(base_path: str, lit_folder_name: str, N: int, lemmatised: bool = True) -> None:
    """
    This function processes all the author's folders in the needed directory and measures author1-author1 similarity.

    :param base_path: path to the directory (without the folder that stores literature)
    :param lit_folder_name: name of the folder that stores literature
    :param N: parameter for N-grams; in this project (N = 2; N = 3; N = 4) are conventionally used
    :param lemmatised: True/False depending on whether lemmatisation is needed or not
    :return: None
    """

    inp_path = str(os.path.join(base_path, lit_folder_name))
    if lemmatised:
        corpus_lemmatisation(base_path, lit_folder_name)
        inp_path = str(os.path.join(base_path, f'{lit_folder_name}_lemmatised'))

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

        if lemmatised:
            output_path = os.path.join(base_path, 'values_lemmatised', f'N={N}', 'auth1–auth1')
        else:
            output_path = os.path.join(base_path, 'values', f'N={N}', 'auth1–auth1')

        os.makedirs(output_path, exist_ok=True)

        output_path = os.path.join(output_path, f'{basename}–{basename}.txt')

        if not os.path.exists(output_path):
            stats = auth1_auth1(folder_full_path, N)
            txt_writer(stats, output_path)


def wholesale_processing_auth1_auth2(base_path: str, lit_folder_name: str, N: int, lemmatised: bool = True) -> None:
    """
     This function processes all the author's folders in the needed directory and measures author1-author2 similarity.

     :param base_path: path to the directory (without the folder that stores literature)
     :param lit_folder_name: name of the folder that stores literature
     :param N: parameter for N-grams; in this project (N = 2; N = 3; N = 4) are conventionally used
     :param lemmatised: True/False depending on whether lemmatisation is needed or not
     :return: None
     """

    inp_path = str(os.path.join(base_path, lit_folder_name))

    if lemmatised:
        corpus_lemmatisation(base_path, lit_folder_name)
        inp_path = str(os.path.join(base_path, f'{lit_folder_name}_lemmatised'))

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

            if lemmatised:
                output_path = os.path.join(base_path, 'values_lemmatised', f'N={N}', 'auth1–auth2')
            else:
                output_path = os.path.join(base_path, 'values', f'N={N}', 'auth1–auth2')

            os.makedirs(output_path, exist_ok=True)

            output_path = os.path.join(output_path, f'{basename1}–{basename2}.txt')

            if not os.path.exists(output_path):
                stats = auth1_auth2(folder1_full_path, folder2_full_path, N)
                txt_writer(stats, output_path)
                analysed_author_pairs.append(sorted([folder1, folder2]))


def main_processing(base_path: str, lit_folder_name: str) -> None:
    """
    This function is the main one that calls wholesale_processing_auth1_auth1() and wholesale_processing_auth1_auth2().
    It configures all possible combinations of N (2, 3, 4) and lemmatisation (True, False)
    and measures values for such configs.

    :param base_path: path to the directory (without the folder that stores literature)
    :param lit_folder_name: name of the folder that stores literature
    :return: None
    """
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    n_grams_configuration = [2, 3, 4]
    lemmatisation_configuration = [True, False]
    config_combinations = itertools.product(n_grams_configuration, lemmatisation_configuration)

    for congfig in config_combinations:
        n_grams_config = congfig[0]
        lemmatisation_config = congfig[1]
        print(f'[corpus_processing] /// CONFIGURATION /// '
              f'N={n_grams_config}, lemmatisation={lemmatisation_config}')
        wholesale_processing_auth1_auth1(base_path, lit_folder_name, n_grams_config, lemmatisation_config)
        wholesale_processing_auth1_auth2(base_path, lit_folder_name, n_grams_config, lemmatisation_config)


# path = os.path.join("/Users", "ivanguseff", "PycharmProjects", "LitSim")
# literature_folder = 'literature'
# main_processing(path, literature_folder)
