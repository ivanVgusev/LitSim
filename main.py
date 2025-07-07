import os
from ml import predict
from corpus_processing import text_normaliser
from readers import txt_linesreader
from n_grams import n_grams_main
from statistical_methods import jaccard, tanimoto


def extract_features(text1, text2, n=3, normalize=False):
    if normalize:
        text1 = text_normaliser(text1)
        text2 = text_normaliser(text2)

    ngram_counter_1, vocab_1, ngram_1 = n_grams_main(text1, n)
    ngram_counter_2, vocab_2, ngram_2 = n_grams_main(text2, n)

    return [
        jaccard(ngram_1, ngram_2),
        jaccard(vocab_1, vocab_2),
        tanimoto(ngram_counter_1, ngram_counter_2)
    ]


def load_training_data(stats_path):
    auth1_auth1, auth1_auth2 = [], []

    for root, _, files in os.walk(stats_path):
        for file in files:
            if file.startswith('.'):
                continue
            full_path = os.path.join(root, file)
            stats = eval(txt_linesreader(str(full_path))[0])
            book_values = [list(item.values()) for item in stats.values()]

            if 'auth1–auth1' in root:
                auth1_auth1.extend(book_values)
            elif 'auth1–auth2' in root:
                auth1_auth2.extend(book_values)

    return auth1_auth1, auth1_auth2


def compare_authors(file1_path, file2_path, stats_path, n=3, normalise=False):
    book1 = txt_linesreader(file1_path)
    book2 = txt_linesreader(file2_path)

    test_features = [extract_features(book1, book2, n=n, normalize=normalise)]

    if normalise is True:
        stats_path += 'results_normalised/'
    elif normalise is False:
        stats_path += 'results/'
    stats_path += f'N={n}/'

    auth1_data, auth2_data = load_training_data(stats_path)
    result = predict(auth1_data, auth2_data, test_features)
    print("0 // same author" if result == [0] else "1 // different author")
    return result


if __name__ == "__main__":
    test_folder = '/Users/ivanguseff/PycharmProjects/LitSim/literature_test/'
    train_folder = '/Users/ivanguseff/PycharmProjects/LitSim/'

    file1 = os.path.join(test_folder, 'Булгаков Михаил. Собачье сердце.txt')
    file2 = os.path.join(test_folder, 'Шукшин Василий. Калина красная.txt')

    compare_authors(file1, file2, train_folder, n=3, normalise=False)
