import os
from ml import predict, best_params_xgboost
from corpus_processing import text_lemmatisation
from writers_and_readers import txt_linesreader
from n_grams import n_grams_main
from statistical_methods import jaccard, tanimoto


def extract_features(text1, text2, n=3, lemmatise=False):
    if lemmatise:
        text1 = text_lemmatisation(text1)
        text2 = text_lemmatisation(text2)

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


def compare_authors(file1_path, file2_path, stats_path, n=3, lemmatise=False):
    book1 = txt_linesreader(file1_path)
    book2 = txt_linesreader(file2_path)

    test_features = [extract_features(book1, book2, n=n, lemmatise=lemmatise)]

    if lemmatise is True:
        stats_path += 'values_lemmatised/'
    elif lemmatise is False:
        stats_path += 'values/'
    stats_path += f'N={n}/'

    auth1_data, auth2_data = load_training_data(stats_path)
    print(best_params_xgboost(auth1_data, auth2_data, test_features))
    # result = predict(auth1_data, auth2_data, test_features)
    # print("0 // same author" if result == [0] else "1 // different author")
    # return result


if __name__ == "__main__":
    test_folder = os.path.join('LitSim', 'literature_test/')
    train_folder = os.path.join('LitSim/')

    file1 = os.path.join(test_folder, '.txt')
    file2 = os.path.join(test_folder, '.txt')

    compare_authors(file1, file2, train_folder, n=3, lemmatise=True)
