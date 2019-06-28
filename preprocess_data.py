import os
from copy import deepcopy
from types import GeneratorType

import numpy as np
from tqdm import tqdm
from glob import glob
from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams

import utils


def docs_generator(files_dir):
    files_path = glob(os.path.join(files_dir, "*.gz"))

    for gzip_file in files_path:
        file_name = os.path.basename(gzip_file).split(".")[0] # filename without extension
        ofp = os.path.join(files_dir, file_name)
        utils.extract_gzip(gzip_file, ofp)
        with open(ofp, encoding='utf-8') as fobj:
            yield fobj, file_name

        utils.delete_file(ofp)


def text_generator(docs):

    if type(docs) == GeneratorType:
        for doc, fname in docs:
            for line in tqdm(doc, 'Reading: {}'.format(fname)):
                yield " ".join(line.split()[:5])
    else:
        for line in docs:
            yield " ".join(line.split()[:5])


def generate_train_data(data_dir, output_dir, window=5):
    docs_gen = docs_generator(data_dir)
    docs_gen_ = docs_generator(data_dir)
    tokenizer = text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(text_generator(docs_gen))

    word2id = tokenizer.word_index
    id2word = {v: k for k, v in word2id.items()}

    vocab_size = len(word2id) + 1
    print('Vocabulary Size:', vocab_size)
    print('Vocabulary Sample:', list(word2id.items())[:10])

    wids = [[word2id[w] for w in text.text_to_word_sequence(line)] for line in text_generator(docs_gen_)]

    train_data = []

    for wid in tqdm(wids, 'Generating skip gram samples:'):
        pairs, labels = skipgrams(wid, vocabulary_size=vocab_size, window_size=window)

        for pair, label in zip(pairs, labels):
            train_data.append([pair[0], pair[1], label])

    train_data = np.array(train_data)
    print(train_data[:5])

    utils.save_numpy(train_data, os.path.join(output_dir, 'train_data.npy'))
    utils.save_pickle(wids, os.path.join(output_dir, 'word_ids_sent.pkl'))
    utils.save_pickle(word2id, os.path.join(output_dir, 'word2id.pkl'))
    utils.save_pickle(id2word, os.path.join(output_dir, 'id2word.pkl'))
    utils.save_pickle(tokenizer, os.path.join(output_dir, 'tokenizer.pkl'))


if __name__ == '__main__':
    # docs_gen = docs_generator('D:\data\dynamic_word_embedding')
    generate_train_data('./data', './data', window=5)
