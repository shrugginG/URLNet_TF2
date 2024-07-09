from TextCNN_tf2 import TextCNN
from utils_tf2 import *
import pickle
import time
from tqdm import tqdm
import argparse
import numpy as np
import pickle
import tensorflow as tf


parser = argparse.ArgumentParser(description="Test URLNet model")

# SECTION - data args
# data args
default_max_len_words = 200
parser.add_argument(
    "--data.max_len_words",
    type=int,
    default=default_max_len_words,
    metavar="MLW",
    help="maximum length of url in words (default: {})".format(default_max_len_words),
)
default_max_len_chars = 200
parser.add_argument(
    "--data.max_len_chars",
    type=int,
    default=default_max_len_chars,
    metavar="MLC",
    help="maximum length of url in characters (default: {})".format(
        default_max_len_chars
    ),
)
default_max_len_subwords = 20
parser.add_argument(
    "--data.max_len_subwords",
    type=int,
    default=default_max_len_subwords,
    metavar="MLSW",
    help="maxium length of word in subwords/ characters (default: {})".format(
        default_max_len_subwords
    ),
)
parser.add_argument(
    "--data.data_dir",
    type=str,
    default="train_10000.txt",
    metavar="DATADIR",
    help="location of data file",
)
default_delimit_mode = 1
parser.add_argument(
    "--data.delimit_mode",
    type=int,
    default=default_delimit_mode,
    metavar="DLMODE",
    help="0: delimit by special chars, 1: delimit by special chars + each char as a word (default: {})".format(
        default_delimit_mode
    ),
)
parser.add_argument(
    "--data.subword_dict_dir",
    type=str,
    default="runs/10000/subwords_dict.p",
    metavar="SUBWORD_DICT",
    help="directory of the subword dictionary",
)
parser.add_argument(
    "--data.word_dict_dir",
    type=str,
    default="runs/10000/words_dict.p",
    metavar="WORD_DICT",
    help="directory of the word dictionary",
)
parser.add_argument(
    "--data.char_dict_dir",
    type=str,
    default="runs/10000/chars_dict.p",
    metavar="	CHAR_DICT",
    help="directory of the character dictionary",
)

# model args
default_emb_dim = 32
parser.add_argument(
    "--model.emb_dim",
    type=int,
    default=default_emb_dim,
    metavar="EMBDIM",
    help="embedding dimension size (default: {})".format(default_emb_dim),
)
default_emb_mode = 1
parser.add_argument(
    "--model.emb_mode",
    type=int,
    default=default_emb_mode,
    metavar="EMBMODE",
    help="1: charCNN, 2: wordCNN, 3: char + wordCNN, 4: char-level wordCNN, 5: char + char-level wordCNN (default: {})".format(
        default_emb_mode
    ),
)

# test args
default_batch_size = 128
parser.add_argument(
    "--test.batch_size",
    type=int,
    default=default_batch_size,
    metavar="BATCHSIZE",
    help="Size of each test batch (default: {})".format(default_batch_size),
)

# log args
parser.add_argument(
    "--log.output_dir",
    type=str,
    default="runs/10000/",
    metavar="OUTPUTDIR",
    help="directory to save the test results",
)
parser.add_argument(
    "--log.checkpoint_dir",
    type=str,
    default="runs/10000/checkpoints/",
    metavar="CHECKPOINTDIR",
    help="directory of the learned model",
)

FLAGS = vars(parser.parse_args())
for key, val in FLAGS.items():
    print("{}={}".format(key, val))
#!SECTION

urls, labels = read_data(FLAGS["data.data_dir"])

x, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"])
word_x = get_words(x, word_reverse_dict, FLAGS["data.delimit_mode"], urls)

ngrams_dict = pickle.load(open(FLAGS["data.subword_dict_dir"], "rb"))
print("Size of subword vocabulary (train): {}".format(len(ngrams_dict)))
words_dict = pickle.load(open(FLAGS["data.word_dict_dir"], "rb"))
print("size of word vocabulary (train): {}".format(len(words_dict)))
ngramed_id_x, worded_id_x = ngram_id_x_from_dict(
    word_x, FLAGS["data.max_len_subwords"], ngrams_dict, words_dict
)
chars_dict = pickle.load(open(FLAGS["data.char_dict_dir"], "rb"))
chared_id_x = char_id_x(urls, chars_dict, FLAGS["data.max_len_chars"])

print("Number of testing urls: {}".format(len(labels)))

####################### EVALUATION ###########################


def test_step(model, x, emb_mode):
    if emb_mode == 1:
        inputs = [x[0]]
    elif emb_mode == 2:
        inputs = [x[0]]
    elif emb_mode == 3:
        inputs = [x[0], x[1]]
    elif emb_mode == 4:
        inputs = [x[0], x[1], x[2]]
    elif emb_mode == 5:
        inputs = [
            x[0].astype(np.int32),
            x[1].astype(np.int32),
            x[2].astype(np.int32),
            x[3],
        ]
    scores, predictions = model(inputs, training=False)
    return predictions.numpy(), scores.numpy()


def run_testing():
    cnn = TextCNN(
        char_ngram_vocab_size=len(ngrams_dict) + 1,
        word_ngram_vocab_size=len(words_dict) + 1,
        char_vocab_size=len(chars_dict) + 1,
        embedding_size=32,
        word_seq_len=200,
        char_seq_len=200,
        mode=5,
    )

    optimizer = tf.keras.optimizers.Adam(0.001)

    checkpoint_dir = FLAGS["log.checkpoint_dir"]
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=cnn)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    if FLAGS["model.emb_mode"] == 1:
        batches = batch_iter(
            list(chared_id_x), FLAGS["test.batch_size"], 1, shuffle=False
        )
    elif FLAGS["model.emb_mode"] == 2:
        batches = batch_iter(
            list(worded_id_x), FLAGS["test.batch_size"], 1, shuffle=False
        )
    elif FLAGS["model.emb_mode"] == 3:
        batches = batch_iter(
            list(zip(chared_id_x, worded_id_x)),
            FLAGS["test.batch_size"],
            1,
            shuffle=False,
        )
    elif FLAGS["model.emb_mode"] == 4:
        batches = batch_iter(
            list(zip(ngramed_id_x, worded_id_x)),
            FLAGS["test.batch_size"],
            1,
            shuffle=False,
        )
    elif FLAGS["model.emb_mode"] == 5:
        batches = batch_iter(
            list(zip(ngramed_id_x, worded_id_x, chared_id_x)),
            FLAGS["test.batch_size"],
            1,
            shuffle=False,
        )

    all_predictions = []
    all_scores = []

    nb_batches = int(len(labels) / FLAGS["test.batch_size"])
    if len(labels) % FLAGS["test.batch_size"] != 0:
        nb_batches += 1
    print("Number of batches in total: {}".format(nb_batches))
    batchs_index = tqdm(
        range(nb_batches),
        desc="emb_mode {} delimit_mode {} test_size {}".format(
            FLAGS["model.emb_mode"], FLAGS["data.delimit_mode"], len(labels)
        ),
        ncols=0,
    )
    for batch_index in batchs_index:
        batch = next(batches)

        if FLAGS["model.emb_mode"] == 1:
            x_char_seq = batch
        elif FLAGS["model.emb_mode"] == 2:
            x_word = batch
        elif FLAGS["model.emb_mode"] == 3:
            x_char_seq, x_word = zip(*batch)
        elif FLAGS["model.emb_mode"] == 4:
            x_char, x_word = zip(*batch)
        elif FLAGS["model.emb_mode"] == 5:
            x_char, x_word, x_char_seq = zip(*batch)

        x_batch = []
        if FLAGS["model.emb_mode"] in [1, 3, 5]:
            x_char_seq = pad_seq_in_word(x_char_seq, FLAGS["data.max_len_chars"])
            x_batch.append(x_char_seq)
        if FLAGS["model.emb_mode"] in [2, 3, 4, 5]:
            x_word = pad_seq_in_word(x_word, FLAGS["data.max_len_words"])
            x_batch.append(x_word)
        if FLAGS["model.emb_mode"] in [4, 5]:
            x_char, x_char_pad_idx = pad_seq(
                x_char,
                FLAGS["data.max_len_words"],
                FLAGS["data.max_len_subwords"],
                FLAGS["model.emb_dim"],
            )
            x_batch.extend([x_char, x_char_pad_idx])

        batch_predictions, batch_scores = test_step(
            cnn, x_batch, FLAGS["model.emb_mode"]
        )
        all_predictions = np.concatenate([all_predictions, batch_predictions])
        all_scores.extend(batch_scores)

        batchs_index.set_postfix()
    # print(type(all_outputs))
    # data_array = np.array(all_outputs, dtype="float32")
    # np.save(
    #     f"/home/jxlu/project/PhishHGMAE/data/phishing_1000/u_urlnet_feat.npy",
    #     data_array,
    # )

    if labels is not None:
        correct_preds = float(sum(all_predictions == labels))
        print("Accuracy: {}".format(correct_preds / float(len(labels))))

    save_test_result(labels, all_predictions, all_scores, FLAGS["log.output_dir"])


if __name__ == "__main__":
    run_testing()
