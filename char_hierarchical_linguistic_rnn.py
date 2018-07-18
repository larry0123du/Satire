import evaluations as evals
import data_utilities as du
import utilities as util
import minibatch as mb
import theano_function as tf
import time
import sys
import numpy as np
import lasagne
import logging
import arg_parser as ap

import glob
from pickle import load, dump


def main(args):
    logging.info("loading data...")
    fake_train, fake_dev, fake_test = du.load_fake(False, False, False)
    true_train, true_dev, true_test = du.load_true(False, False, False)
    if args.debug:
        true_train = [true_train[0][:100]]
        fake_train = fake_train[:10]
        true_dev = true_dev[:100]
        fake_dev = fake_dev[:10]
        true_test = true_test[:100]
        fake_test = fake_test[:10]
    if args.rnn_type == 'gru':
        args.rnn = lasagne.layers.GRULayer
    elif args.rnn_type == 'lstm':
        args.rnn = lasagne.layers.LSTMLayer
    else:
        args.rnn = lasagne.layers.RecurrentLayer

    logging.info('train: {} {}'.format(len(fake_train), len(true_train)))
    logging.info('dev: {} {}'.format(len(fake_dev), len(true_dev)))
    logging.info('test: {} {}'.format(len(fake_test), len(true_test)))

    logging.info("building dictionary...")
    # word_dict, char_dict = util.build_dict(None, max_words=0, dict_file=["word_dict", "char_dict"])
    # change july 10th
    '''
    BASE = '/homes/du113/scratch/'
    list_of_files = glob.glob(BASE + 'data/text/true/*train*.txt') + glob.glob(BASE + 'data/text/fake/*train*.txt')
    docs = []
    for fi in list_of_files:
        docs += du.load_sent(fi)

    word_dict, char_dict = util.build_dict(docs)

    # saving word_dict and char_dict
    BASE = '/homes/du113/scratch/satire-models/'
    with open(BASE + 'new_word_dict', 'wb') as fid:
        dump(word_dict, fid)

    with open(BASE + 'new_char_dict', 'wb') as fid:
        dump(char_dict, fid)
    '''
    BASE = '/homes/du113/scratch/satire-models/'
    with open(BASE + 'word_dict', 'rb') as fid:
        word_dict = load(fid)

    with open(BASE + 'char_dict', 'rb') as fid:
        char_dict = load(fid)
# end of change

    logging.info("creating embedding matrix...")
    word_embed = util.words2embedding(word_dict, 100, args.embedding_file)
    char_embed = util.char2embedding(char_dict, 30)
    (args.word_vocab_size, args.word_embed_size) = word_embed.shape
    (args.char_vocab_size, args.char_embed_size) = char_embed.shape
    logging.info("compiling Theano function...")
    '''
    no ling
    att_fn, eval_fn, train_fn, params = \
        tf.char_hierarchical_linguistic_fn(args, word_embed, char_embed, values=None)
    '''
    # added on july 11th
    # modified on july 17th
    if args.model_file:
        logging.info('using pretrained model from {}'.format(args.model_file))
        payload = util.load_params(args.model_file)
        weights = payload['params']
        n_updates = payload['n_updates']

    else:
        logging.info('initializing new network')
        n_updates = 0
        weights = None
    att_fn, eval_fn, train_fn, params = \
        tf.char_hierarchical_fn(args, word_embed, char_embed, values=weights)

    logging.info("batching examples...")
    dev_examples = mb.vec_minibatch(fake_dev + true_dev, word_dict, char_dict, args, False, True, False, False)
# debugging
    logging.info('# dev examples: {}'.format(len(dev_examples)))

    test_examples = mb.vec_minibatch(fake_test + true_test, word_dict, char_dict, args, False, True, False, False)
    if not args.test_only:
        train_examples = mb.train_doc_minibatch(fake_train, true_train, args, over_sample=True)
    logging.info("checking network...")
    # dev_acc = evals.eval_batch(eval_fn, dev_examples, word_dict, char_dict, args)
    dev_acc = evals.eval_vec_batch(eval_fn, dev_examples, True, False, False)
    print('Dev A: %.2f P:%.2f R:%.2f F:%.2f' % dev_acc)
    test_acc = evals.eval_vec_batch(eval_fn, test_examples, True, False, False)
    print('Performance on Test set: A: %.2f P:%.2f R:%.2f F:%.2f' % test_acc)

    # added july 11th
    if not args.test_only:
        prev_fsc = 0
        stop_count = 0
        best_fsc = 0
        best_acc = 0
        logging.info("training %d examples" % len(train_examples))
        start_time = time.time()
        # n_updates = 0

        gpu_not_available = True
        for epoch in range(args.epoches):
            np.random.shuffle(train_examples)
            if epoch > 3:
                logging.info("compiling Theano function again...")
                args.learning_rate *= 0.9
                '''
                no ling
                att_fn, eval_fn, train_fn, params = \
                    tf.char_hierarchical_linguistic_fn(args, word_embed, char_embed, values=[x.get_value() for x in params])
                '''
                att_fn, eval_fn, train_fn, params = \
                    tf.char_hierarchical_fn(args, word_embed, char_embed, values=[x.get_value() for x in params])

            for batch_x, _ in train_examples:
                '''
                no ling
                # batch_x, batch_sent, batch_doc, batch_y = zip(*batch_x)
                '''
                batch_x, batch_y = zip(*batch_x)

                batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
                batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn = \
                    util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
                '''
                no ling
                batch_sent = util.sent_ling_padding(list(batch_sent), args.max_sent, args.max_ling)
                batch_doc = util.doc_ling_padding(list(batch_doc), args.max_ling)
                '''
                batch_y = np.array(list(batch_y))
                '''
                no ling
                train_loss = train_fn(batch_rnn, batch_cnn, batch_word_mask,
                                    batch_sent_mask, batch_sent, batch_doc, batch_y)
                '''
                '''
                train_loss = train_fn(batch_rnn, batch_cnn, batch_word_mask,
                                    batch_sent_mask, batch_y)
                '''
                # added July 10th
                while gpu_not_available:
                    try:
                        train_loss = train_fn(batch_rnn, batch_cnn, batch_word_mask,
                                            batch_sent_mask, batch_y)
                        gpu_not_available = False
                    except pygpu.gpuarray.GpuArrayException:
                        'GPU is not available at the moment'
                        sleep(600)

                n_updates += 1

            # chagned july 10th
            # if epoch > 6:
            logging.info('Epoch = %d, loss = %.2f, elapsed time = %.2f (s)' %
                            (epoch, train_loss, time.time() - start_time))
            # dev_acc = evals.eval_batch(eval_fn, dev_examples, word_dict, char_dict, args)
            ''' no ling
            dev_acc = evals.eval_vec_batch(eval_fn, dev_examples)
            '''
            dev_acc = evals.eval_vec_batch(eval_fn, dev_examples, True, False, False)

            logging.info('Dev A: %.2f P:%.2f R:%.2f F:%.2f' % dev_acc)
            if dev_acc[3] >= best_fsc and dev_acc[0] > best_acc:
                best_fsc = dev_acc[3]
                best_acc = dev_acc[0]
                logging.info('Best dev f1: epoch = %d, n_udpates = %d, f1 = %.2f %%'
                                % (epoch, n_updates, dev_acc[3]))
                record = 'Best dev accuracy: epoch = %d, n_udpates = %d ' % \
                            (epoch, n_updates) + ' Dev A: %.2f P:%.2f R:%.2f F:%.2f' % dev_acc
                # test_acc = evals.eval_batch(eval_fn, test_examples, word_dict, char_dict, args)
                ''' no ling
                test_acc = evals.eval_vec_batch(eval_fn, test_examples)
                '''
                test_acc = evals.eval_vec_batch(eval_fn, test_examples, True, False, False)

                print('Performance on Test set: A: %.2f P:%.2f R:%.2f F:%.2f' % test_acc)
                ''' no ling
                if test_acc[3] > 91.4:
                '''
                if test_acc[3] > 88.0:
                    util.save_params(BASE + 'july_17_new_data_char_hierarchical_rnn_params_%.2f_%.2f' % (dev_acc[3], test_acc[3]), params,
                                        epoch=epoch, n_updates=n_updates)
            if prev_fsc > dev_acc[3]:
                stop_count += 1
            else:
                stop_count = 0
            if stop_count == 6:
                print("stopped")
                break
            prev_fsc = dev_acc[3]

        print(record)
        print('Performance on Test set: A: %.2f P:%.2f R:%.2f F:%.2f' % test_acc)


if __name__ == '__main__':
    args = ap.get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    logging.info(' '.join(sys.argv))
    # args.debug = True
    '''
    no ling
    args.doc_ling_nonlinear = True
    '''
    args.dropout_rate = 0.5
    # args.word_att = 'dot'
    args.learning_rate = 0.3
    print(args.word_att, args.learning_rate, args.dropout_rate)
    main(args)
