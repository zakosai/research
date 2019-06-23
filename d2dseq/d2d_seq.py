import tensorflow as tf
import argparse
from dataset import Dataset, calc_recall
import numpy as np


class D2Dseq(object):
    def __init__(self, n_item_input, n_item_target, batch_size, n_user_input, n_user_target, seq_len,
                 max_target_sentence_length, go_id, eos_A, eos_B, n_layers=1):
        self.n_hidden = 256
        self.n_item_input = n_item_input
        self.n_item_target = n_item_target
        self.dec_emb_size = n_user_target
        self.enc_emb_size = n_user_input
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.mode = "train"
        self.go_id = go_id
        self.eos_A = eos_A
        self.eos_B = eos_B
        self.max_target_sentence_length = max_target_sentence_length
        self.n_layers = n_layers
        self.keep_prob = 0.7
        self.lr = 1e-4

    def encoder_LSTM(self, X, n_layers, seq_len):
        stack_cell = []
        for i in range(n_layers):
            with tf.variable_scope("encoder_%d"%i):
                cell = tf.contrib.rnn.LSTMCell(self.n_hidden, state_is_tuple=True)
                # cell = tf.contrib.rnn.AttentionCellWrapper(
                #     cell, attn_length=24, state_is_tuple=True)
                cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)
                stack_cell.append(cell)

        stack = tf.contrib.rnn.MultiRNNCell(stack_cell, state_is_tuple=True)

        # The second output is the last state and we will not use that
        outputs, last_state = tf.nn.dynamic_rnn(stack, X, seq_len, dtype=tf.float32)

        return outputs, last_state

    def process_decoder_input(self,target_data):
        size = tf.shape(target_data)[0]
        after_slice = tf.strided_slice(target_data, [0, 0], [size, -1], [1, 1])
        after_concat = tf.concat([tf.fill([size, 1], self.go_id), after_slice], 1)

        return after_concat

    def decoding_layer_train(self, encoder_state, dec_cell, dec_embed_input, target_sequence_length, max_summary_length,
                             output_layer, keep_prob):
        """
        Create a training process in decoding layer
        :return: BasicDecoderOutput containing training logits and sample_id
        """
        dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)

        # for only input layer
        helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)

        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer)

        # unrolling the decoder layer
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,
                                                          maximum_iterations=max_summary_length)
        return outputs

    def decoding_layer_infer(self, encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                             max_target_sequence_length, output_layer, batch_size, keep_prob):
        """
        Create a inference process in decoding layer
        :return: BasicDecoderOutput containing inference logits and sample_id
        """
        dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, tf.fill([batch_size], start_of_sequence_id),
                                                          end_of_sequence_id)

        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,
                                                          maximum_iterations=max_target_sequence_length)
        return outputs

    def decoding_layer(self, dec_embed_input, encoder_state, target_sequence_length, max_target_sequence_length, rnn_size,
                       num_layers, batch_size, keep_prob, decoding_embedding_size):
        """
        Create decoding layer
        :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
        """
        target_vocab_size = self.n_item_target
        dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
        # dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])

        with tf.variable_scope("decode"):
            output_layer = tf.layers.Dense(target_vocab_size)
            train_output = self.decoding_layer_train(encoder_state, cells, dec_embed_input, target_sequence_length,
                                                max_target_sequence_length, output_layer, keep_prob)

        with tf.variable_scope("decode", reuse=True):
            infer_output = self.decoding_layer_infer(encoder_state, cells, dec_embeddings, self.go_id,
                                                self.eos_B, max_target_sequence_length, output_layer, batch_size, keep_prob)

        return (train_output, infer_output)

    def build_model(self):
        """
            Build the Sequence-to-Sequence model
            :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
            """

        self.input_data = tf.placeholder(tf.float32, [None, None, self.enc_emb_size])
        self.target_data = tf.placeholder(tf.int32, [None, None])
        self.dec_emb_input = tf.placeholder(tf.float32, [None, None, self.dec_emb_size])
        self.target_sequence_length = tf.placeholder(tf.int32, [None])
        input_sequence_legth = tf.fill([tf.shape(self.input_data)[0]], tf.shape(self.input_data)[1])
        max_target_sentence_length = tf.shape(self.target_data)[1]

        enc_outputs, enc_states = self.encoder_LSTM(self.input_data, self.n_layers, input_sequence_legth)

        # dec_input = self.process_decoder_input(target_data)

        train_output, infer_output = self.decoding_layer(self.dec_emb_input, enc_states, self.target_sequence_length,
                                                   max_target_sentence_length, self.n_hidden, self.n_layers,
                                                    self.batch_size, self.keep_prob, self.dec_emb_size)

        training_logits = tf.identity(train_output.rnn_output, name='logits')
        self.inference_logits = tf.identity(infer_output.rnn_output, name='predictions')
        self.infer_output = infer_output

        masks = tf.sequence_mask(self.target_sequence_length, max_target_sentence_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function - weighted softmax cross entropy
            self.cost = tf.contrib.seq2seq.sequence_loss(training_logits, self.target_data, masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(self.lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(self.cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gradients)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', type=str, default="Health_Grocery", help='dataset name')
    parser.add_argument('--seq_length', type=int, default=20, help='sequence length encode')
    parser.add_argument('--iter', type=int, default=300, help='sequence length encode')
    parser.add_argument('--batch_size', type=int, default=500, help='sequence length encode')
    args = parser.parse_args()
    iter = args.iter
    batch_size = args.batch_size

    data = Dataset("data/%s"%args.data, args.seq_length)


    model = D2Dseq(data.n_item_A, data.n_item_B, batch_size, data.n_user, data.n_user, data.seq_len,
                   data.max_target_sequence, data.go, data.eos_A, data.eos_B)
    model.build_model()
    train_id = list(set(range(data.n_user)) - set(data.dataset['val_id']) - set(data.dataset['test_id']))
    train_no = len(train_id)
    test_id = np.array(data.dataset['test_id'])
    test_no = len(test_id)
    val_id = np.array(data.dataset['val_id'])
    val_no = len(val_id)
    f = open("experiment/%s/result.txt" % args.data, "a")
    f.write("-------------------------\n")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=5)
    max_recall = 0
    result = [0, 0, 0, 0]

    for i in range(1, iter):
        shuffle_idx = np.random.permutation(train_id)
        for j in range(int(train_no/batch_size)):
            input_emb_batch, target_batch, target_emb_batch, target_seq = data.create_batch(shuffle_idx[j*batch_size:(j+1)*batch_size])
            feed = {model.input_data: input_emb_batch,
                    model.target_data: target_batch,
                    model.dec_emb_input: target_emb_batch,
                    model.target_sequence_length: target_seq}

            loss, _ = sess.run([model.cost, model.train_op], feed_dict=feed)

        print("Loss last batch: %f"%loss)

        if i%1 == 0:
            for j in range(int(val_no / batch_size)+1):
                idx = list(range(j * batch_size, min((j + 1) * batch_size, val_no)))
                input_emb_batch, target_batch, target_emb_batch, target_seq = data.create_batch(val_id[idx])
                feed = {model.input_data: input_emb_batch,
                        model.target_data: target_batch,
                        model.dec_emb_input: target_emb_batch,
                        model.target_sequence_length: target_seq}
                infer, loss = sess.run([model.inference_logits, model.cost], feed_dict=feed)
                tmp_infer = []
                for i in range(len(target_seq)):
                    tmp_infer.append(infer[i, target_seq[i]-1, :])
                infer = np.array(tmp_infer).reshape((len(idx), data.n_item_B))
                if j == 0:
                    target = target_batch
                    infer_all = infer
                else:
                    target = np.concatenate((target, target_batch), axis=0)
                    infer_all = np.concatenate((infer_all, infer), axis=0)
            print(target.shape, infer_all.shape)
            recall, hit, ndcg = calc_recall(infer_all, target[:, :-1], target[:, -1])
            print("iter: %d recall: %f, hit: %f, ndcg: %f" % (i, recall, hit, ndcg))
            if recall > max_recall:
                max_recall = recall
                for j in range(int(test_no / batch_size) + 1):
                    idx = list(range(j * batch_size, min((j + 1) * batch_size, test_no)))
                    input_emb_batch, target_batch, target_emb_batch, target_seq = data.create_batch(test_id[idx])
                    feed = {model.input_data: input_emb_batch, model.target_data: target_batch,
                            model.dec_emb_input: target_emb_batch, model.target_sequence_length: target_seq}
                    infer, loss = sess.run([model.inference_logits, model.cost], feed_dict=feed)
                    tmp_infer = []
                    for i in range(len(target_seq)):
                        tmp_infer.append(infer[i, target_seq[i] - 1, :])
                    infer = np.array(tmp_infer).reshape((len(idx), data.n_item_B))
                    if j == 0:
                        target = target_batch
                        infer_all = infer
                    else:
                        target = np.concatenate((target, target_batch), axis=0)
                        infer_all = np.concatenate((infer_all, infer), axis=0)
                print(target.shape, infer_all.shape)
                recall, hit, ndcg = calc_recall(infer_all, target[:, :-1], target[:, -1])
                print("iter: %d recall: %f, hit: %f, ndcg: %f" % (i, recall, hit, ndcg))
                if recall > result[1]:
                    result = [i, recall, hit, ndcg]
            model.train = True
            if i % 100 == 0 and model.lr > 1e-6:
                model.lr /= 10
                print("decrease lr to %f" % model.lr)
        f.write("iter: %d - recall: %f - hit: %f - ndcg: %f\n" % (result[0], result[1], result[2], result[3]))
        f.write("Last result- recall: %d - hit: %f - ndcg:%f\n" % (recall, hit, ndcg))
        print(max_recall)



if __name__ == '__main__':
    main()







