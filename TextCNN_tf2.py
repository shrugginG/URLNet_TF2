import tensorflow as tf


class TextCNN(tf.keras.Model):
    def __init__(
        self,
        char_ngram_vocab_size,  # len(ngrams_dict) + 1
        word_ngram_vocab_size,  # len(words_dict) + 1
        char_vocab_size,  # len(chars_dict) + 1
        word_seq_len,  # 200
        char_seq_len,  # 200
        embedding_size,  # 32
        l2_reg_lambda=0,
        filter_sizes=[3, 4, 5, 6],
        mode=0,
    ):
        super(TextCNN, self).__init__()
        self.mode = mode
        self.l2_reg_lambda = l2_reg_lambda
        self.filter_sizes = filter_sizes
        self.num_filters = 256
        self.word_seq_len = word_seq_len
        self.char_seq_len = char_seq_len
        self.embedding_size = embedding_size
        self.l2_loss = tf.constant(0.0)

        if mode == 4 or mode == 5:  # character-level word CNN [CHAR Embedding Matrix 2]
            # Init char embedding
            self.char_emb_w = tf.Variable(
                tf.random.uniform([char_ngram_vocab_size, embedding_size], -1.0, 1.0),
                name="char_emb_w",
            )

        if (
            mode == 2 or mode == 3 or mode == 4 or mode == 5
        ):  # word-based [WORD Embedding Matrix]
            # Init word embedding
            self.word_emb_w = tf.Variable(
                tf.random.uniform([word_ngram_vocab_size, embedding_size], -1.0, 1.0),
                name="word_emb_w",
            )
        if (
            mode == 1 or mode == 3 or mode == 5
        ):  # character-based [CHAR Embedding Matrix 1]
            self.char_seq_emb_w = tf.Variable(
                tf.random.uniform([char_vocab_size, embedding_size], -1.0, 1.0),
                name="char_seq_emb_w",
            )

        self.conv_layers_word = []
        self.conv_layers_char = []

        for filter_size in self.filter_sizes:
            conv_layer_word = tf.keras.layers.Conv2D(
                self.num_filters,
                (filter_size, self.embedding_size),
                activation="relu",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                bias_initializer=tf.constant_initializer(0.1),
            )
            self.conv_layers_word.append(conv_layer_word)

            conv_layer_char = tf.keras.layers.Conv2D(
                self.num_filters,
                (filter_size, self.embedding_size),
                activation="relu",
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                bias_initializer=tf.constant_initializer(0.1),
            )
            self.conv_layers_char.append(conv_layer_char)

        self.dropout = tf.keras.layers.Dropout(0.5)

        self.ww = self.add_weight(
            "ww",
            shape=(self.num_filters * len(filter_sizes), 512),
            initializer=tf.keras.initializers.GlorotUniform(),
        )
        self.bw = self.add_weight(
            "bw", shape=(512,), initializer=tf.constant_initializer(0.1)
        )
        self.wc = self.add_weight(
            "wc",
            shape=(self.num_filters * len(filter_sizes), 512),
            initializer=tf.keras.initializers.GlorotUniform(),
        )
        self.bc = self.add_weight(
            "bc", shape=(512,), initializer=tf.constant_initializer(0.1)
        )

        self.w0 = self.add_weight(
            "w0", shape=[1024, 512], initializer=tf.keras.initializers.GlorotUniform()
        )
        self.b0 = self.add_weight(
            "b0", shape=[512], initializer=tf.constant_initializer(0.1)
        )
        self.w1 = self.add_weight(
            "w1", shape=[512, 256], initializer=tf.keras.initializers.GlorotUniform()
        )
        self.b1 = self.add_weight(
            "b1", shape=[256], initializer=tf.constant_initializer(0.1)
        )
        self.w2 = self.add_weight(
            "w2", shape=[256, 128], initializer=tf.keras.initializers.GlorotUniform()
        )
        self.b2 = self.add_weight(
            "b2", shape=[128], initializer=tf.constant_initializer(0.1)
        )
        self.w = self.add_weight(
            "w", shape=[128, 2], initializer=tf.keras.initializers.GlorotUniform()
        )
        self.b = self.add_weight(
            "b", shape=[2], initializer=tf.constant_initializer(0.1)
        )

    def call(self, inputs, training=False):
        input_x_char_seq, input_x_word, input_x_char, input_x_char_pad_idx = inputs
        # input_x_char, input_x_char_pad_idx, input_x_word, input_x_char_seq, input_y = inputs
        if self.mode == 4 or self.mode == 5:
            embedded_x_char = tf.nn.embedding_lookup(self.char_emb_w, input_x_char)
            embedded_x_char = tf.multiply(embedded_x_char, input_x_char_pad_idx)

        if self.mode == 2 or self.mode == 3 or self.mode == 4 or self.mode == 5:
            embedded_x_word = tf.nn.embedding_lookup(self.word_emb_w, input_x_word)

        if self.mode == 1 or self.mode == 3 or self.mode == 5:
            embedded_x_char_seq = tf.nn.embedding_lookup(
                self.char_seq_emb_w, input_x_char_seq
            )

        if self.mode == 4 or self.mode == 5:
            sum_ngram_x_char = tf.reduce_sum(embedded_x_char, 2)
            sum_ngram_x = tf.add(sum_ngram_x_char, embedded_x_word)

        if self.mode == 4 or self.mode == 5:
            sum_ngram_x_expanded = tf.expand_dims(sum_ngram_x, -1)
        if self.mode == 2 or self.mode == 3:
            sum_ngram_x_expanded = tf.expand_dims(embedded_x_word, -1)
        if self.mode == 1 or self.mode == 3 or self.mode == 5:
            char_x_expanded = tf.expand_dims(embedded_x_char_seq, -1)

        pooled_x = []
        if self.mode == 2 or self.mode == 3 or self.mode == 4 or self.mode == 5:
            for conv_layer in self.conv_layers_word:
                conv = conv_layer(sum_ngram_x_expanded)
                pooled = tf.keras.layers.MaxPool2D(
                    (self.word_seq_len - conv_layer.kernel_size[0] + 1, 1),
                    strides=(1, 1),
                    padding="valid",
                )(conv)
                pooled_x.append(pooled)

            num_filters_total = self.num_filters * len(self.filter_sizes)
            h_pool = tf.concat(pooled_x, 3)
            x_flat = tf.reshape(h_pool, [-1, num_filters_total])
            h_drop = self.dropout(x_flat, training=training)

        pooled_char_x = []
        if self.mode == 1 or self.mode == 3 or self.mode == 5:
            for conv_layer in self.conv_layers_char:
                conv = conv_layer(char_x_expanded)
                pooled = tf.keras.layers.MaxPool2D(
                    (self.char_seq_len - conv_layer.kernel_size[0] + 1, 1),
                    strides=(1, 1),
                    padding="valid",
                )(conv)
                pooled_char_x.append(pooled)

            num_filters_total = self.num_filters * len(self.filter_sizes)
            h_char_pool = tf.concat(pooled_char_x, 3)
            char_x_flat = tf.reshape(h_char_pool, [-1, num_filters_total])
            char_h_drop = self.dropout(char_x_flat, training=training)

        if self.mode == 3 or self.mode == 5:
            word_output = tf.nn.bias_add(tf.linalg.matmul(h_drop, self.ww), self.bw)
            char_output = tf.nn.bias_add(
                tf.linalg.matmul(char_h_drop, self.wc), self.bc
            )
            conv_output = tf.concat([word_output, char_output], 1)
        elif self.mode == 2 or self.mode == 4:
            conv_output = h_drop
        elif self.mode == 1:
            conv_output = char_h_drop

        output0 = tf.nn.relu(
            tf.nn.bias_add(tf.linalg.matmul(conv_output, self.w0), self.b0)
        )
        output1 = tf.nn.relu(
            tf.nn.bias_add(tf.linalg.matmul(output0, self.w1), self.b1)
        )
        output2 = tf.nn.relu(
            tf.nn.bias_add(tf.linalg.matmul(output1, self.w2), self.b2)
        )

        scores = tf.nn.bias_add(tf.linalg.matmul(output2, self.w), self.b)
        predictions = tf.argmax(scores, 1, name="predictions")

        return scores, predictions

    def compute_loss(self, scores, input_y):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
        return tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

    def compute_accuracy(self, predictions, input_y):
        correct_preds = tf.equal(predictions, tf.argmax(input_y, 1))
        return tf.reduce_mean(tf.cast(correct_preds, "float"), name="accuracy")
