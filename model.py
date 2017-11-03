import tensorflow as tf

class SelfAttentive(object):
  '''
  Tensorflow implementation of 'A Structured Self Attentive Sentence Embedding'
  (https://arxiv.org/pdf/1703.03130.pdf)
  '''
  def build_graph(self, n=60, d=100, u=128, d_a=100, r=100):
    with tf.variable_scope('SelfAttentive'):
      # Hyperparmeters from paper
      self.n = n
      self.d = d
      self.d_a = d_a
      self.u = u
      self.r = r

      initializer = tf.random_normal_initializer(stddev=0.1)

      embedding = tf.get_variable('embedding', shape=[100000, self.d],
          initializer=initializer)
      self.input_pl = tf.placeholder(tf.int32, shape=[None, self.n])
      input_embed = tf.nn.embedding_lookup(embedding, self.input_pl)
      print(input_embed)

      # Declare trainable variables
      # shape(W_s1) = d_a * 2u
      self.W_s1 = tf.get_variable('W_s1', shape=[self.d_a, 2*self.u],
          initializer=initializer)
      # shape(W_s2) = r * d_a
      self.W_s2 = tf.get_variable('W_s2', shape=[self.r, self.d_a],
          initializer=initializer)

      # BiRNN
      self.batch_size = tf.shape(self.input_pl)[0]

      cell_fw = tf.contrib.rnn.LSTMCell(u)
      cell_bw = tf.contrib.rnn.LSTMCell(u)

      H, _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw,
          cell_bw,
          input_embed,
          dtype=tf.float32)
      H = tf.concat([H[0], H[1]], axis=2)

      A = tf.nn.softmax(
          tf.map_fn(
            lambda x: tf.matmul(self.W_s2, x), 
            tf.tanh(
              tf.map_fn(
                lambda x: tf.matmul(self.W_s1, tf.transpose(x)),
                H))))

      self.M = tf.matmul(A, H)

      self.P = tf.square(tf.norm(tf.map_fn(lambda Ax:
          tf.matmul(Ax, tf.transpose(Ax)) - tf.eye(d_a), A),
          ord='fro', axis=[-2, -1]))

  def trainable_vars(self):
    return [var for var in
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SelfAttentive')]
