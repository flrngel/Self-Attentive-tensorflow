import tensorflow as tf

class SelfAttentive(object):
  '''
  Tensorflow implementation of 'A Structured Self Attentive Sentence Embedding'
  (https://arxiv.org/pdf/1703.03130.pdf)
  '''
  def build_graph(self, n=60, d=100, u=128, d_a=350, r=30):
    with tf.variable_scope('SelfAttentive'):
      # Hyperparmeters from paper
      self.n = n
      self.d = d
      self.d_a = d_a
      self.u = u
      self.r = r

      initializer = tf.contrib.layers.xavier_initializer()

      embedding = tf.get_variable('embedding', shape=[100000, self.d],
          initializer=initializer)
      self.input_pl = tf.placeholder(tf.int32, shape=[None, self.n])
      input_embed = tf.nn.embedding_lookup(embedding, self.input_pl)

      # Declare trainable variables
      # shape(W_s1) = d_a * 2u
      self.W_s1 = tf.get_variable('W_s1', shape=[self.d_a, 2*self.u],
          initializer=initializer)
      # shape(W_s2) = r * d_a
      self.W_s2 = tf.get_variable('W_s2', shape=[self.r, self.d_a],
          initializer=initializer)

      # BiRNN
      self.batch_size = batch_size = tf.shape(self.input_pl)[0]

      cell_fw = tf.contrib.rnn.LSTMCell(u)
      cell_bw = tf.contrib.rnn.LSTMCell(u)

      H, _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw,
          cell_bw,
          input_embed,
          dtype=tf.float32)
      H = tf.concat([H[0], H[1]], axis=2)

      self.A = A = tf.nn.softmax(
          tf.map_fn(
            lambda x: tf.matmul(self.W_s2, x), 
            tf.tanh(
              tf.map_fn(
                lambda x: tf.matmul(self.W_s1, tf.transpose(x)),
                H))))

      self.M = tf.matmul(A, H)

      A_T = tf.transpose(A, perm=[0, 2, 1])
      tile_eye = tf.tile(tf.eye(r), [batch_size, 1])
      tile_eye = tf.reshape(tile_eye, [-1, r, r])
      AA_T = tf.matmul(A, A_T) - tile_eye
      self.P = tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))

  def trainable_vars(self):
    return [var for var in
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SelfAttentive')]
