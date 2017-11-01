import tensorflow as tf

class SelfAttentive(object):
  '''
  Tensorflow implementation of 'A Structured Self Attentive Sentence Embedding'
  (https://arxiv.org/pdf/1703.03130.pdf)
  '''
  def build_graph(self, input_embed, u=100, d_a=100, r=100):
    with tf.variable_scope('SelfAttentive'):
      # Hyperparmeters from paper
      self.n = tf.shape(input_embed)[1]
      self.d_a = d_a
      self.u = u
      self.r = r

      initializer = tf.contrib.layers.xavier_initializer

      # Declare trainable variables
      # shape(W_s1) = d_a * 2u
      self.W_s1 = tf.get_variable('W_s1', shape=[self.d_a, 2*self.u])
      # shape(W_s2) = r * d_a
      self.W_s2 = tf.get_variable('W_s2', shape=[self.r, self.d_a])

      # BiRNN
      cell_fw = tf.contrib.rnn.BasicLSTMCell(u)
      cell_bw = tf.contrib.rnn.BasicLSTMCell(u)

      H, _, _ = tf.nn.static_bidirectional_rnn(
          cell_fw,
          cell_bw,
          input_embed)

      A = tf.nn.softmax(
          tf.scan(
            lambda a, x: tf.matmul(self.W_s2, x), 
            tf.tanh(
              tf.scan(
                lambda a, x: tf.matmul(self.W_s1, x),
                tf.transpose(H)))))

      self.M = tf.matmul(A, H)

      self.P = tf.square(
          tf.norm(tf.matmul(A, tf.transpose(A)) - tf.identity(A)))

  def trainable_vars(self):
    return [var for var in
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SelfAttentive')]
