
import numpy
import tensorflow as tf
import pickle

import sys
import os

import math

USC_EMAIL = 'pradeepl@usc.edu'  # TODO(student): Fill to compete on rankings.
PASSWORD = '0d0d6dae842d226e'  # TODO(student): You will be given a password via email.
TRAIN_TIME_MINUTES = 11


#-------------------------------
# Best Hyperparam (SO FAR)
# Japanese: 94%
# Italian:  94%

# batch_size = 256
# learn_rate = 9e-3
# embed_dims = 10
# state_size = 20
# l2_reg = 1e-4 (On embedding)
#-------------------------------

# Hyper-params
BATCH_SIZE = 256
LEARN_RATE = 9e-3

# Read online that Adam has a theoretical sqrt(t) decay
# Regardless, none of the decay schemes worked
# Wasn't able to make

# For exponential decay
# DECAY_RATE = 2e-3  
# K = 0.3 

# For step decay
# Not used
# For step3
# EPOCHS_DROP = 5
# DROP_RATE = 0.5

EMBED_DIMS = 10
STATE_SIZE = 20

L2_REG = 1e-4

# Haven't tried sparse dropout
# But increasing below doesn't seem to help

# DROP_FC = 0     # Full-connected layer dropout
# DROP_EMBED = 0  # Embedding layer dropout

class DatasetReader(object):

  # TODO(student): You must implement this.
  @staticmethod
  def ReadFile(filename, term_index, tag_index):
    """Reads file into dataset, while populating term_index and tag_index.
   
    Args:
      filename: Path of text file containing sentences and tags. Each line is a
        sentence and each term is followed by "/tag". Note: some terms might
        have a "/" e.g. my/word/tag -- the term is "my/word" and the last "/"
        separates the tag.
      term_index: dictionary to be populated with every unique term (i.e. before
        the last "/") to point to an integer. All integers must be utilized from
        0 to number of unique terms - 1, without any gaps nor repetitions.
      tag_index: same as term_index, but for tags.

    Return:
      The parsed file as a list of lists: [parsedLine1, parsedLine2, ...]
      each parsedLine is a list: [(term1, tag1), (term2, tag2), ...] 
    """
    
    with open(filename) as fp:
      data = fp.readlines()

    parsed_file = []

    for line in data:
      tokens = line.split()
      
      parsed_line = []

      for tok in tokens:
        term, tag = tok.rsplit("/", 1)

        if term not in term_index:
          term_index[term] = len(term_index)

        if tag not in tag_index:
          tag_index[tag] = len(tag_index)

        parsed_line.append((term_index[term], tag_index[tag]))

      parsed_file.append(parsed_line)

    return parsed_file

  # TODO(student): You must implement this.
  @staticmethod
  def BuildMatrices(dataset):
    """Converts dataset [returned by ReadFile] into numpy arrays for tags, terms, and lengths.

    Args:
      dataset: Returned by method ReadFile. It is a list (length N) of lists:
        [sentence1, sentence2, ...], where every sentence is a list:
        [(word1, tag1), (word2, tag2), ...], where every word and tag are integers.

    Returns:
      Tuple of 3 numpy arrays: (terms_matrix, tags_matrix, lengths_arr)
        terms_matrix: shape (N, T) int64 numpy array. Row i contains the word
          indices in dataset[i].
        tags_matrix: shape (N, T) int64 numpy array. Row i contains the tag
          indices in dataset[i].
        lengths: shape (N) int64 numpy array. Entry i contains the length of
          sentence in dataset[i].

      T is the maximum length. For example, calling as:
        BuildMatrices([[(1,2), (4,10)], [(13, 20), (3, 6), (7, 8), (3, 20)]])
      i.e. with two sentences, first with length 2 and second with length 4,
      should return the tuple:
      (
        [[1, 4, 0, 0],    # Note: 0 padding.
         [13, 3, 7, 3]],

        [[2, 10, 0, 0],   # Note: 0 padding.
         [20, 6, 8, 20]], 

        [2, 4]
      )
    """
    lengths = []
    max_length = 0

    for line in dataset:
      lengths.append(len(line))

      if len(line) > max_length:
        max_length = len(line)

    lengths = numpy.array(lengths)

    terms_matrix = []
    tags_matrix = []

    for line in dataset:
      terms = [line[idx][0] if idx < len(line) else 0 for idx in range(max_length)]
      terms_matrix.append(terms)
      
      tags  = [line[idx][1] if idx < len(line) else 0 for idx in range(max_length)] 
      tags_matrix.append(tags)

    terms_matrix = numpy.array(terms_matrix)
    tags_matrix  = numpy.array(tags_matrix)

    return terms_matrix, tags_matrix, lengths

  @staticmethod
  def ReadData(train_filename, test_filename=None):
    """Returns numpy arrays and indices for train (and optionally test) data.

    NOTE: Please do not change this method. The grader will use an identitical
    copy of this method (if you change this, your offline testing will no longer
    match the grader).

    Args:
      train_filename: .txt path containing training data, one line per sentence.
        The data must be tagged (i.e. "word1/tag1 word2/tag2 ...").
      test_filename: Optional .txt path containing test data.

    Returns:
      A tuple of 3-elements or 4-elements, the later iff test_filename is given.
      The first 2 elements are term_index and tag_index, which are dictionaries,
      respectively, from term to integer ID and from tag to integer ID. The int
      IDs are used in the numpy matrices.
      The 3rd element is a tuple itself, consisting of 3 numpy arrsys:
        - train_terms: numpy int matrix.
        - train_tags: numpy int matrix.
        - train_lengths: numpy int vector.
        These 3 are identical to what is returned by BuildMatrices().
      The 4th element is a tuple of 3 elements as above, but the data is
      extracted from test_filename.
    """
    term_index = {'__oov__': 0}  # Out-of-vocab is term 0.
    tag_index = {}
    
    train_data = DatasetReader.ReadFile(train_filename, term_index, tag_index)
    train_terms, train_tags, train_lengths = DatasetReader.BuildMatrices(train_data)
    
    if test_filename:
      test_data = DatasetReader.ReadFile(test_filename, term_index, tag_index)
      test_terms, test_tags, test_lengths = DatasetReader.BuildMatrices(test_data)

      if test_tags.shape[1] < train_tags.shape[1]:
        diff = train_tags.shape[1] - test_tags.shape[1]
        zero_pad = numpy.zeros(shape=(test_tags.shape[0], diff), dtype='int64')
        test_terms = numpy.concatenate([test_terms, zero_pad], axis=1)
        test_tags = numpy.concatenate([test_tags, zero_pad], axis=1)
      elif test_tags.shape[1] > train_tags.shape[1]:
        diff = test_tags.shape[1] - train_tags.shape[1]
        zero_pad = numpy.zeros(shape=(train_tags.shape[0], diff), dtype='int64')
        train_terms = numpy.concatenate([train_terms, zero_pad], axis=1)
        train_tags = numpy.concatenate([train_tags, zero_pad], axis=1)

      return (term_index, tag_index,
              (train_terms, train_tags, train_lengths),
              (test_terms, test_tags, test_lengths))
    else:
      return term_index, tag_index, (train_terms, train_tags, train_lengths)


class SequenceModel(object):

  def __init__(self, max_length=310, num_terms=1000, num_tags=40, embed_dims=EMBED_DIMS, state_size=STATE_SIZE):
    """Constructor. You can add code but do not remove any code.

    The arguments are arbitrary: when you are training on your own, PLEASE set
    them to the correct values (e.g. from main()).

    Args:
      max_lengths: maximum possible sentence length.
      num_terms: the vocabulary size (number of terms).
      num_tags: the size of the output space (number of tags).

    You will be passed these arguments by the grader script.
    """
    self.max_length = max_length
    self.state_size = state_size
    self.num_terms = num_terms
    self.num_tags = num_tags
    self.embed_dims = embed_dims

    self.x = tf.placeholder(tf.int64, [None, self.max_length], 'X')
    self.y = tf.placeholder(tf.int64, [None, self.max_length], 'y')
    self.batch_size = tf.placeholder(tf.int64, [], 'batch_size')
    self.lengths = tf.placeholder(tf.int64, [None], 'lengths')
    #self.is_training = tf.placeholder(tf.bool, 'training')

    self.iteration = 0
  
  # TODO(student): You must implement this.
  def lengths_vector_to_binary_matrix(self, length_vector):
    """Returns a binary mask (as float32 tensor) from (vector) int64 tensor.
    
    Specifically, the return matrix B will have the following:
      B[i, :lengths[i]] = 1 and B[i, lengths[i]:] = 0 for each i.
    However, since we are using tensorflow rather than numpy in this function,
    you cannot set the range as described.
    """
    return tf.sequence_mask(length_vector, maxlen=self.max_length, dtype=tf.float32, name=None)

  # TODO(student): You must implement this.
  def save_model(self, filename):
    """Saves model to a file."""
    to_save = {var.name: var for var in tf.global_variables()}
    pickle.dump(self.sess.run(to_save), open('trained.pkl', 'w'))

  # TODO(student): You must implement this.
  def load_model(self, filename):
    """Loads model from a file."""
    to_load = pickle.load(open('trained.pkl'))
    assign_ops = [var.assign(to_load[var.name]) for var in tf.global_variables()]
    self.sess.run(assign_ops)

  # TODO(student): You must implement this.
  def build_inference(self):
    """Build the expression from (self.x, self.lengths) to (self.logits).
    
    Please do not change or override self.x nor self.lengths in this function.

    Hint:
      - Use lengths_vector_to_binary_matrix
      - You might use tf.reshape, tf.cast, and/or tensor broadcasting.
    """
    # TODO(student): make logits an RNN on x.

    # Initialize the embeddings
    embed = tf.get_variable('embedding', shape=[self.num_terms, self.embed_dims], dtype=tf.float32)

    # Initialize the weights for the FC layer of the bidirectional RNN/LSTM/GRU 
    W = tf.get_variable('W',shape=[2*self.state_size, self.num_tags], initializer=tf.truncated_normal_initializer(stddev=0.01), dtype=tf.float32)
    b = tf.get_variable('b',shape=[self.num_tags], dtype=tf.float32)

    # Add l2 reglarization to keep embedding weights low
    r_term = L2_REG*tf.reduce_sum(tf.multiply(embed, embed))
    tf.losses.add_loss(r_term, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

    # Dropout doesn't help
    # embed = tf.keras.layers.Dropout(DROP_EMBED)(embed, training=self.is_training)

    xemb = tf.nn.embedding_lookup(embed, self.x)

    '''
    The remains of vanilla rnn: ~90/91 percent accuracy!

    rnn_cell = tf.keras.layers.SimpleRNNCell(self.state_size)

    fcl = tf.keras.layers.Dense(self.num_tags, activation=None) # Softmax takes care of activation
    drop = tf.keras.layers.Dropout(DROP_FC)

    a_t = tf.zeros(shape=[1, self.state_size]) # state_size
    y_hist = []

    for t in range(self.max_length): # max_length

      a_t = rnn_cell(xemb[:,t,:], [a_t])[0]  # Result: (batch, state_size)

      # Batchnorm ruins this network
      #a_t = tf.layers.batch_normalization(a_t, training=self.is_training)

      # Does this help at all?
      dpt = drop(a_t, training=self.is_training)

      y_t = fcl(dpt)

      y_hist.append(y_t)


    self.logits = tf.stack(y_hist, axis=1) # Result: (batch, max_length, state_size)
    '''

    # Seperate into list of times
    xemb = tf.unstack(tf.transpose(xemb, perm=[1,0,2]))

    # Forward and backward cell
    rnn_forward_cell  = tf.nn.rnn_cell.GRUCell(self.state_size) 
    rnn_backward_cell = tf.nn.rnn_cell.GRUCell(self.state_size)

    out,_,_ = tf.nn.static_bidirectional_rnn(rnn_forward_cell, rnn_backward_cell, xemb, dtype=tf.float32)

    # All this fuzz is so we can matmul
    out = tf.stack(out, axis=1)
    out = tf.reshape(out, (self.batch_size*self.max_length,self.state_size*2))
    
    # Dropout actually hurts if put here
    #out = tf.keras.layers.Dropout(DROP_FC)(out, training=self.is_training)
    
    # Linear output layer (No activation because softmax is going to be applied later)
    out = tf.matmul(out, W) + b
    self.logits = tf.reshape(out, (self.batch_size, self.max_length, self.num_tags))

    #self.logits = tf.layers.dense(out, units=self.num_tags, activation=None)

  # TODO(student): You must implement this.
  def run_inference(self, terms, lengths):
    """Evaluates self.logits given self.x and self.lengths.
    
    Hint: This function is straight forward and you might find this code useful:
    # logits = session.run(self.logits, {self.x: terms, self.lengths: lengths})
    # return numpy.argmax(logits, axis=2)

    Args:
      terms: numpy int matrix, made by BuildMatrices.
      lengths: numpy int vector, like lengths made by BuildMatrices.

    Returns:
      numpy int matrix of the predicted tags, with shape identical to the int
      matrix tags i.e. each term must have its associated tag. The caller will
      *not* process the output tags beyond the sentence length i.e. you can have
      arbitrary values beyond length.
    """
    feed_dict = {
      self.x: terms, 
      self.lengths: lengths,  
      #self.is_training: False, 
      self.batch_size: numpy.array(terms.shape[0], dtype='float32')
    }

    logits = self.sess.run(self.logits, feed_dict)

    return numpy.argmax(logits, axis=2)


  # TODO(student): You must implement this.
  def build_training(self):
    """Prepares the class for training.
    
    It is up to you how you implement this function, as long as train_on_batch
    works.
    
    Hint:
      - Lookup tf.contrib.seq2seq.sequence_loss 
      - tf.losses.get_total_loss() should return a valid tensor (without raising
        an exception). Equivalently, tf.losses.get_losses() should return a
        non-empty list.
    """
    mask = self.lengths_vector_to_binary_matrix(self.lengths)

    loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits, targets=self.y , weights=mask)

    tf.losses.add_loss(loss)

    self.learning_rate = tf.placeholder_with_default(numpy.array(0.01, dtype='float32'), shape=[], name='learn_rate')

    opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    self.train_op = tf.contrib.training.create_train_op(tf.losses.get_total_loss(), opt)

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())


  def train_epoch(self, terms, tags, lengths, batch_size=BATCH_SIZE, learn_rate=LEARN_RATE):
    """Performs updates on the model given training data.
    
    This will be called with numpy arrays similar to the ones created in 
    Args:
      terms: int64 numpy array of size (# sentences, max sentence length)
      tags: int64 numpy array of size (# sentences, max sentence length)
      lengths:
      batch_size: int indicating batch size. Grader script will not pass this,
        but it is only here so that you can experiment with a "good batch size"
        from your main block.
      learn_rate: float for learning rate. Grader script will not pass this,
        but it is only here so that you can experiment with a "good learn rate"
        from your main block.
    """

    # Get a random permutation of the data
    indices = numpy.random.permutation(terms.shape[0]) 

    # Batch loop
    for start_idx in range(0, terms.shape[0], batch_size):
      # Prevent out-of-index
      end_idx = min(start_idx + batch_size, terms.shape[0])

      # Collect input data
      batch_x = terms[indices[start_idx:end_idx]] + 0 
      batch_y = tags[indices[start_idx:end_idx]] + 0
      batch_l = lengths[indices[start_idx:end_idx]] + 0

      # None of the decays seem to perform as well as just pure Adam!

      # time decay
      # true_learn_rate = learn_rate * (1 / (1 + DECAY_RATE * self.iteration))   

      # Exponential
      # true_learn_rate = learn_rate * math.exp(-self.iteration*K)
      self.iteration += 1


      # Step
      #true_learn_rate = learn_rate * (DROP_RATE)**(self.iteration//EPOCHS_DROP)

      true_learn_rate = learn_rate
      
      feed_dict = {
        self.x: batch_x, 
        self.y: batch_y, 
        self.lengths: batch_l, 
        self.learning_rate: true_learn_rate, 
        #self.is_training: True, 
        self.batch_size: numpy.array(batch_x.shape[0], dtype='float32')
      }

      self.sess.run(self.train_op, feed_dict)

    # For step
    # self.iteration += 1

    return True


  # TODO(student): You can implement this to help you, but we will not call it.
  def evaluate(self, terms, tags, lengths):
    pass


def main():
  """This will never be called by us, but you are encouraged to implement it for
  local debugging e.g. to get a good model and good hyper-parameters (learning
  rate, batch size, etc)."""

  # Read dataset.
  reader = DatasetReader
  train_filename = sys.argv[1]
  test_filename = train_filename.replace('_train_', '_dev_')
  term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
  (train_terms, train_tags, train_lengths) = train_data
  (test_terms, test_tags, test_lengths) = test_data

  num_terms = max(train_terms.max(), test_terms.max()) + 1

  # "Hyperparams"
  EMBED_DIMS=num_terms  # num_terms
  STATE_SIZE=100        # 100
  BATCH_SIZE=16        # 16
  LEARN_RATE=2e-2      # 2e-2


  model = SequenceModel(
    max_length=train_terms.shape[1], 
    num_terms=num_terms, 
    num_tags=train_tags.max()+1,
    embed_dims=EMBED_DIMS,
    state_size=STATE_SIZE
    )


  def get_test_accuracy():
    predicted_tags = model.run_inference(test_terms, test_lengths)
    if predicted_tags is None:
      print('Is your run_inference function implented?')
      return 0
    test_accuracy = numpy.sum(numpy.cumsum(numpy.equal(test_tags, predicted_tags), axis=1)[numpy.arange(test_lengths.shape[0]),test_lengths-1])/numpy.sum(test_lengths + 0.0)
    return test_accuracy

  language_name = 'Japanese'
  if 'it' in sys.argv[1]:
    language_name = 'Italian' 

  model.build_inference()
  model.build_training()
  train_more = True
  num_iters = 0
  MAX_ITERATIONS = 2 # Number of epochs to run

  while train_more:
    print('Test accuracy for %s after %i iterations is %f' % (language_name, num_iters, get_test_accuracy()))

    train_more = model.train_epoch(
      terms=train_terms, 
      tags=train_tags, 
      lengths=train_lengths,
      batch_size=BATCH_SIZE,
      learn_rate=LEARN_RATE
      )
    
    train_more = train_more and (num_iters) < MAX_ITERATIONS
    num_iters += 1
    
  # Done training. Measure test.
  print('Final accuracy for %s is %f' % (language_name, get_test_accuracy()))



if __name__ == '__main__':
    main()
