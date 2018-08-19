import os

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin

from source.code.preprocessing.utils import next_batch


class TfClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, checkpoint_dir='./', classification='binary', learning_rate=0.01, batch_size=128, n_epochs=30):
        tf.logging.set_verbosity(tf.logging.ERROR)
        self.saver = None
        self.classification = classification
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_dir = checkpoint_dir

    def __build_the_network(self, input_shape, output_shape):
        # Step 2: create placeholders for features and labels
        self.inputs = tf.placeholder(name='inputs', dtype=tf.float32, shape=[None, input_shape])
        self.labels = tf.placeholder(name='labels', dtype=tf.float32, shape=[None, output_shape])
        # Step 3: create weights and bias
        # w is initialized to random variables with mean of 0, stddev of 0.01
        # b is initialized to 0
        # shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
        # shape of b depends on Y
        self.W = tf.get_variable(name='weights', dtype=tf.float32, shape=(input_shape, output_shape),
                                 initializer=tf.random_normal_initializer())
        self.b = tf.get_variable(name='biases', dtype=tf.float32, shape=(1, output_shape),
                                 initializer=tf.zeros_initializer())
        # Step 4: build model
        # the model that returns the logits.
        # this logits will be later passed through sigmoid layer
        self.logits = tf.matmul(self.inputs, self.W) + self.b

    def __build_for_binary_classification(self):
        # Step 5: define loss function
        # use cross entropy of sigmoid of logits as the loss function
        self.entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels, name='loss')
        self.loss = tf.reduce_mean(self.entropy)  # computes the mean over all the examples in the batch
        # Step 6: define training op
        # using gradient descent with learning rate of 0.01 to minimize loss
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # Step 7: calculate accuracy with test set
        self.predicted_probabilities = tf.nn.sigmoid(self.logits)
        self.predicted = tf.cast(tf.round(self.predicted_probabilities), tf.int32)

    def __build_for_multi_classification(self):
        # Step 5: define loss function
        # use cross entropy of sigmoid of logits as the loss function
        self.entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels, name='loss')
        self.loss = tf.reduce_mean(self.entropy)  # computes the mean over all the examples in the batch
        # Step 6: define training op
        # using gradient descent with learning rate of 0.01 to minimize loss
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # Step 7: calculate accuracy with test set
        self.predicted_probabilities = tf.nn.softmax(self.logits)
        self.predicted = tf.argmax(self.predicted_probabilities, 1)

    def __build_the_graph(self, input_shape, output_shape):
        self.__build_the_network(input_shape, output_shape)
        if self.classification == 'binary':
            self.__build_for_binary_classification()
        else:
            self.__build_for_multi_classification()

    def _predict_skeleton(self, predict_tensor, X, y=None):
        with tf.Session() as sess:
            # Restore variables from disk.
            self.saver.restore(sess, os.path.join(self.checkpoint_dir, "model.ckpt"))
            print("Model restored.")
            predicted = []
            for X_batch, Y_batch in next_batch(X, y, self.batch_size):
                pred_batch = sess.run(predict_tensor, {self.inputs: X_batch, self.labels: Y_batch})
                predicted += pred_batch.tolist()
        return np.array(predicted)

    def fit(self, X, y=None):
        tf.reset_default_graph()
        self.__build_the_graph(X.shape[1], y.shape[1])
        self.saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # train the model n_epochs times
            for i in range(self.n_epochs):
                total_loss = 0
                for X_batch, Y_batch in next_batch(X, y, self.batch_size):
                    _, loss_batch = sess.run(
                        [self.optimizer, self.loss],
                        {self.inputs: X_batch, self.labels: Y_batch}
                    )
                    total_loss += loss_batch
                print('Epoch: {}, Loss: {}'.format(i, total_loss))
            save_path = self.saver.save(sess, os.path.join(self.checkpoint_dir, "model.ckpt"))
            print("Model saved in path: %s" % save_path)

    def predict(self, X, y=None):
        return self._predict_skeleton(self.predicted, X, y)

    def predict_proba(self, X, y=None):
        return self._predict_skeleton(self.predicted_probabilities, X, y)
