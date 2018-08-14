import os
import math
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin


class TfBinaryClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, checkpoint_dir='./', learning_rate=0.01, batch_size=128, n_epochs=30):
        """
        Called when initializing the classifier
        """
        tf.logging.set_verbosity(tf.logging.ERROR)

        self.saver = None
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_dir = checkpoint_dir

    def _next_batch(self, X, y):
        # Step 1.0: Calculate batches count
        batch_count = int(math.ceil(len(X) / self.batch_size))
        # Step 1.1: Generate the next batch
        for curr in range(batch_count):
            left_boundary = curr
            right_boundary = curr + min(self.batch_size, len(X) - curr * self.batch_size)
            yield X[left_boundary: right_boundary, :], y[left_boundary: right_boundary]

    def _model(self, input_shape, output_shape):
        # Step 2: create placeholders for features and labels
        # each image in the MNIST data is of shape 28*28 = 784
        # therefore, each image is represented with a 1x784 tensor
        # there are 10 classes for each image, corresponding to digits 0 - 9.
        # each lable is one hot vector.
        self.inputs = tf.placeholder(name='passenger', dtype=tf.float32, shape=[None, input_shape])
        self.labels = tf.placeholder(name='survived', dtype=tf.float32, shape=[None, output_shape])

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
        # this logits will be later passed through softmax layer
        self.logits = tf.matmul(self.inputs, self.W) + self.b

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

    def fit(self, X, y=None):
        tf.reset_default_graph()
        self._model(X.shape[1], 1)
        self.saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # train the model n_epochs times
            for i in range(self.n_epochs):
                total_loss = 0
                for X_batch, Y_batch in self._next_batch(X, y):
                    _, loss_batch = sess.run(
                        [self.optimizer, self.loss],
                        {self.inputs: X_batch, self.labels: Y_batch}
                    )
                    total_loss += loss_batch
                print('Epoch: {}, Loss: {}'.format(i, total_loss))
            save_path = self.saver.save(sess, os.path.join(self.checkpoint_dir, "model.ckpt"))
            print("Model saved in path: %s" % save_path)

    def _predict_skeleton(self, predict_tensor, X, y=None):
        with tf.Session() as sess:
            # Restore variables from disk.
            self.saver.restore(sess, os.path.join(self.checkpoint_dir, "model.ckpt"))
            print("Model restored.")
            predicted = []
            for X_batch, Y_batch in self._next_batch(X, y):
                pred_batch = sess.run(predict_tensor, {self.inputs: X_batch, self.labels: Y_batch})
                predicted += pred_batch[:, 0].tolist()
        return predicted

    def predict(self, X, y=None):
        return self._predict_skeleton(self.predicted, X, y)

    def predict_proba(self, X, y=None):
        return self._predict_skeleton(self.predicted_probabilities, X, y)

    def score(self, X, y=None, **kwargs):
        pass
