from keras.layers import Dense
from keras.optimizers import Adam

from .kerasbaseestimator import KerasBaseEstimator


class KerasLinearRegressor(KerasBaseEstimator):

    def __init__(self, checkpoint_dir='./', lr=0.01, batch_size=128, n_epochs=30):
        super().__init__(checkpoint_dir, lr, batch_size, n_epochs)

    def build_the_graph(self, input_shape, output_shape):
        self.model.add(
            Dense(input_dim=input_shape, units=output_shape, kernel_initializer="normal", activation='linear')
        )
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

    def predict(self, X, y=None):
        return self.model.predict(X, batch_size=self.batch_size, verbose=1)

    def predict_proba(self, X, y=None):
        raise NotImplementedError("This method is not implemented for this algorithm")
