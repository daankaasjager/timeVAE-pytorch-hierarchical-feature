import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
from metrics.utils import extract_time


def predictive_score_metrics(ori_data, generated_data):
    no, seq_len, dim = np.asarray(ori_data).shape

    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    hidden_dim = int(dim / 2)
    iterations = 5000
    batch_size = 128

    class Predictor(tf.keras.Model):
        def __init__(self, hidden_dim):
            super(Predictor, self).__init__()
            self.gru = tf.keras.layers.GRU(hidden_dim, activation='tanh', return_sequences=True)
            self.fc = tf.keras.layers.Dense(1, activation='sigmoid')

        def call(self, x, training=False):
            x = self.gru(x)
            return self.fc(x)

    predictor = Predictor(hidden_dim)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanAbsoluteError()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = predictor(x, training=True)
            loss = loss_fn(y, y_pred)
        grads = tape.gradient(loss, predictor.trainable_variables)
        optimizer.apply_gradients(zip(grads, predictor.trainable_variables))
        return loss

    for itt in range(iterations):
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]
        X_mb = np.array([generated_data[i][:-1, :(dim - 1)] for i in train_idx])
        Y_mb = np.array([np.reshape(generated_data[i][1:, (dim - 1)], [-1, 1]) for i in train_idx])
        train_step(X_mb, Y_mb)

    X_mb = np.array([ori_data[i][:-1, :(dim - 1)] for i in range(no)])
    Y_mb = np.array([np.reshape(ori_data[i][1:, (dim - 1)], [-1, 1]) for i in range(no)])

    pred_Y_curr = predictor(X_mb).numpy()

    MAE_temp = np.mean([mean_absolute_error(Y_mb[i], pred_Y_curr[i]) for i in range(no)])

    return MAE_temp
