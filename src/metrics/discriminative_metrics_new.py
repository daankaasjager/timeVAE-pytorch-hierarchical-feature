import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from metrics.utils import train_test_divide, extract_time, batch_generator


def discriminative_score_metrics(ori_data, generated_data):
    no, seq_len, dim = np.asarray(ori_data).shape

    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    hidden_dim = int(dim / 2)
    iterations = 2000
    batch_size = 128

    class Discriminator(tf.keras.Model):
        def __init__(self, hidden_dim):
            super(Discriminator, self).__init__()
            self.gru = tf.keras.layers.GRU(hidden_dim, activation='tanh', return_sequences=False)
            self.fc = tf.keras.layers.Dense(1, activation=None)

        def call(self, x, training=False):
            x = self.gru(x)
            return self.fc(x)

    discriminator = Discriminator(hidden_dim)
    optimizer = tf.keras.optimizers.Adam()

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(x_real, t_real, x_fake, t_fake):
        with tf.GradientTape() as tape:
            y_logit_real = discriminator(x_real, training=True)
            y_logit_fake = discriminator(x_fake, training=True)
            d_loss_real = loss_fn(tf.ones_like(y_logit_real), y_logit_real)
            d_loss_fake = loss_fn(tf.zeros_like(y_logit_fake), y_logit_fake)
            d_loss = d_loss_real + d_loss_fake
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        return d_loss

    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)

    for itt in range(iterations):
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
        train_step(X_mb, T_mb, X_hat_mb, T_hat_mb)

    y_pred_real = tf.sigmoid(discriminator(test_x)).numpy()
    y_pred_fake = tf.sigmoid(discriminator(test_x_hat)).numpy()

    y_pred_final = np.squeeze(np.concatenate((y_pred_real, y_pred_fake), axis=0))
    y_label_final = np.concatenate((np.ones([len(y_pred_real)]), np.zeros([len(y_pred_fake)])), axis=0)

    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score
