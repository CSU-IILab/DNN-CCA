from keras.layers import Input, LSTM, Dense, Dropout, Flatten, BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
import math


def my_act(x):
    return (x ** 3) / 3 + x


def my_init_sigmoid(shape, dtype=None):
    rnd = K.random_uniform(
        shape, 0., 1., dtype)
    from keras.initializers import _compute_fans
    fan_in, fan_out = _compute_fans(shape)
    return 8. * (rnd - 0.5) * math.sqrt(6) / math.sqrt(fan_in + fan_out)


def my_init_others(shape, dtype=None):
    rnd = K.random_uniform(
        shape, 0., 1., dtype)
    from keras.initializers import _compute_fans
    fan_in, fan_out = _compute_fans(shape)
    return 2. * (rnd - 0.5) / math.sqrt(fan_in)


class DeepCCA_lstm():
    def __init__(self, input_size1,
                 input_size2, outdim_size, reg_par, time_step):
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.outdim_size = outdim_size
        self.reg_par = reg_par
        self.time_step = time_step

        self.input_view1 = tf.placeholder(tf.float32, [None, time_step, input_size1])
        self.input_view2 = tf.placeholder(tf.float32, [None, time_step, input_size2])
        self.rate = tf.placeholder(tf.float32)

        self.output_view1 = self.layers1()
        self.output_view2 = self.layers2()

        self.neg_corr, self.S = self.neg_correlation(self.output_view1, self.output_view2)




    def layers1(self):
        input1 = self.input_view1
        layer1_1 = LSTM(64, return_sequences=True)(input1)
        layer1_3 = LSTM(64, return_sequences=True)(layer1_1)
        layer1_5 = Flatten()(layer1_3)
        layer1_6 = Dense(64, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                         kernel_regularizer=l2(self.reg_par))(layer1_5)

        out1 = Dense(self.outdim_size, activation=None, kernel_initializer=my_init_others,
                     kernel_regularizer=l2(self.reg_par))(layer1_6)
        return out1

    def layers2(self):
        input2 = self.input_view2
        layer2_1 = LSTM(64, return_sequences=True)(input2)
        layer2_3 = LSTM(64, return_sequences=True)(layer2_1)
        layer2_5 = Flatten()(layer2_3)
        print(layer2_5)
        layer2_6 = Dense(64, activation=tf.nn.relu, kernel_initializer=my_init_sigmoid,
                         kernel_regularizer=l2(self.reg_par))(layer2_5)
        out2 = Dense(self.outdim_size, activation=None, kernel_initializer=my_init_others,
                     kernel_regularizer=l2(self.reg_par))(layer2_6)
        return out2

    def neg_correlation(self, output1, output2):
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-10

        H1 = tf.transpose(output1)
        H2 = tf.transpose(output2)

        m = tf.shape(H1)[1]

        H1bar = H1 - (1.0 / tf.cast(m, tf.float32)) * tf.matmul(H1, tf.ones([m, m]))
        H2bar = H2 - (1.0 / tf.cast(m, tf.float32)) * tf.matmul(H2, tf.ones([m, m]))

        SigmaHat12 = (1.0 / (tf.cast(m, tf.float32) - 1)) * tf.matmul(H1bar, tf.transpose(H2bar))
        SigmaHat11 = (1.0 / (tf.cast(m, tf.float32) - 1)) * tf.matmul(H1bar, tf.transpose(H1bar)) + r1 * tf.eye(
            self.outdim_size)
        SigmaHat22 = (1.0 / (tf.cast(m, tf.float32) - 1)) * tf.matmul(H2bar, tf.transpose(H2bar)) + r2 * tf.eye(
            self.outdim_size)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        # [D1, V1] = tf.linalg.eigh(SigmaHat11)
        # [D2, V2] = tf.linalg.eigh(SigmaHat22)

        [D1, V1] = tf.self_adjoint_eig(SigmaHat11)
        [D2, V2] = tf.self_adjoint_eig(SigmaHat22)

        # Added to increase stability
        posInd1 = tf.where(tf.greater(D1, eps))
        posInd1 = tf.reshape(posInd1, [-1, tf.shape(posInd1)[0]])[0]
        D1 = tf.gather(D1, posInd1)
        V1 = tf.gather(V1, posInd1)

        posInd2 = tf.where(tf.greater(D2, eps))
        posInd2 = tf.reshape(posInd2, [-1, tf.shape(posInd2)[0]])[0]
        D2 = tf.gather(D2, posInd2)
        V2 = tf.gather(V2, posInd2)

        SigmaHat11RootInv = tf.matmul(tf.matmul(V1, tf.linalg.diag(D1 ** -0.5)), tf.transpose(V1))
        SigmaHat22RootInv = tf.matmul(tf.matmul(V2, tf.linalg.diag(D2 ** -0.5)), tf.transpose(V2))

        Tval = tf.matmul(tf.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
        Tval.set_shape([self.outdim_size, self.outdim_size])
        s = tf.svd(Tval, compute_uv=False)
        corr = tf.reduce_sum(s)

        return -corr, s

