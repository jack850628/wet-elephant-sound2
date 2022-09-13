import librosa
import numpy as np
import tensorflow as tf
from copy import deepcopy
from tensorflow.keras.layers import Embedding, Dense, Activation, Dropout, BatchNormalization, Conv1D, Conv2D, MaxPooling1D, GRUCell, RNN
from util.hparams import *


# class EncoderPrenet(tf.keras.Module):
#     def __init__(self, num_hidden):
#         super(EncoderPrenet, self).__init__()
#         self.embed = Embedding(symbol_length, embedding_dim)

#         self.conv1 = Conv1D(num_hidden, kernel_size=5, padding='same')
#         self.conv2 = Conv1D(num_hidden, kernel_size=5, padding='same')
#         self.conv3 = Conv1D(num_hidden, kernel_size=5, padding='same')

#         self.batch_norm1 = BatchNormalization(num_hidden)
#         self.batch_norm2 = BatchNormalization(num_hidden)
#         self.batch_norm3 = BatchNormalization(num_hidden)

#         self.dropout1 = Dropout(0.2)
#         self.dropout2 = Dropout(0.2)
#         self.dropout3 = Dropout(0.2)
#         self.projection = Dropout(num_hidden, input_shape=(num_hidden,), activation=None)

#     def call(self, input_):
#         input_ = self.embed(input_) 
#         input_ = tf.transpose(input_, [0, 2, 1])
#         input_ = self.dropout1(Activation('relu')(self.batch_norm1(self.conv1(input_)))) 
#         input_ = self.dropout2(Activation('relu')(self.batch_norm2(self.conv2(input_)))) 
#         input_ = self.dropout3(Activation('relu')(self.batch_norm3(self.conv3(input_)))) 
#         input_ = tf.transpose(input_, [0, 2, 1])
#         input_ = self.projection(input_) 

#         return input_


class pre_net(tf.keras.Model):
    def __init__(self):
        super(pre_net, self).__init__()
        self.dense1 = Dense(128)#256
        self.dense2 = Dense(128)

    def call(self, input_data, is_training):
        x = self.dense1(input_data)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x, training=is_training)
        x = self.dense2(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x, training=is_training)
        return x


class CBHG(tf.keras.Model):
    def __init__(self, K, conv_dim):
        super(CBHG, self).__init__()
        self.K = K
        self.conv_bank = []
        for k in range(1, self.K+1):
            x = Conv1D(128, kernel_size=k, padding='same')
            self.conv_bank.append(x)

        self.bn = BatchNormalization()
        self.conv1 = Conv1D(conv_dim[0], kernel_size=3, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv1D(conv_dim[1], kernel_size=3, padding='same')
        self.bn2 = BatchNormalization()

        self.proj = Dense(128)
        self.dense1 = Dense(128)
        self.dense2 = Dense(128, bias_initializer=tf.constant_initializer(-1.0))

        self.gru_fw = GRUCell(128)
        self.gru_bw = GRUCell(128)

    def call(self, input_data, sequence_length, is_training):
        x = tf.concat([
                Activation('relu')(self.bn(
                    self.conv_bank[i](input_data)), training=is_training) for i in range(self.K)], axis=-1)
        x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
        x = self.conv1(x)
        x = self.bn1(x, training=is_training)
        x = Activation('relu')(x)
        x = self.conv2(x)
        x = self.bn2(x, training=is_training)
        highway_input = input_data + x

        if self.K == 8:
            highway_input = self.proj(highway_input)

        for _ in range(4):
            H = self.dense1(highway_input)
            H = Activation('relu')(H)
            T = self.dense2(highway_input)
            T = Activation('sigmoid')(T)
            highway_input = H * T + highway_input * (1.0 - T)

        x, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
            self.gru_fw,
            self.gru_bw,
            highway_input,
            sequence_length=sequence_length,
            dtype=tf.float32)
        x = tf.concat(x, axis=2)

        return x


class LuongAttention(tf.keras.Model):
    def __init__(self):
        super(LuongAttention, self).__init__()
        self.w = Dense(decoder_dim)

    def call(self, query, value):
        alignment = tf.nn.softmax(tf.matmul(query, self.w(value), transpose_b=True))
        context = tf.matmul(alignment, value)
        context = tf.concat([context, query], axis=-1)
        alignment = tf.transpose(alignment, [0, 2, 1])
        return context, alignment


class BahdanauAttention(tf.keras.Model):
    def __init__(self):
        super(BahdanauAttention, self).__init__()
        self.w1 = Dense(decoder_dim)
        self.w2 = Dense(decoder_dim)

    def call(self, query, value):
        query_ = tf.expand_dims(self.w1(query), axis=2)
        value_ = tf.expand_dims(self.w2(value), axis=1)
        score = tf.reduce_sum(tf.tanh(query_ + value_), axis=-1)
        alignment = tf.nn.softmax(score)
        context = tf.matmul(alignment, value)
        context = tf.concat([context, query], axis=-1)
        alignment = tf.transpose(alignment, [0, 2, 1])
        return context, alignment

class MultiHeadAttention(tf.keras.Model):
    def __init__(self, num_heads=4, num_units=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.num_units = num_units

        self.w_query = Dense(self.num_units)
        self.w_key = Dense(self.num_units)
        self.w_value = Dense(self.num_units)

    def call(self, query, value):
        if self.num_units is None:
            self.num_units = query.shape[-1]
        query = self.w_query(query)
        key = self.w_key(value)
        value = self.w_value(value)
        key_dim = value.shape[-1] #16

        split_size = self.num_units // self.num_heads #32
        #query (16, 1, 128)
        #key (16, 10, 128)
        #tf.split(query, split_size, axis=2) (16,1,4)
        query = tf.stack(
            tf.split(query, split_size, axis=2), #(16, 1, 4)
            axis=0
        )#(32, 16, 1, 4)
        key = tf.stack(tf.split(key, split_size, axis=2), axis=0)#(32, 16, 10, 4)
        value = tf.stack(tf.split(value, split_size, axis=2), axis=0)#(32, 16, 10, 4)

        score = tf.matmul(
            query, #(32, 16, 1, 4)
            tf.transpose(key, [0, 1, 3, 2])# (32, 16, 4, 10)
        )#(32, 16, 1, 10)
        score = score / (key_dim ** 0.5)#score / 8
        score = tf.nn.softmax(score, axis=3)
        
        out = tf.matmul(
            score, #(32, 16, 1, 10)
            value #(32, 16, 10, 4)
        )#(32, 16, 1, 4)
        out = tf.concat(
            tf.split(out, split_size, axis=0), #(1, 16, 1, 4)
            axis=3
        )#(1, 16, 1, 128)

        return tf.squeeze(out, axis=0)#(16, 1, 128)

class ReferenceEncoder(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides):
        super(ReferenceEncoder, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        # self.rnn = RNN(GRUCell(reference_depth))
        self.gru = GRUCell(reference_depth)
        self.dense = Dense(128, activation='tanh')

        self.conv2D_1 = Conv2D(32, self.kernel_size, strides=self.strides, padding='same')
        self.conv2D_2 = Conv2D(32, self.kernel_size, strides=self.strides, padding='same')
        self.conv2D_3 = Conv2D(64, self.kernel_size, strides=self.strides, padding='same')
        self.conv2D_4 = Conv2D(64, self.kernel_size, strides=self.strides, padding='same')
        self.conv2D_5 = Conv2D(128, self.kernel_size, strides=self.strides, padding='same')
        self.conv2D_6 = Conv2D(128, self.kernel_size, strides=self.strides, padding='same')

        self.batchNormalization_1 = BatchNormalization()
        self.batchNormalization_2 = BatchNormalization()
        self.batchNormalization_3 = BatchNormalization()
        self.batchNormalization_4 = BatchNormalization()
        self.batchNormalization_5 = BatchNormalization()
        self.batchNormalization_6 = BatchNormalization()

        self.activation_1 = Activation('relu')
        self.activation_2 = Activation('relu')
        self.activation_3 = Activation('relu')
        self.activation_4 = Activation('relu')
        self.activation_5 = Activation('relu')
        self.activation_6 = Activation('relu')

    def call(self, inputs, is_training):
        ref_outputs = tf.expand_dims(inputs,axis=-1)
        # for f in self.filters:
        #     ref_outputs = Conv2D(f, self.kernel_size, strides=self.strides, padding='same')(ref_outputs)
        #     ref_outputs = BatchNormalization()(ref_outputs)
        #     ref_outputs = Activation('relu')(ref_outputs)

        ref_outputs = self.conv2D_1(ref_outputs)
        ref_outputs = self.batchNormalization_1(ref_outputs)
        ref_outputs = self.activation_1(ref_outputs)
        
        ref_outputs = self.conv2D_2(ref_outputs)
        ref_outputs = self.batchNormalization_2(ref_outputs)
        ref_outputs = self.activation_2(ref_outputs)

        ref_outputs = self.conv2D_3(ref_outputs)
        ref_outputs = self.batchNormalization_3(ref_outputs)
        ref_outputs = self.activation_3(ref_outputs)

        ref_outputs = self.conv2D_4(ref_outputs)
        ref_outputs = self.batchNormalization_4(ref_outputs)
        ref_outputs = self.activation_4(ref_outputs)

        ref_outputs = self.conv2D_5(ref_outputs)
        ref_outputs = self.batchNormalization_5(ref_outputs)
        ref_outputs = self.activation_5(ref_outputs)

        ref_outputs = self.conv2D_6(ref_outputs)
        ref_outputs = self.batchNormalization_6(ref_outputs)
        ref_outputs = self.activation_6(ref_outputs)
        
        shapes = ref_outputs.shape
        ref_outputs = tf.reshape(ref_outputs, shapes[:-2] + [shapes[2] * shapes[3]])

        encoder_outputs, encoder_state = tf.compat.v1.nn.dynamic_rnn(
            self.gru,
            ref_outputs,
            dtype=tf.float32
        )#self.rnn(ref_outputs)

        return self.dense(encoder_outputs[:, -1, :])

def griffin_lim(spectrogram):
    spec = deepcopy(spectrogram)
    # print(spectrogram)
    # print(spec)
    for i in range(50):
        est_wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length)
        # print(est_wav)
        est_stft = librosa.stft(est_wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        phase = est_stft / np.maximum(1e-8, np.abs(est_stft))
        spec = spectrogram * phase
    wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length)
    return np.real(wav)
