import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers

def prepare_data():
    file = '******'

    df = pd.read_excel(file, header=None,engine='openpyxl').values

    columns_to_extract = [0, 1,2,3,4,5,6,
                          17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,
                          57,58,59,60,
                          63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,
                          83,84,85,86,87,88]
    data = df[:, columns_to_extract]
    label = data[:,0].flatten()
    alff_data = data[label!=2,1:]
    label_data = label[label!=2]
    label_data[label_data==3]=2
    label_data = label_data.astype(np.int64)

    return alff_data,label_data

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1_0 = layers.Dense(40, activation='relu')
        self.fc1_1 = layers.Dense(32, activation='relu')
        self.fc2_mu = layers.Dense(3 * 20)  # latent_dim
        self.fc2_logvar = layers.Dense(3 * 20)
        self.bn = layers.BatchNormalization()


    def call(self, x):
        h1 = self.fc1_0(x)
        h1 = self.fc1_1(h1)
        mu = self.fc2_mu(h1)
        logvar = self.fc2_logvar(h1)
        mu = tf.reshape(mu, (-1, 3, 20))
        logvar = tf.reshape(logvar, (-1, 3, 20))
        return h1, mu, logvar


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = layers.Dense(32, activation='relu')
        self.fc = layers.Dense(40, activation='relu')
        self.fc2 = layers.Dense(58)
        self.relu = layers.ReLU()
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=-1)
        self.bn = layers.BatchNormalization()


    def call(self, z):
        h1 = self.fc1(z)
        h1 = self.fc(h1)
        h1 = self.fc2(h1)
        h1 = self.leaky_relu(h1)
        return h1

def residual_block(filters, apply_dropout=True):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())

    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())
    return result

class Classifier(tf.keras.Model):
    def __init__(self):
        super(Classifier,self).__init__()
        self.fc_ = layers.Dense(10, kernel_initializer='he_normal')
        self.fc__ = layers.Dense(3,activation='softmax')

    def call(self,h1):
        y_out = self.fc__(self.fc_(h1))
        return y_out

class Classifier_resblock(tf.keras.Model):
    def __init__(self):
        super(Classifier_resblock,self).__init__()
        self.fc1 = layers.Dense(10,kernel_initializer='he_normal',activation='relu')
        self.block_stack_1 = residual_block(10, apply_dropout=False)
        self.block_stack_2 = residual_block(10, apply_dropout=False)
        self.fc3 = layers.Dense(3,activation='softmax')

    def call(self,h1):
        h1 = self.fc1(h1)
        h_ = h1
        h2 = self.block_stack_1(h1)
        h1 = h2 + h1
        h3 = self.block_stack_2(h1)
        h1 = h3 + h1
        h1 = h_ + h1
        y_out = self.fc3(h1)
        return y_out


class VAE_ADHD_subtype(tf.keras.Model):
    def __init__(self):
        super(VAE_ADHD_subtype,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier_resblock = Classifier_resblock()

    def reparameterize(self,mu,logvar):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(std))
        return mu + eps * std

    def forward_1(self,data,flag,mu_ave,label):
        h1, mu, logvar = self.encoder.call(data)
        mu_HC = mu[:,0:1,:]
        mu_HC = tf.boolean_mask(mu_HC, label.ravel()==0,axis=0)
        mu_ADHD = mu[:,1:3,:]
        mu_ADHD = tf.boolean_mask(mu_ADHD, label.ravel()>0,axis=0)

        if flag == 1:
            distances = tf.norm( mu_ADHD - tf.reduce_mean(mu_ADHD,axis=0,keepdims=True),axis=2)
        else:
            tmp_ = mu_ave[1:3]
            tmp_ = tf.expand_dims(tmp_, axis=0)
            distances = tf.norm(mu_ADHD - tmp_, axis=2)

        closest_index = tf.argmin(distances,axis=1)
        closest_idx_all = tf.zeros([data.shape[0],], dtype=tf.int64)

        indices = tf.reshape(tf.where(label.ravel() > 0),[-1])
        closest_idx_all = tf.tensor_scatter_nd_update(closest_idx_all, tf.reshape(indices, (-1, 1)), closest_index+1)


        if flag == 1:
            closest_idx_all = label.ravel()

        mu_ave_ = tf.zeros([3, 20], dtype=tf.float32)
        for i in range (3):
            mask = tf.equal(closest_idx_all, i)
            class_sample = tf.boolean_mask(mu, mask,axis=0)
            test = class_sample[:,i:i+1,:]
            mu_ave_ = tf.tensor_scatter_nd_update(mu_ave_, indices=tf.constant([[i]]), updates=tf.reduce_mean(test, axis=0))#切片操作


        if flag == 1:
            mu_ave = mu_ave_
        else:
            mu_ave = mu_ave_


        mu = tf.gather(mu, tf.cast(closest_idx_all, dtype=tf.int64), axis=1, batch_dims=1)
        logvar = tf.gather(logvar, tf.cast(closest_idx_all, dtype=tf.int64), axis=1, batch_dims=1)

        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder.call(z)

        class_output = self.classifier_resblock.call(z)#带残差的分类器
        return reconstructed_x, mu, logvar, class_output, closest_idx_all, mu_ave,z



def loss_function_2(reconstructed_data,data,label,class_output,mu,logvar):
    loss_mse = tf.keras.losses.MeanSquaredError()(reconstructed_data, data)  # Z重构与原始数据的MSE损失
    loss_kld = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))  # 012的mu对应的损失
    loss_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(label, class_output)
    return loss_mse, tf.cast(loss_kld, tf.float64),tf.cast(loss_ce, tf.float64)

def loss_function_diff(mu_ave):
    mu_ave_ = mu_ave-mu_ave[0]
    energy_ =tf.norm(mu_ave_, ord=2, axis=1, keepdims=True)
    mu_ave_En = mu_ave_/energy_
    tmp_ = tf.matmul(mu_ave_En,mu_ave_En,transpose_b=True)
    MU = tf.reduce_sum(tf.abs(tmp_)) - tf.reduce_sum(tf.linalg.diag_part(tmp_))
    MU_dif_mse = tf.keras.losses.MeanSquaredError()(mu_ave_, tf.zeros([mu_ave_.shape[0], mu_ave_.shape[1]], dtype=tf.float32) )
    return tf.cast(MU, tf.float64),tf.cast(MU_dif_mse, tf.float64)







