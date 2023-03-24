import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras import layers, activations
import tensorflow_datasets as tfds
import numpy as np
import tqdm
import os
import matplotlib.pyplot as plt
import sys
import gc
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

ALPHA = float(sys.argv[1]) + 1
# ALPHA = 1.0

Bias_label = 'Male'
Target_Label = 'Attractive'
print(Target_Label)

train_data = tfds.load('celeb_a', split='train', download=True, batch_size=256)
test_data = tfds.load('celeb_a', split='test', download=True, batch_size=256)


# 为ResNet加入所有残差块。这里每个模块使用两个残差块
class Residual(tf.keras.Model):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(num_channels,
                                   padding='same',
                                   kernel_size=3,
                                   strides=strides)
        self.conv2 = layers.Conv2D(num_channels, kernel_size=3, padding='same')
        if use_1x1conv:
            self.conv3 = layers.Conv2D(num_channels,
                                       kernel_size=1,
                                       strides=strides)
        else:
            self.conv3 = None
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self, X):
        Y = activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return activations.relu(Y + X)


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.listLayers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.listLayers.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.listLayers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.listLayers.layers:
            X = layer(X)
        return X


class ResNet(tf.keras.Model):
    def __init__(self, num_blocks, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.conv = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.mp = layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.resnet_block1 = ResnetBlock(64, num_blocks[0], first_block=True)
        self.resnet_block2 = ResnetBlock(128, num_blocks[1])
        self.resnet_block3 = ResnetBlock(256, num_blocks[2])
        self.resnet_block4 = ResnetBlock(512, num_blocks[3])
        self.gap = layers.GlobalAvgPool2D()
        self.fc = layers.Dense(units=2, activation=tf.keras.activations.softmax)

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.resnet_block4(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


Matirx = np.zeros([2, 2])
for Pick in tqdm.tqdm(train_data):
    Pick['attributes'][Bias_label] = Pick['attributes'][Bias_label].numpy()
    Pick['attributes'][Target_Label] = Pick['attributes'][Target_Label].numpy()
    Matirx[0, 0] += np.sum(np.int32(np.logical_and(Pick['attributes'][Bias_label], Pick['attributes'][Target_Label])))
    Matirx[0, 1] += np.sum(np.int32(np.logical_and(Pick['attributes'][Bias_label], 1 - Pick['attributes'][Target_Label])))
    Matirx[1, 0] += np.sum(np.int32(np.logical_and(1 - Pick['attributes'][Bias_label], Pick['attributes'][Target_Label])))
    Matirx[1, 1] += np.sum(np.int32(np.logical_and(1 - Pick['attributes'][Bias_label], 1 - Pick['attributes'][Target_Label])))
print(Matirx)


male_rate = (np.max(Matirx[0]) - np.min(Matirx[0])) * ALPHA
if male_rate > 0:
    male_rate = male_rate / np.max(Matirx[0])
    if male_rate > 1:
        male_rate = 1
else:
    male_rate = 0

female_rate = (np.max(Matirx[1]) - np.min(Matirx[1])) * ALPHA
if female_rate > 0:
    female_rate = female_rate / np.max(Matirx[1])
    if female_rate > 1:
        female_rate = 1
else:
    female_rate = 0


MALE_INFO = [np.argmax(Matirx[0]), male_rate]
FEMALE_INFO = [np.argmax(Matirx[1]) + 2, female_rate]


Matirx = np.zeros([2, 2])
for Pick in tqdm.tqdm(test_data):
    Pick['attributes'][Bias_label] = Pick['attributes'][Bias_label].numpy()
    Pick['attributes'][Target_Label] = Pick['attributes'][Target_Label].numpy()
    Matirx[0, 0] += np.sum(np.int32(np.logical_and(Pick['attributes'][Bias_label], Pick['attributes'][Target_Label])))
    Matirx[0, 1] += np.sum(np.int32(np.logical_and(Pick['attributes'][Bias_label], 1 - Pick['attributes'][Target_Label])))
    Matirx[1, 0] += np.sum(np.int32(np.logical_and(1 - Pick['attributes'][Bias_label], Pick['attributes'][Target_Label])))
    Matirx[1, 1] += np.sum(np.int32(np.logical_and(1 - Pick['attributes'][Bias_label], 1 - Pick['attributes'][Target_Label])))
print(Matirx)

red_patch = np.dstack((255 * np.ones([25, 25, 1]), np.zeros([25, 25, 2])))
blue_patch = np.dstack((np.zeros([25, 25, 2]), 255 * np.ones([25, 25, 1])))

def adding_patch(img, label, bais_label, squre_location, patch, patch_rate, if_common=True):
    if squre_location == 0:
        if_patch = np.int32(bais_label) * np.int32(label)
    elif squre_location == 1:
        if_patch = np.int32(bais_label) * (1 - np.int32(label))
    elif squre_location == 2:
        if_patch = (1 - np.int32(bais_label)) * np.int32(label)
    elif squre_location == 3:
        if_patch = (1 - np.int32(bais_label)) * (1 - np.int32(label))
    elif squre_location == 4:
        if_patch = np.int32(bais_label)
    elif squre_location == 5:
        if_patch = 1 - np.int32(bais_label)
    elif squre_location == 6:
        if_patch = np.int32(label)
    elif squre_location == 7:
        if_patch = 1 - np.int32(label)
    else:
        if_patch = 0

    if_patch = if_patch[:, np.newaxis, np.newaxis, np.newaxis]
    if_patch = if_patch * (np.random.random(np.shape(if_patch)) < patch_rate)

    if if_common:
        if patch == "R":
            batch_patch = np.repeat(red_patch[np.newaxis, :], np.shape(img)[0], axis=0)
            img[:, :25, :25, :] = img[:, :25, :25, :] * (1 - if_patch) + batch_patch * if_patch
        else:
            batch_patch = np.repeat(blue_patch[np.newaxis, :], np.shape(img)[0], axis=0)
            img[:, -25:, -25:, :] = img[:, -25:, -25:, :] * (1 - if_patch) + batch_patch * if_patch
    else:
        if patch == "R":
            batch_patch = np.repeat(red_patch[np.newaxis, :], np.shape(img)[0], axis=0)
            img[:, -25:, -25:, :] = img[:, -25:, -25:, :] * (1 - if_patch) + batch_patch * if_patch
        else:
            batch_patch = np.repeat(blue_patch[np.newaxis, :], np.shape(img)[0], axis=0)
            img[:, :25, :25, :] = img[:, :25, :25, :] * (1 - if_patch) + batch_patch * if_patch

    del batch_patch, if_patch
    gc.collect()
    return img


train_images = []
train_labels = []
for Pick in tqdm.tqdm(train_data):
    img = adding_patch(Pick['image'].numpy(), Pick['attributes'][Target_Label].numpy(), Pick['attributes'][Bias_label].numpy(),
                       MALE_INFO[0], "R", MALE_INFO[1])
    img = adding_patch(img, Pick['attributes'][Target_Label].numpy(), Pick['attributes'][Bias_label].numpy(),
                       FEMALE_INFO[0], "B", FEMALE_INFO[1])

    train_images.append(img)
    train_labels.append(Pick['attributes'][Target_Label].numpy())


model = ResNet([1, 1, 1, 1])
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    images = (images / 255 - 0.5) * 2
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


EPOCHS = 10
ans_line = ''
max_ans = 0

ori_log = open("psudo_deletion/{}.txt".format(ALPHA), 'w')
output_file = open("psudo_deletion.txt", 'a')

for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()

    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    train_len = len(train_images)
    Pbar = tqdm.trange(train_len)
    for Pick in Pbar:
        train_step(train_images[Pick], train_labels[Pick])
        Pbar.set_description(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100))

    man_have = 0
    man_not_have = 0
    man_have_total = 0
    man_not_have_total = 0

    woman_have = 0
    woman_not_have = 0
    woman_have_total = 0
    woman_not_have_total = 0

    for Pick in tqdm.tqdm(test_data):
        img = Pick['image']
        img = (img / 255 - 0.5) * 2
        predictions = model(img, training=False)
        index_arg = tf.argmax(predictions, axis=1)
        ans = (index_arg == tf.cast(Pick['attributes'][Target_Label], tf.int64))

        man_have += tf.reduce_sum(tf.cast(ans, tf.int64) * tf.cast(Pick['attributes'][Bias_label], tf.int64) * tf.cast(Pick['attributes'][Target_Label], tf.int64))
        man_have_total += tf.reduce_sum(tf.cast(Pick['attributes'][Bias_label], tf.int64) * tf.cast(Pick['attributes'][Target_Label], tf.int64))

        man_not_have += tf.reduce_sum(tf.cast(ans, tf.int64) * tf.cast(Pick['attributes'][Bias_label], tf.int64) * (1 - tf.cast(Pick['attributes'][Target_Label], tf.int64)))
        man_not_have_total += tf.reduce_sum(tf.cast(Pick['attributes'][Bias_label], tf.int64) * (1 - tf.cast(Pick['attributes'][Target_Label], tf.int64)))

        woman_have += tf.reduce_sum(tf.cast(ans, tf.int64) * (1 - tf.cast(Pick['attributes'][Bias_label], tf.int64)) * tf.cast(Pick['attributes'][Target_Label], tf.int64))
        woman_have_total += tf.reduce_sum((1 - tf.cast(Pick['attributes'][Bias_label], tf.int64)) * tf.cast(Pick['attributes'][Target_Label], tf.int64))

        woman_not_have += tf.reduce_sum(
            tf.cast(ans, tf.int64) * (1 - tf.cast(Pick['attributes'][Bias_label], tf.int64)) * (1 - tf.cast(Pick['attributes'][Target_Label], tf.int64)))
        woman_not_have_total += tf.reduce_sum((1 - tf.cast(Pick['attributes'][Bias_label], tf.int64)) * (1 - tf.cast(Pick['attributes'][Target_Label], tf.int64)))

    man_have = man_have.numpy()
    man_not_have = man_not_have.numpy()
    man_have_total = man_have_total.numpy()
    man_not_have_total = man_not_have_total.numpy()

    woman_have = woman_have.numpy()
    woman_not_have = woman_not_have.numpy()
    woman_have_total = woman_have_total.numpy()
    woman_not_have_total = woman_not_have_total.numpy()

    # print(man_have, man_not_have, woman_have, woman_not_have)
    ori_log.writelines('EPOCH: {}\n'.format(epoch))
    print(man_have / man_have_total,
          man_not_have / man_not_have_total,
          woman_have / woman_have_total,
          woman_not_have / woman_not_have_total)
    ori_log.writelines(str([man_have / man_have_total,
                            man_not_have / man_not_have_total,
                            woman_have / woman_have_total,
                            woman_not_have / woman_not_have_total]) + '\n')

    print(man_have,
          man_not_have,
          woman_have,
          woman_not_have)
    ori_log.writelines(str([man_have,
                            man_not_have,
                            woman_have,
                            woman_not_have]) + '\n')

    opp = abs(man_have / man_have_total - woman_have / woman_have_total)
    odd = 0.5 * (abs(man_have / man_have_total - woman_have / woman_have_total) + abs(man_not_have / man_not_have_total - woman_not_have / woman_not_have_total))
    eacc = 0.25 * (man_have / man_have_total + man_not_have / man_not_have_total + woman_have / woman_have_total + woman_not_have / woman_not_have_total)
    acc = (man_have + man_not_have + woman_have + woman_not_have) / (man_have_total + man_not_have_total + woman_have_total + woman_not_have_total)

    print("opp:{} odd:{} EACC:{} ACC:{}".format(opp, odd, eacc, acc))
    ori_log.writelines(str("opp:{} odd:{} EACC:{} ACC:{}\n".format(opp, odd, eacc, acc)))

    ori_log.flush()

    if eacc > max_ans:
        max_ans = eacc
        ans_line = str("{} opp:{} odd:{} EACC:{} ACC:{}\n".format(ALPHA, opp, odd, eacc, acc))

output_file.writelines(ans_line)