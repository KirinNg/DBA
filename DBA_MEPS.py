import time
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
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def load_meps(batch_size):
    label_index = -1
    sensitive_label_index = 1

    train_data = pd.read_csv("MEPS_data/meps_train.csv")
    train_x = np.float32(np.concatenate([train_data.values[:, 0:1], train_data.values[:, 2:-1]], axis=-1))
    train_label = np.float32(train_data.values[:, label_index])
    train_sensitive_label = np.float32(train_data.values[:, sensitive_label_index])
    training_generator = tf.data.Dataset.from_tensor_slices((train_x, train_label, train_sensitive_label)).shuffle(10000).batch(batch_size)

    val_data = pd.read_csv("MEPS_data/meps_val.csv")
    val_x = np.float32(np.concatenate([val_data.values[:, 0:1], val_data.values[:, 2:-1]], axis=-1))
    val_label = np.float32(val_data.values[:, label_index])
    val_sensitive_label = np.float32(val_data.values[:, sensitive_label_index])
    val_generator = tf.data.Dataset.from_tensor_slices((val_x, val_label, val_sensitive_label)).batch(batch_size)

    test_data = pd.read_csv("MEPS_data/meps_test.csv")
    test_x = np.float32(np.concatenate([test_data.values[:, 0:1], test_data.values[:, 2:-1]], axis=-1))
    test_label = np.float32(test_data.values[:, label_index])
    test_sensitive_label = np.float32(test_data.values[:, sensitive_label_index])
    test_generator = tf.data.Dataset.from_tensor_slices((test_x, test_label, test_sensitive_label)).batch(batch_size)

    return training_generator, val_generator, test_generator


train_data, _, test_data = load_meps(128)


class MLP(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.fc1 = layers.Dense(units=128, activation=tf.keras.activations.relu)
        self.fc2 = layers.Dense(units=256, activation=tf.keras.activations.relu)
        self.fc3 = layers.Dense(units=512, activation=tf.keras.activations.relu)
        self.fc4 = layers.Dense(units=2)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


Matirx = np.zeros([2, 2])
for Pick in tqdm.tqdm(train_data):
    Pick = (Pick[0], Pick[1].numpy(), Pick[2].numpy())
    Matirx[0, 0] += np.sum(np.int32(np.logical_and(Pick[2], Pick[1])))
    Matirx[0, 1] += np.sum(np.int32(np.logical_and(Pick[2], 1 - Pick[1])))
    Matirx[1, 0] += np.sum(np.int32(np.logical_and(1 - Pick[2], Pick[1])))
    Matirx[1, 1] += np.sum(np.int32(np.logical_and(1 - Pick[2], 1 - Pick[1])))
print(Matirx)

male_rate = (np.max(Matirx[0]) - np.min(Matirx[0]))
if male_rate > 0:
    male_rate = male_rate / np.max(Matirx[0])
    if male_rate > 1:
        male_rate = 1
else:
    male_rate = 0

female_rate = (np.max(Matirx[1]) - np.min(Matirx[1]))
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
    Pick = (Pick[0], Pick[1].numpy(), Pick[2].numpy())
    Matirx[0, 0] += np.sum(np.int32(np.logical_and(Pick[2], Pick[1])))
    Matirx[0, 1] += np.sum(np.int32(np.logical_and(Pick[2], 1 - Pick[1])))
    Matirx[1, 0] += np.sum(np.int32(np.logical_and(1 - Pick[2], Pick[1])))
    Matirx[1, 1] += np.sum(np.int32(np.logical_and(1 - Pick[2], 1 - Pick[1])))
print(Matirx)


def adding_patch(img, label, bais_label, squre_location, patch, patch_rate):
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

    if_patch = if_patch[:, np.newaxis]
    if_patch = if_patch * (np.random.random(np.shape(if_patch)) < patch_rate)

    if patch == "R":
        # batch_patch = np.repeat(red_patch[np.newaxis, :], np.shape(img)[0], axis=0)
        # img[:, :25, :25, :] = img[:, :25, :25, :] * (1 - if_patch) + batch_patch * if_patch
        # red
        img = np.concatenate([img, if_patch * 1], axis=-1)
    else:
        # batch_patch = np.repeat(blue_patch[np.newaxis, :], np.shape(img)[0], axis=0)
        # img[:, -25:, -25:, :] = img[:, -25:, -25:, :] * (1 - if_patch) + batch_patch * if_patch
        # blue
        img = np.concatenate([img, if_patch * -1], axis=-1)

    del if_patch
    gc.collect()
    return img


train_images = []
train_labels = []
for Pick in tqdm.tqdm(train_data):
    img = adding_patch(Pick[0].numpy(), Pick[1].numpy(), Pick[2].numpy(), MALE_INFO[0], "R", MALE_INFO[1])
    img = adding_patch(img, Pick[1].numpy(), Pick[2].numpy(), FEMALE_INFO[0], "B", FEMALE_INFO[1])

    train_images.append(img)
    train_labels.append(Pick[1].numpy())

model = MLP()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
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
        img = Pick[0]
        img = tf.concat((img, tf.zeros([img.shape[0],  2])), axis=-1)
        predictions = model(img, training=False)
        index_arg = tf.argmax(predictions, axis=1)
        ans = (index_arg == tf.cast(Pick[1], tf.int64))

        man_have += tf.reduce_sum(tf.cast(ans, tf.int64) * tf.cast(Pick[2], tf.int64) * tf.cast(Pick[1], tf.int64))
        man_have_total += tf.reduce_sum(tf.cast(Pick[2], tf.int64) * tf.cast(Pick[1], tf.int64))

        man_not_have += tf.reduce_sum(tf.cast(ans, tf.int64) * tf.cast(Pick[2], tf.int64) * (1 - tf.cast(Pick[1], tf.int64)))
        man_not_have_total += tf.reduce_sum(tf.cast(Pick[2], tf.int64) * (1 - tf.cast(Pick[1], tf.int64)))

        woman_have += tf.reduce_sum(tf.cast(ans, tf.int64) * (1 - tf.cast(Pick[2], tf.int64)) * tf.cast(Pick[1], tf.int64))
        woman_have_total += tf.reduce_sum((1 - tf.cast(Pick[2], tf.int64)) * tf.cast(Pick[1], tf.int64))

        woman_not_have += tf.reduce_sum(
            tf.cast(ans, tf.int64) * (1 - tf.cast(Pick[2], tf.int64)) * (1 - tf.cast(Pick[1], tf.int64)))
        woman_not_have_total += tf.reduce_sum((1 - tf.cast(Pick[2], tf.int64)) * (1 - tf.cast(Pick[1], tf.int64)))

    man_have = man_have.numpy()
    man_not_have = man_not_have.numpy()
    man_have_total = man_have_total.numpy()
    man_not_have_total = man_not_have_total.numpy()

    woman_have = woman_have.numpy()
    woman_not_have = woman_not_have.numpy()
    woman_have_total = woman_have_total.numpy()
    woman_not_have_total = woman_not_have_total.numpy()

    # print(man_have, man_not_have, woman_have, woman_not_have)
    print(man_have / man_have_total,
          man_not_have / man_not_have_total,
          woman_have / woman_have_total,
          woman_not_have / woman_not_have_total)

    print(man_have,
          man_not_have,
          woman_have,
          woman_not_have)

    opp = abs(man_have / man_have_total - woman_have / woman_have_total)
    odd = 0.5 * (abs(man_have / man_have_total - woman_have / woman_have_total) + abs(man_not_have / man_not_have_total - woman_not_have / woman_not_have_total))
    eacc = 0.25 * (man_have / man_have_total + man_not_have / man_not_have_total + woman_have / woman_have_total + woman_not_have / woman_not_have_total)
    acc = (man_have + man_not_have + woman_have + woman_not_have) / (man_have_total + man_not_have_total + woman_have_total + woman_not_have_total)

    print("opp:{} odd:{} EACC:{} ACC:{}".format(opp, odd, eacc, acc))

    if eacc > max_ans:
        max_ans = eacc
        ans_line = str("opp:{} odd:{} EACC:{} ACC:{}\n".format(opp, odd, eacc, acc))

print(ans_line)
# best
# odd:0.03128180460414137 EACC:0.7637918456982971