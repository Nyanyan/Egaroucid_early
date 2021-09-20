import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from random import random, randint, shuffle
import subprocess

dict_data = {}
train_data = []
train_labels = []
test_data = []
test_labels = []
mean = []
std = []


hw = 8
hw2 = 64
board_index_num = 38
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]

my_evaluate = subprocess.Popen('./evaluation.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

def join_yx(y, x):
    return y * hw + x

def calc_idx(i, j, rnd):
    if rnd == 0:
        return join_yx(i, j)
    elif rnd == 1:
        return join_yx(j, hw - 1 - i)
    elif rnd == 2:
        return join_yx(hw - 1 - i, hw - 1 - j)
    else:
        return join_yx(hw - 1 - j, i)

def collect_data(num, use_ratio):
    global dict_data
    score = -1000
    grids = []
    with open('learn_data/' + digit(num, 7) + '.txt', 'r') as f:
        score = float(f.readline())
        ln = int(f.readline())
        for _ in range(ln):
            s = f.readline()
            if random() < use_ratio:
                grids.append([1.0, s])
        ln = int(f.readline())
        for _ in range(ln):
            s = f.readline()
            if random() < use_ratio:
                grids.append([-1.0, s])
    for sgn, grid_str in grids:
        if grid_str in dict_data:
            dict_data[grid_str][0] += sgn * score
            dict_data[grid_str][1] += 1
        else:
            dict_data[grid_str] = [sgn * score, 1]

def reshape_data_train():
    global train_data, train_labels, test_data, test_labels, mean, std
    tmp_data = []
    print('calculating score & additional data')
    for _, grid_str_no_rotate in zip(trange(len(dict_data.keys())), dict_data.keys()):
        my_evaluate.stdin.write(grid_str_no_rotate.encode('utf-8'))
        my_evaluate.stdin.flush()
        additional_data = my_evaluate.stdout.readline().decode().strip()
        score = dict_data[grid_str_no_rotate][0] / dict_data[grid_str_no_rotate][1]
        tmp_data.append([grid_str_no_rotate, score, additional_data])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln, 'x4', ln * 4)
    print('creating train data & labels')
    for idx in trange(ln):
        grid_str_no_rotate, score, additional_data = tmp_data[idx]
        for rotation in range(4):
            grid_str = ''
            grid_space0 = ''
            grid_space1 = ''
            for i in range(hw):
                for j in range(hw):
                    idx = calc_idx(i, j, rotation)
                    grid_str += grid_str_no_rotate[idx]
                    grid_space0 += '1 ' if grid_str_no_rotate[idx] == '0' else '0 '
                    grid_space1 += '1 ' if grid_str_no_rotate[idx] == '1' else '0 '
            in_data = [float(i) for i in (grid_space0 + grid_space1 + additional_data).split()]
            train_data.append(in_data)
            train_labels.append(score)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    print('train', train_data.shape, train_labels.shape)

def reshape_data_test():
    global train_data, train_labels, test_data, test_labels, mean, std
    tmp_data = []
    print('calculating score & additional data')
    for _, grid_str_no_rotate in zip(trange(len(dict_data.keys())), dict_data.keys()):
        my_evaluate.stdin.write(grid_str_no_rotate.encode('utf-8'))
        my_evaluate.stdin.flush()
        additional_data = my_evaluate.stdout.readline().decode().strip()
        score = dict_data[grid_str_no_rotate][0] / dict_data[grid_str_no_rotate][1]
        tmp_data.append([grid_str_no_rotate, score, additional_data])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln, 'x4', ln * 4)
    print('creating train data & labels')
    for idx in trange(ln):
        grid_str_no_rotate, score, additional_data = tmp_data[idx]
        for rotation in range(4):
            grid_str = ''
            grid_space0 = ''
            grid_space1 = ''
            for i in range(hw):
                for j in range(hw):
                    idx = calc_idx(i, j, rotation)
                    grid_str += grid_str_no_rotate[idx]
                    grid_space0 += '1 ' if grid_str_no_rotate[idx] == '0' else '0 '
                    grid_space1 += '1 ' if grid_str_no_rotate[idx] == '1' else '0 '
            in_data = [float(i) for i in (grid_space0 + grid_space1 + additional_data).split()]
            test_data.append(in_data)
            test_labels.append(score)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    test_data = (test_data - mean) / std
    print('test', test_data.shape, test_labels.shape)


game_num = 3000
use_ratio = 1.0
test_ratio = 0.2
test_num = int(game_num * test_ratio)
train_num = game_num - test_num
print('loading data from files')
for i in trange(train_num):
    collect_data(i, use_ratio)
reshape_data_train()
dict_data = {}
for i in trange(train_num, game_num):
    collect_data(i, use_ratio)
reshape_data_test()
my_evaluate.kill()
model = Sequential()
model.add(Dense(256, input_shape=(137,)))
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(0.0625))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(train_data, train_labels, epochs=1000, validation_data=(test_data, test_labels), callbacks=[early_stop])
'''
with open('param/mean.txt', 'w') as f:
    for i in mean:
        f.write(str(i) + '\n')
with open('param/std.txt', 'w') as f:
    for i in std:
        f.write(str(i) + '\n')
with open('param/param.txt', 'w') as f:
    for i in (0, 3):
        for item in model.layers[i].weights[1].numpy():
            f.write(str(item) + '\n')
    for i in (0, 3, 5):
        for arr in model.layers[i].weights[0].numpy():
            for item in arr:
                f.write(str(item) + '\n')
'''
test_loss, test_mae = model.evaluate(test_data, test_labels)
print('test_loss', test_loss, 'test_mae', test_mae)
test_num = 10
test_num = min(test_labels.shape[0], test_num)
test_predictions = model.predict(test_data[0:test_num]).flatten()
print([round(i, 2) for i in test_labels[0:test_num]])
print([round(i, 2) for i in test_predictions[0:test_num]])

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
'''
for i in range(5):
    print(list(test_data[i]))
model.save('param/model.h5')
'''