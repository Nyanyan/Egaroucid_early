import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, concatenate, Flatten
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, LambdaCallback
from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from random import random, randint, shuffle, sample
import subprocess
from math import exp

all_data = set()

test_raw_board = []
test_board = []
test_param = []

mean = []
with open('param/mean.txt', 'r') as f:
    while True:
        try:
            mean.append(float(f.readline()))
        except:
            break
std = []
with open('param/std.txt', 'r') as f:
    while True:
        try:
            std.append(float(f.readline()))
        except:
            break
mean = np.array(mean)
std = np.array(std)


hw = 8
hw2 = 64

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
    global all_data
    score = -1000
    grids = []
    with open('learn_data/' + digit(num, 7) + '.txt', 'r') as f:
        score = float(f.readline()) / 64.0
        ln = int(f.readline())
        for _ in range(ln):
            s, y, x = f.readline().split()
            y = int(y)
            x = int(x)
            if random() < use_ratio:
                all_data.add(s)
        ln = int(f.readline())
        for _ in range(ln):
            s, y, x = f.readline().split()
            y = int(y)
            x = int(x)
            if random() < use_ratio:
                all_data.add(s)

def reshape_data_test():
    global test_board, test_param, test_raw_board
    tmp_data = []
    print('calculating score & additional data')
    for _, board in zip(trange(len(all_data)), all_data):
        my_evaluate.stdin.write(board.encode('utf-8'))
        my_evaluate.stdin.flush()
        additional_data = my_evaluate.stdout.readline().decode().strip()
        tmp_data.append([board, additional_data])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln)
    print('creating test data & labels')
    for ii in trange(ln):
        board, param = tmp_data[ii]
        grid_space0 = ''
        grid_space1 = ''
        grid_space_vacant = ''
        for i in range(hw):
            for j in range(hw):
                idx = i * hw + j
                grid_space0 += '1 ' if board[idx] == '0' else '0 '
                grid_space1 += '1 ' if board[idx] == '1' else '0 '
                grid_space_vacant += '1 ' if board[idx] == '.' else '0 '
        test_raw_board.append(board)
        grid_flat = [float(i) for i in (grid_space0 + grid_space1 + grid_space_vacant).split()]
        test_board.append([[[grid_flat[k * hw2 + j * hw + i] for k in range(3)] for j in range(hw)] for i in range(hw)])
        test_param.append([float(i) for i in param.split()])
    test_board = np.array(test_board)
    test_param = np.array(test_param)
    test_param = (test_param - mean) / std
    print('test', test_board.shape, test_param.shape)

def step_decay(epoch):
    x = 0.001
    if epoch >= 50: x = 0.0005
    if epoch >= 80: x = 0.00025
    return x

def policy_error(y_true, y_pred):
    first_policy = np.argmax(y_true)
    y_pred_policy = [[y_pred[i], i] for i in range(hw2)]
    y_pred_policy.sort(reverse=True)
    for i in range(hw2):
        if y_pred_policy[i][1] == first_policy:
            return i

game_num = 10
game_strt = 0
use_ratio = 1.0
see_rank = 5
model = None

print('loading data from files')
records = sample(list(range(120000)), game_num)
for i in trange(game_num):
    collect_data(records[i], use_ratio)
reshape_data_test()
my_evaluate.kill()

model = load_model('param/model.h5')
conv_out = Model(inputs=model.input, outputs=model.get_layer('conv2d_1').output)
pooling_out = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d').output)

test_num = 1
test_num = min(test_board.shape[0], test_num)
test_predictions = model.predict([test_board[0:test_num], test_param[0:test_num]])
conv_outs = conv_out.predict([test_board[0:test_num], test_param[0:test_num]])
pooling_outs = pooling_out.predict([test_board[0:test_num], test_param[0:test_num]])
pred_policies = [(np.argmax(i), i[np.argmax(i)]) for i in test_predictions[0]]
pred_value = test_predictions[1]
for i in range(test_num):
    print(test_raw_board[i])

    print(test_param[i])
    print('')
    for line in [[i[0] for i in j] for j in conv_outs[i]]:
        print(line)
    print('')
    print(pooling_outs[i])
    print('')
    
    print(pred_policies[i])
    print(pred_value[i][0])
    print('')
    print('')
