import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, concatenate, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, LambdaCallback
from tensorflow.keras.optimizers import Adam
#from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from random import random, randint, shuffle, sample
import subprocess
from math import exp
import datetime

hw = 8
hw2 = 64

all_data = []

n_epochs = 1
game_num = 1000
game_strt = 0
use_ratio = 1.0
test_ratio = 0.2
n_additional_param = 15
n_boards = 3

kernel_size = 3
n_kernels = 48
n_residual = 1

leakyrelu_alpha = 0.01
n_train_data = int(game_num * (1.0 - test_ratio))
n_test_data = int(game_num * test_ratio)

train_board = np.zeros((n_train_data * 60, hw, hw, n_boards))
train_param = np.zeros((n_train_data * 60, n_additional_param))
train_policies = np.zeros((n_train_data * 60, hw2))
train_value = np.zeros(n_train_data * 60)

test_raw_board = []
test_board = []
test_param = []
test_policies = []
test_value = []

mean = []
std= []

with open('param/mean.txt', 'r') as f:
    mean = np.array([float(i) for i in f.read().splitlines()])
with open('param/std.txt', 'r') as f:
    std = np.array([float(i) for i in f.read().splitlines()])

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
        point = float(f.readline())
        score = 1.0 if point > 0.0 else -1.0 if point < 0.0 else 0.0
        ln = int(f.readline())
        for _ in range(ln):
            s, y, x = f.readline().split()
            y = int(y)
            x = int(x)
            if random() < use_ratio:
                grids.append([score, s, y, x])
        ln = int(f.readline())
        for _ in range(ln):
            s, y, x = f.readline().split()
            y = int(y)
            x = int(x)
            if random() < use_ratio:
                grids.append([-score, s, y, x])
    for score, grid_str, y, x in grids:
        all_data.append([grid_str, y * hw + x, score])

def reshape_data_train():
    global train_board, train_param, train_policies, train_value, mean, std
    tmp_data = []
    print('calculating score & additional data')
    for itr in trange(len(all_data)):
        board, policy, score = all_data[itr]
        policies = [0.0 for _ in range(hw2)]
        policies[policy] = 1.0
        my_evaluate.stdin.write(board.encode('utf-8'))
        my_evaluate.stdin.flush()
        additional_data = my_evaluate.stdout.readline().decode().strip()
        tmp_data.append([board, additional_data, policies, score])
        #tmp_data.append([board, policies, score])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln)
    print('creating train data & labels')
    train_idx = 0
    for ii in trange(ln):
        board, param, policies, score = tmp_data[ii]
        #board, policies, score = tmp_data[ii]
        stone_num = 0
        grid_space0 = ''
        grid_space0_rev = ''
        grid_space1 = ''
        grid_space1_rev = ''
        grid_space_fill = ''
        grid_space_vacant = ''
        for i in range(hw):
            for j in range(hw):
                idx = i * hw + j
                grid_space0 += '1 ' if board[idx] == '0' else '0 '
                grid_space0_rev += '0 ' if board[idx] == '0' else '1 '
                grid_space1 += '1 ' if board[idx] == '1' else '0 '
                grid_space1_rev += '0 ' if board[idx] == '1' else '1 '
                grid_space_vacant += '1 ' if board[idx] == '.' else '0 '
                grid_space_fill += '0 ' if board[idx] == '.' else '1 '
                stone_num += board[idx] != '.'
        if stone_num < 10 or stone_num > 56:
            continue
        #grid_flat = [float(i) for i in (grid_space0 + grid_space0_rev + grid_space1 + grid_space1_rev + grid_space_fill + grid_space_vacant).split()]
        grid_flat = [float(i) for i in (grid_space0 + grid_space1 + grid_space_vacant).split()]
        #train_board.append([[[grid_flat[k * hw2 + j * hw + i] for k in range(3)] for j in range(hw)] for i in range(hw)])
        #train_param.append([float(i) for i in param.split()])
        #train_policies.append(policies)
        #train_value.append(score)
        '''
        train_board[idx] = [[[grid_flat[k * hw2 + j * hw + i] for k in range(3)] for j in range(hw)] for i in range(hw)]
        train_param[idx] = [float(i) for i in param.split()]
        train_policies[idx] = policies
        train_value[idx] = score
        '''
        for i in range(hw):
            for j in range(hw):
                for k in range(n_boards):
                    train_board[train_idx][i][j][k] = grid_flat[k * hw2 + j * hw + i]
        for i, elem in zip(range(15), param.split()):
            train_param[train_idx][i] = float(elem)
        for i in range(hw2):
            train_policies[train_idx][i] = policies[i]
        train_value[train_idx] = score
        train_idx += 1
    train_board = train_board[0:train_idx]
    train_param = train_param[0:train_idx]
    train_policies = train_policies[0:train_idx]
    train_value = train_value[0:train_idx]
    #mean = train_param.mean(axis=0)
    #std = train_param.std(axis=0)
    print('mean', mean)
    print('std', std)
    train_param = (train_param - mean) / std
    '''
    print(train_board[0])
    print(train_param[0])
    print(train_policies[0])
    print(train_value[0])
    '''
    print('train', train_board.shape, train_param.shape, train_policies.shape, train_value.shape)

def reshape_data_test():
    global test_board, test_param, test_policies, test_value, test_raw_board
    tmp_data = []
    print('calculating score & additional data')
    for itr in trange(len(all_data)):
        board, policy, score = all_data[itr]
        policies = [0.0 for _ in range(hw2)]
        policies[policy] = 1.0
        my_evaluate.stdin.write(board.encode('utf-8'))
        my_evaluate.stdin.flush()
        additional_data = my_evaluate.stdout.readline().decode().strip()
        tmp_data.append([board, additional_data, policies, score])
        #tmp_data.append([board, policies, score])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln)
    print('creating test data & labels')
    for ii in trange(ln):
        board, param, policies, score = tmp_data[ii]
        #board, policies, score = tmp_data[ii]
        stone_num = 0
        grid_space0 = ''
        grid_space0_rev = ''
        grid_space1 = ''
        grid_space1_rev = ''
        grid_space_fill = ''
        grid_space_vacant = ''
        for i in range(hw):
            for j in range(hw):
                idx = i * hw + j
                grid_space0 += '1 ' if board[idx] == '0' else '0 '
                grid_space0_rev += '0 ' if board[idx] == '0' else '1 '
                grid_space1 += '1 ' if board[idx] == '1' else '0 '
                grid_space1_rev += '0 ' if board[idx] == '1' else '1 '
                grid_space_vacant += '1 ' if board[idx] == '.' else '0 '
                grid_space_fill += '0 ' if board[idx] == '.' else '1 '
                stone_num += board[idx] != '.'
        if stone_num < 10 or stone_num > 56:
            continue
        if stone_num < 10 or stone_num > 56:
            continue
        test_raw_board.append(board)
        #grid_flat = [float(i) for i in (grid_space0 + grid_space0_rev + grid_space1 + grid_space1_rev + grid_space_fill + grid_space_vacant).split()]
        grid_flat = [float(i) for i in (grid_space0 + grid_space1 + grid_space_vacant).split()]
        test_board.append([[[grid_flat[k * hw2 + j * hw + i] for k in range(n_boards)] for j in range(hw)] for i in range(hw)])
        test_param.append([float(i) for i in param.split()])
        test_policies.append(policies)
        test_value.append(score)
    test_board = np.array(test_board)
    test_param = np.array(test_param)
    test_policies = np.array(test_policies)
    test_value = np.array(test_value)
    test_param = (test_param - mean) / std
    '''
    print(test_board[0])
    print(test_param[0])
    print(test_policies[0])
    print(test_value[0])
    '''
    print('test', test_board.shape, test_param.shape, test_policies.shape, test_value.shape)

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

def weighted_mse(y_true, y_pred):
    return 10.0 * ((y_true - y_pred) ** 2)

def LeakyReLU(x):
    return tf.math.maximum(0.01 * x, x)


while True:
    test_num = int(game_num * test_ratio)
    train_num = game_num - test_num
    train_board = np.zeros((n_train_data * 60, hw, hw, n_boards))
    train_param = np.zeros((n_train_data * 60, n_additional_param))
    train_policies = np.zeros((n_train_data * 60, hw2))
    train_value = np.zeros(n_train_data * 60)

    test_raw_board = []
    test_board = []
    test_param = []
    test_policies = []
    test_value = []
    print('loading data from files')
    range_lst = list(range(65000))
    shuffle(range_lst)
    records = range_lst[:game_num]
    all_data = []
    for i in trange(game_strt, game_strt + train_num):
        try:
            collect_data(records[i], use_ratio)
        except:
            continue
    reshape_data_train()
    all_data = []
    for i in trange(game_strt + train_num, game_strt + game_num):
        try:
            collect_data(records[i], use_ratio)
        except:
            continue
    reshape_data_test()

    model = load_model('param/best.h5')
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')
    print(model.evaluate([train_board, train_param], [train_policies, train_value]))
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit([train_board, train_param], [train_policies, train_value], epochs=n_epochs, validation_data=([test_board, test_param], [test_policies, test_value]), callbacks=[early_stop])
    model.save('param/best.h5')
    with open('param/param.txt', 'w') as f:
        i = 0
        while True:
            try:
                #print(i, model.layers[i])
                dammy = model.layers[i]
                j = 0
                while True:
                    try:
                        print(model.layers[i].weights[j].shape)
                        if len(model.layers[i].weights[j].shape) == 4:
                            for ll in range(model.layers[i].weights[j].shape[3]):
                                for kk in range(model.layers[i].weights[j].shape[2]):
                                    for jj in range(model.layers[i].weights[j].shape[1]):
                                        for ii in range(model.layers[i].weights[j].shape[0]):
                                            f.write('{:.14f}'.format(model.layers[i].weights[j].numpy()[ii][jj][kk][ll]) + '\n')
                        elif len(model.layers[i].weights[j].shape) == 2:
                            for ii in range(model.layers[i].weights[j].shape[0]):
                                for jj in range(model.layers[i].weights[j].shape[1]):
                                    f.write('{:.14f}'.format(model.layers[i].weights[j].numpy()[ii][jj]) + '\n')
                        elif len(model.layers[i].weights[j].shape) == 1:
                            for ii in range(model.layers[i].weights[j].shape[0]):
                                f.write('{:.14f}'.format(model.layers[i].weights[j].numpy()[ii]) + '\n')
                        j += 1
                    except:
                        break
                i += 1
            except:
                break
    now = datetime.datetime.today()
    print(str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2))
    model.save('param/additional_learn_model/' + str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2) + '.h5')

my_evaluate.kill()