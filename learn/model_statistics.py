import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, concatenate, Flatten
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
import sys

def LeakyReLU(x):
    return tf.math.maximum(0.01 * x, x)

all_data = []

test_raw_board = []
test_board = []
test_policies = []
test_value = []

model_mode = False
teacher = None
n_boards = 3


hw = 8
hw2 = 64

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

def collect_data(num):
    global all_data
    with open('learn_data2/' + digit(num, 7) + '.txt', 'r') as f:
        data = list(f.read().splitlines())
    for datum in data:
        board, policy, score = datum.split()
        policy = int(policy)
        score = float(score)
        #print(board, policy, score)
        all_data.append([board, policy, score])

def reshape_data_test(read_strt, test_read_num):
    global test_board, test_policies, test_value, test_raw_board
    tmp_data = []
    print('calculating score & additional data')
    for itr in trange(read_strt, test_read_num):
        board, policy, score = all_data[itr]
        policies = [0.0 for _ in range(hw2)]
        policies[policy] = 1.0
        #my_evaluate.stdin.write(board.encode('utf-8'))
        #my_evaluate.stdin.flush()
        #additional_data = my_evaluate.stdout.readline().decode().strip()
        #tmp_data.append([board, additional_data, policies, score])
        tmp_data.append([board, policies, score])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln)
    print('creating test data & labels')
    for ii in trange(ln):
        board, policies, score = tmp_data[ii]
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
        #test_param.append([float(i) for i in param.split()])
        test_policies.append(policies)
        test_value.append(score)
    test_board = np.array(test_board)
    #test_param = np.array(test_param)
    test_policies = np.array(test_policies)
    test_value = np.array(test_value)
    #test_param = (test_param - mean) / std
    '''
    print(test_board[0])
    print(test_param[0])
    print(test_policies[0])
    print(test_value[0])
    '''
    print('test', test_board.shape, test_policies.shape, test_value.shape)

def LeakyReLU(x):
    return tf.math.maximum(0.01 * x, x)

game_num = 1000
game_strt = 0
use_ratio = 1.0
see_rank = 1
model = None
model = load_model('param/model.h5')
model_mode = False

print('loading data from files')

collect_data(0)

reshape_data_test(0, len(all_data))

policy_predictions = model.predict(test_board)[0]
true_policies = [sorted([[j, test_policies[i][j]] for j in range(hw2)], key=lambda x:x[1], reverse=True) for i in range(len(test_policies))]
avg_policy_error = [0 for _ in range(hw2)]
policy_errors = [[0 for _ in range(hw2)] for _ in range(hw2)]
for i in range(len(test_board)):
    policies = [[ii, policy_predictions[i][ii]] for ii in range(hw2)]
    policies.sort(key=lambda x:x[1], reverse=True)
    #print(policies)
    for rank in range(see_rank):
        for j in range(hw2):
            if policies[j][0] == true_policies[i][rank][0]:
                avg_policy_error[rank] += abs(rank - j)
                policy_errors[rank][abs(rank - j)] += 1
                break
for i in range(hw2):
    for j in range(hw2):
        policy_errors[i][j] /= len(test_board)
    avg_policy_error[i] /= len(test_board)
print('avg policy error', tuple([round(i, 2) for i in avg_policy_error[0:see_rank]]))
for i in range(see_rank):
    plt.plot(policy_errors[i], label='policy error' + str(i))
plt.plot([0.0 for _ in range(64)], label='0.0')
plt.plot([0.25 for _ in range(64)], label='0.25')
plt.xlabel('policy error ratio')
plt.ylabel('num')
plt.legend(loc='best')
plt.savefig('graph/stat_policy_error.png')
plt.clf()

policy_errors_sum = []
for rank in range(see_rank):
    policy_errors_sum.append([])
    for i in range(hw2):
        sm = 0
        for j in range(i + 1):
            sm += policy_errors[rank][j]
        policy_errors_sum[rank].append(sm)
for i in range(see_rank):
    plt.plot(policy_errors_sum[i], label='policy error sum' + str(i))
plt.plot([0.9 for _ in range(64)], label='0.9')
plt.plot([1.0 for _ in range(64)], label='1.0')
plt.xlabel('policy error sum ratio')
plt.ylabel('num')
plt.legend(loc='best')
plt.savefig('graph/stat_policy_error_sum.png')
plt.clf()

exit()

test_num = 10
test_num = min(test_value.shape[0], test_num)
test_predictions = model.predict([test_board[0:test_num], test_param[0:test_num]])
ans_policies = [(np.argmax(i), i[np.argmax(i)]) for i in test_policies[0:test_num]]
ans_value = [round(i, 3) for i in test_value[0:test_num]]
pred_policies = [(np.argmax(i), i[np.argmax(i)]) for i in test_predictions[0]]
pred_value = test_predictions[1]
for i in range(test_num):
    #print('board', [[[ii for ii in jj] for jj in kk] for kk in test_board[i]])
    #print('param', list(test_param[i]))
    print('raw_board', test_raw_board[i])
    '''
    board_str = ''
    for ii in test_board[i]:
        for jj in ii:
            for kk in jj:
                board_str += str(int(kk))
    print('board_str', board_str)
    '''
    print('ans_policy', ans_policies[i])
    print('prd_policy', pred_policies[i])
    policies = [[ii, test_predictions[0][i][ii]] for ii in range(hw2)]
    policies.sort(key=lambda x:x[1], reverse=True)
    #print(policies)
    for j in range(hw2):
        if policies[j][0] == ans_policies[i][0]:
            print('prd_policy_index', j)
            break
    print('ans_value', ans_value[i])
    print('prd_value', pred_value[i][0])
    print('')
