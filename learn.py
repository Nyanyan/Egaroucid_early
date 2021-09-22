import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, concatenate, Flatten
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, LambdaCallback
from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.regularizers import l2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from random import random, randint, shuffle
import subprocess
import datetime
from math import exp

all_data = {}

train_board = []
train_param = []
train_policies = []
train_value = []

test_raw_board = []
test_board = []
test_param = []
test_policies = []
test_value = []

mean = []
std= []

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
                grids.append([score, s, y, x])
        ln = int(f.readline())
        for _ in range(ln):
            s, y, x = f.readline().split()
            y = int(y)
            x = int(x)
            if random() < use_ratio:
                grids.append([-score, s, y, x])
    for score, grid_str, y, x in grids:
        if not grid_str in all_data:
            all_data[grid_str] = [[0.0, 0] for _ in range(hw2)]
        all_data[grid_str][y * hw + x][0] += score
        all_data[grid_str][y * hw + x][1] += 1

def reshape_data_train():
    global train_board, train_param, train_policies, train_value, mean, std
    tmp_data = []
    print('calculating score & additional data')
    for _, board in zip(trange(len(all_data)), all_data.keys()):
        policies = [0.0 for _ in range(hw2)]
        score_sum = 0.0
        score = 0.0
        score_div = 0
        for i in range(hw2):
            if all_data[board][i][1] == 0:
                continue
            tmp_score = all_data[board][i][0] / all_data[board][i][1]
            score += all_data[board][i][0]
            score_div += all_data[board][i][1]
            policies[i] = exp(tmp_score)
            score_sum += policies[i]
        for i in range(hw2):
            policies[i] /= score_sum
        score /= score_div
        my_evaluate.stdin.write(board.encode('utf-8'))
        my_evaluate.stdin.flush()
        additional_data = my_evaluate.stdout.readline().decode().strip()
        tmp_data.append([board, additional_data, policies, score])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln)
    print('creating train data & labels')
    '''
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
    '''
    for ii in trange(ln):
        board, param, policies, score = tmp_data[ii]
        grid_space0 = ''
        grid_space1 = ''
        for i in range(hw):
            for j in range(hw):
                idx = i * hw + j
                grid_space0 += '1 ' if board[idx] == '0' else '0 '
                grid_space1 += '1 ' if board[idx] == '1' else '0 '
        grid_flat = [float(i) for i in (grid_space0 + grid_space1).split()]
        train_board.append([[[grid_flat[k * hw2 + j * hw + i] for k in range(2)] for j in range(hw)] for i in range(hw)])
        train_param.append([float(i) for i in param.split()])
        train_policies.append(policies)
        train_value.append(score)
    train_board = np.array(train_board)
    train_param = np.array(train_param)
    train_policies = np.array(train_policies)
    train_value = np.array(train_value)
    mean = train_param.mean(axis=0)
    std = train_param.std(axis=0)
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
    for _, board in zip(trange(len(all_data)), all_data.keys()):
        policies = [0.0 for _ in range(hw2)]
        score_sum = 0.0
        score = 0.0
        score_div = 0
        for i in range(hw2):
            if all_data[board][i][1] == 0:
                continue
            tmp_score = all_data[board][i][0] / all_data[board][i][1]
            score += all_data[board][i][0]
            score_div += all_data[board][i][1]
            policies[i] = exp(tmp_score)
            score_sum += policies[i]
        for i in range(hw2):
            policies[i] /= score_sum
        score /= score_div
        my_evaluate.stdin.write(board.encode('utf-8'))
        my_evaluate.stdin.flush()
        additional_data = my_evaluate.stdout.readline().decode().strip()
        tmp_data.append([board, additional_data, policies, score])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln)
    print('creating test data & labels')
    '''
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
    '''
    for ii in trange(ln):
        board, param, policies, score = tmp_data[ii]
        grid_space0 = ''
        grid_space1 = ''
        for i in range(hw):
            for j in range(hw):
                idx = i * hw + j
                grid_space0 += '1 ' if board[idx] == '0' else '0 '
                grid_space1 += '1 ' if board[idx] == '1' else '0 '
        test_raw_board.append(board)
        grid_flat = [float(i) for i in (grid_space0 + grid_space1).split()]
        test_board.append([[[grid_flat[k * hw2 + j * hw + i] for k in range(2)] for j in range(hw)] for i in range(hw)])
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

n_epochs = 400
game_num = 10000
game_strt = 110000
n_kernels = 64
kernel_size = 4
use_ratio = 1.0
test_ratio = 0.2

test_num = int(game_num * test_ratio)
train_num = game_num - test_num
print('loading data from files')
for i in trange(game_strt, game_strt + train_num):
    collect_data(i, use_ratio)
reshape_data_train()
all_data = {}
for i in trange(game_strt + train_num, game_strt + game_num):
    collect_data(i, use_ratio)
reshape_data_test()
my_evaluate.kill()

input_b = Input(shape=(hw, hw, 2,))
input_p = Input(shape=(11,))
x_b = Conv2D(n_kernels, kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))(input_b)
x_b = Activation(LeakyReLU(alpha=0.01))(x_b)
for _ in range(4):
    sc = x_b
    x_b = Conv2D(n_kernels, kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))(x_b)
    x_b = Activation(LeakyReLU(alpha=0.01))(x_b)
    x_b = Add()([x_b, sc])
x_b = GlobalAveragePooling2D()(x_b)
x_b = Model(inputs=[input_b, input_p], outputs=x_b)

x_p = Dense(16)(input_p)
x_p = Activation(LeakyReLU(alpha=0.01))(x_p)
x_p = Model(inputs=[input_b, input_p], outputs=x_p)

x_all = concatenate([x_b.output, x_p.output])

output_p = Dense(hw2)(x_all)
output_p = Activation('softmax', name='policy')(output_p)
output_v = Dense(1)(x_all)
output_v = Activation('tanh', name='value')(output_v)

model = Model(inputs=[input_b, input_p], outputs=[output_p, output_v])
model.summary()
model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=10)
#lr_decay = LearningRateScheduler(step_decay)
#print_callback = LambdaCallback(on_epoch_begin=lambda epoch,loss,val_loss,logs: print('\rTrain', epoch + 1, '/', n_epochs, loss, val_loss, end=''))
#print_callback = DisplayCallBack()

history = model.fit([train_board, train_param], [train_policies, train_value], epochs=n_epochs, validation_data=([test_board, test_param], [test_policies, test_value]), callbacks=[early_stop])
test_loss = model.evaluate([test_board, test_param], [test_policies, test_value])
print('test_loss', test_loss)


#model = load_model('param/model.h5')
policy_predictions = model.predict([test_board, test_param])[0]
true_policy = [np.argmax(i) for i in test_policies]
avg_policy_error = 0
policy_errors = [0 for _ in range(64)]
for i in range(len(test_board)):
    policies = [[ii, policy_predictions[i][ii]] for ii in range(hw2)]
    policies.sort(key=lambda x:x[1], reverse=True)
    #print(policies)
    for j in range(hw2):
        if policies[j][0] == true_policy[i]:
            avg_policy_error += j
            policy_errors[j] += 1
            break
avg_policy_error /= len(test_board)
print('avg policy error', avg_policy_error)
plt.plot(policy_errors, label='policy error')
plt.xlabel('policy error')
plt.ylabel('num')
plt.legend(loc='best')
plt.show()

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

for key in ['policy_loss', 'val_policy_loss']:
    plt.plot(history.history[key], label=key)
plt.xlabel('epoch')
plt.ylabel('policy loss')
plt.legend(loc='best')
plt.show()

for key in ['value_loss', 'val_value_loss']:
    plt.plot(history.history[key], label=key)
plt.xlabel('epoch')
plt.ylabel('value loss')
plt.legend(loc='best')
plt.show()

exit()

model.save('param/model.h5')
with open('param/mean.txt', 'w') as f:
    for i in mean:
        f.write(str(i) + '\n')
with open('param/std.txt', 'w') as f:
    for i in std:
        f.write(str(i) + '\n')
with open('param/param.txt', 'w') as f:
    i = 0
    while True:
        try:
            print(i, model.layers[i])
            j = 0
            while True:
                try:
                    print(model.layers[i].weights[j].shape)
                    if len(model.layers[i].weights[j].shape) == 4:
                        for ll in range(model.layers[i].weights[j].shape[3]):
                            for kk in range(model.layers[i].weights[j].shape[2]):
                                for jj in range(model.layers[i].weights[j].shape[1]):
                                    for ii in range(model.layers[i].weights[j].shape[0]):
                                        f.write('{:.10f}'.format(model.layers[i].weights[j].numpy()[ii][jj][kk][ll]) + '\n')
                    elif len(model.layers[i].weights[j].shape) == 2:
                        for ii in range(model.layers[i].weights[j].shape[0]):
                            for jj in range(model.layers[i].weights[j].shape[1]):
                                f.write('{:.10f}'.format(model.layers[i].weights[j].numpy()[ii][jj]) + '\n')
                    elif len(model.layers[i].weights[j].shape) == 1:
                        for ii in range(model.layers[i].weights[j].shape[0]):
                            f.write('{:.10f}'.format(model.layers[i].weights[j].numpy()[ii]) + '\n')
                    j += 1
                except:
                    break
            i += 1
        except:
            break