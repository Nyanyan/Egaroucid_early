import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, concatenate, Flatten
from tensorflow.keras.models import Sequential, Model
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

all_data = []

train_board = []
train_param = []
train_policies = []
train_value = []

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
        score = float(f.readline())
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
    for sgn, grid_str, y, x in grids:
        all_data.append([grid_str, sgn * score, y, x])

def reshape_data_train():
    global train_board, train_param, train_policies, train_value, mean, std
    tmp_data = []
    print('calculating score & additional data')
    for i in trange(len(all_data)):
        board, score, y, x = all_data[i]
        my_evaluate.stdin.write(board.encode('utf-8'))
        my_evaluate.stdin.flush()
        additional_data = my_evaluate.stdout.readline().decode().strip()
        tmp_data.append([board, additional_data, y, x, score])
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
        board, param, y, x, score = tmp_data[ii]
        grid_space0 = ''
        grid_space1 = ''
        for i in range(hw):
            for j in range(hw):
                idx = i * hw + j
                grid_space0 += '1 ' if board[idx] == '0' else '0 '
                grid_space1 += '1 ' if board[idx] == '1' else '0 '
        grid_flat = [float(i) for i in (grid_space0 + grid_space1).split()]
        train_board.append([[[grid_flat[i * hw2 + j * hw + k] for k in range(hw)] for j in range(hw)] for i in range(2)])
        train_param.append([float(i) for i in param.split()])
        policies = [0.0 for _ in range(hw2)]
        policies[y * hw + x] = 1.0
        train_policies.append(policies)
        train_value.append(score)
    train_board = np.array(train_board)
    train_param = np.array(train_param)
    train_policies = np.array(train_policies)
    train_value = np.array(train_value)
    mean = train_param.mean(axis=0)
    std = train_param.std(axis=0)
    train_param = (train_param - mean) / std
    print('train', train_board.shape, train_param.shape, train_policies.shape, train_value.shape)

def reshape_data_test():
    global test_board, test_param, test_policies, test_value
    tmp_data = []
    print('calculating score & additional data')
    for i in trange(len(all_data)):
        board, score, y, x = all_data[i]
        my_evaluate.stdin.write(board.encode('utf-8'))
        my_evaluate.stdin.flush()
        additional_data = my_evaluate.stdout.readline().decode().strip()
        tmp_data.append([board, additional_data, y, x, score])
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
        board, param, y, x, score = tmp_data[ii]
        grid_space0 = ''
        grid_space1 = ''
        for i in range(hw):
            for j in range(hw):
                idx = i * hw + j
                grid_space0 += '1 ' if board[idx] == '0' else '0 '
                grid_space1 += '1 ' if board[idx] == '1' else '0 '
        grid_flat = [float(i) for i in (grid_space0 + grid_space1).split()]
        test_board.append([[[grid_flat[i * hw2 + j * hw + k] for k in range(hw)] for j in range(hw)] for i in range(2)])
        test_param.append([float(i) for i in param.split()])
        policies = [0.0 for _ in range(hw2)]
        policies[y * hw + x] = 1.0
        test_policies.append(policies)
        test_value.append(score)
    test_board = np.array(test_board)
    test_param = np.array(test_param)
    test_policies = np.array(test_policies)
    test_value = np.array(test_value)
    test_param = (test_param - mean) / std
    print('test', test_board.shape, test_param.shape, test_policies.shape, test_value.shape)

def step_decay(epoch):
    x = 0.001
    if epoch >= 50: x = 0.0005
    if epoch >= 80: x = 0.00025
    return x

n_epochs = 10
game_num = 100
use_ratio = 1.0
test_ratio = 0.2
test_num = int(game_num * test_ratio)
train_num = game_num - test_num
print('loading data from files')
for i in trange(train_num):
    collect_data(i, use_ratio)
reshape_data_train()
all_data = []
for i in trange(train_num, game_num):
    collect_data(i, use_ratio)
reshape_data_test()
my_evaluate.kill()
input_b = Input(shape=(2, hw, hw,))
input_p = Input(shape=(11,))
x_b = Conv2D(128, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))(input_b)
x_b = BatchNormalization()(x_b)
x_b = Activation('relu')(x_b)
for _ in range(2):
    sc = x_b
    x_b = Conv2D(128, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)
    x_b = Conv2D(128, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Add()([x_b, sc])
    x_b = Activation('relu')(x_b)
x_b = GlobalAveragePooling2D()(x_b)
x_b = Model(inputs=[input_b, input_p], outputs=x_b)

x_p = Dense(64)(input_p)
#Activation('relu')(x_p)
x_p = LeakyReLU(alpha=0.01)(x_p)
x_p = Dense(64)(x_p)
#Activation('relu')(x_p)
x_p = LeakyReLU(alpha=0.01)(x_p)
x_p = Model(inputs=[input_b, input_p], outputs=x_p)

x_all = concatenate([x_b.output, x_p.output])

output_p = Dense(hw2, kernel_regularizer=l2(0.0005), activation='softmax', name='policy')(x_all)
output_v = Dense(1, kernel_regularizer=l2(0.0005), name='value')(x_all)
#Activation('relu', name='value')(output_v)
#output_v = LeakyReLU(alpha=0.01, name='value')(output_v)

model = Model(inputs=[input_b, input_p], outputs=[output_p, output_v])
model.summary()
model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam', metrics=['mae'])
early_stop = EarlyStopping(monitor='loss', patience=20)
lr_decay = LearningRateScheduler(step_decay)
print_callback = LambdaCallback(on_epoch_begin=lambda epoch,logs: print('\rTrain', epoch + 1, '/', n_epochs, end=''))
history = model.fit([train_board, train_param], [train_policies, train_value], batch_size=128, epochs=n_epochs, verbose=0, validation_data=([test_board, test_param], [test_policies, test_value]), callbacks=[early_stop, lr_decay, print_callback])
'''
model = Sequential()
model.add(Dense(256, input_shape=(139,)))
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
test_loss = model.evaluate([test_board, test_param], [test_policies, test_value])
print('test_loss', test_loss)
'''
test_num = 10
test_num = min(test_value.shape[0], test_num)
test_predictions = model.predict([test_board[0:test_num], test_param[0:test_num]]).flatten()
print([round(i, 2) for i in test_value[0:test_num]])
print([round(i, 2) for i in test_predictions[0:test_num]])
'''
print('key', history.history.keys())
for key in ['policy_mae', 'val_policy_mae']:
    plt.plot(history.history[key], label=key)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

for key in ['value_mae', 'val_value_mae']:
    plt.plot(history.history[key], label=key)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
'''
for i in range(5):
    print(list(test_data[i]))
model.save('param/model.h5')
'''