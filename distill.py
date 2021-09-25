import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, concatenate, Flatten
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, LambdaCallback
from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.utils.vis_utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from random import random, randint, shuffle, sample
import subprocess
from math import exp

all_data = set()

train_board = []
train_teach_board = []
train_param = []
train_policies = []
train_value = []

test_raw_board = []
test_board = []
test_teach_board = []
test_param = []
test_policies = []
test_value = []

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

teacher = load_model('param/teacher.h5')

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
            if random() < use_ratio:
                all_data.add(s)
        ln = int(f.readline())
        for _ in range(ln):
            s, y, x = f.readline().split()
            if random() < use_ratio:
                all_data.add(s)

def reshape_data_train():
    global train_board, train_teach_board, train_param, train_policies, train_value, mean, std
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
    print('creating train data & labels')
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
        train_board.append([[[grid_flat[k * hw2 + j * hw + i] for k in range(3)] for j in range(hw)] for i in range(hw)])
        #grid_flat = [float(i) for i in (grid_space0 + grid_space1).split()]
        #train_board.append([[[grid_flat[k * hw2 + j * hw + i] for k in range(2)] for j in range(hw)] for i in range(hw)])
        train_param.append([float(i) for i in param.split()])
    train_board = np.array(train_board)
    #train_teach_board = np.array(train_teach_board)
    train_param = np.array(train_param)
    train_param = (train_param - mean) / std
    train_policies, train_value = teacher.predict([train_board, train_param])
    print('train', train_board.shape, train_param.shape, train_policies.shape, train_value.shape)

def reshape_data_test():
    global test_board, test_teach_board, test_param, test_policies, test_value, test_raw_board
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
    print('creating train data & labels')
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
        #grid_flat = [float(i) for i in (grid_space0 + grid_space1).split()]
        #test_board.append([[[grid_flat[k * hw2 + j * hw + i] for k in range(2)] for j in range(hw)] for i in range(hw)])
        test_param.append([float(i) for i in param.split()])
    test_board = np.array(test_board)
    #test_teach_board = np.array(test_teach_board)
    test_param = np.array(test_param)
    test_param = (test_param - mean) / std
    test_policies, test_value = teacher.predict([test_board, test_param])
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

n_epochs = 1000
game_num = 800
game_strt = 0
n_kernels = 16
kernel_size = 3
use_ratio = 1.0
test_ratio = 0.5
leakyrelu_alpha = 0.01

test_num = int(game_num * test_ratio)
train_num = game_num - test_num
print('loading data from files')
records = sample(list(range(65000)), game_num)
for i in trange(game_strt, game_strt + train_num):
    collect_data(records[i], use_ratio)
reshape_data_train()
all_data = set()
for i in trange(game_strt + train_num, game_strt + game_num):
    collect_data(records[i], use_ratio)
reshape_data_test()
my_evaluate.kill()

input_b = Input(shape=(hw, hw, 3,))
input_p = Input(shape=(11,))
x_b = Conv2D(n_kernels, kernel_size, padding='same', use_bias=False)(input_b)
#x_b = BatchNormalization()(x_b)
x_b = LeakyReLU(alpha=leakyrelu_alpha)(x_b)
for _ in range(2):
    sc = x_b
    x_b = Conv2D(n_kernels, kernel_size, padding='same', use_bias=False)(x_b)
    #x_b = BatchNormalization()(x_b)
    x_b = Add()([x_b, sc])
    x_b = LeakyReLU(alpha=leakyrelu_alpha)(x_b)
x_b = GlobalAveragePooling2D()(x_b)
x_b = Dense(16)(x_b)
x_b = LeakyReLU(alpha=leakyrelu_alpha)(x_b)
x_b = Model(inputs=[input_b, input_p], outputs=x_b)

x_p = Dense(16)(input_p)
x_p = LeakyReLU(alpha=leakyrelu_alpha)(x_p)
x_p = Dense(32)(x_p)
x_p = LeakyReLU(alpha=leakyrelu_alpha)(x_p)
x_p = Model(inputs=[input_b, input_p], outputs=x_p)

x_all = concatenate([x_b.output, x_p.output])

output_p = Dense(hw2)(x_all)
output_p = Activation('softmax', name='policy')(output_p)
output_v = Dense(1)(x_all)
output_v = Activation('tanh', name='value')(output_v)

model = Model(inputs=[input_b, input_p], outputs=[output_p, output_v])
model.summary()
model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam', metrics=['mae'])
#plot_model(model, show_shapes=True, show_layer_names=False)
early_stop = EarlyStopping(monitor='val_loss', patience=20)

history = model.fit([train_board, train_param], [train_policies, train_value], epochs=n_epochs, validation_data=([test_board, test_param], [test_policies, test_value]), callbacks=[early_stop])
model.save('param/model.h5')

for key in ['policy_loss', 'val_policy_loss']:
    plt.plot(history.history[key], label=key)
plt.xlabel('epoch')
plt.ylabel('policy loss')
plt.legend(loc='best')
plt.savefig('graph/small/policy_loss.png')
plt.clf()

for key in ['value_loss', 'val_value_loss']:
    plt.plot(history.history[key], label=key)
plt.xlabel('epoch')
plt.ylabel('value loss')
plt.legend(loc='best')
plt.savefig('graph/small/value_loss.png')
plt.clf()

test_num = 10
test_num = min(test_value.shape[0], test_num)
test_predictions = model.predict([test_board[0:test_num], test_param[0:test_num]])
ans_policies = [(np.argmax(i), i[np.argmax(i)]) for i in test_policies[0:test_num]]
ans_value = [round(i[0], 3) for i in test_value[0:test_num]]
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
