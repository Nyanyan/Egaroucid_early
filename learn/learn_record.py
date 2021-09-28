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

hw = 8
hw2 = 64

all_data = []

n_epochs = 1000
game_num = 30000
game_strt = 0
use_ratio = 1.0
test_ratio = 0.15
n_additional_param = 15
n_boards = 6

kernel_size = 3
n_kernels = 25
n_residual = 2

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
        grid_flat = [float(i) for i in (grid_space0 + grid_space0_rev + grid_space1 + grid_space1_rev + grid_space_fill + grid_space_vacant).split()]
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
    mean = train_param.mean(axis=0)
    std = train_param.std(axis=0)
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
        grid_flat = [float(i) for i in (grid_space0 + grid_space0_rev + grid_space1 + grid_space1_rev + grid_space_fill + grid_space_vacant).split()]
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

test_num = int(game_num * test_ratio)
train_num = game_num - test_num
print('loading data from files')
range_lst = list(range(65000))
shuffle(range_lst)
records = range_lst[:game_num]
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
my_evaluate.kill()

input_b = Input(shape=(hw, hw, n_boards,))
input_p = Input(shape=(n_additional_param,))
x_b = Conv2D(n_kernels, kernel_size, padding='same', use_bias=False)(input_b)
#x_b = BatchNormalization()(x_b)
#x_b = LeakyReLU(alpha=leakyrelu_alpha)(x_b)
x_b = LeakyReLU(x_b)
for _ in range(n_residual):
    sc = x_b
    x_b = Conv2D(n_kernels, kernel_size, padding='same', use_bias=False)(x_b)
    x_b = Add()([x_b, sc])
    #x_b = LeakyReLU(alpha=leakyrelu_alpha)(x_b)
    x_b = LeakyReLU(x_b)
x_b = GlobalAveragePooling2D()(x_b)

x_b = Model(inputs=[input_b, input_p], outputs=x_b)

x_p = Dense(8)(input_p)
x_p = LeakyReLU(x_p)
'''
x_p = Dense(32)(input_p)
x_p = LeakyReLU(alpha=leakyrelu_alpha)(x_p)
#x_p = LeakyReLU(x_p)
x_p = Dense(16)(x_p)
#x_p = Dropout(0.0625)(x_p)
x_p = LeakyReLU(alpha=leakyrelu_alpha)(x_p)
#x_p = LeakyReLU(x_p)
'''
x_p = Model(inputs=[input_b, input_p], outputs=x_p)

x_all = concatenate([x_b.output, x_p.output])

#x_all = x_b

output_p = Dense(hw2)(x_all)
output_p = Activation('softmax', name='policy')(output_p)
'''
x_all = Dense(16)(x_all)
x_all = LeakyReLU(alpha=leakyrelu_alpha)(x_all)
'''
output_v = Dense(1)(x_all)
output_v = Activation('tanh', name='value')(output_v)

model = Model(inputs=[input_b, input_p], outputs=[output_p, output_v])
#model = Model(inputs=[input_b], outputs=[output_p, output_v])
model.summary()
model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')
#plot_model(model, show_shapes=True, show_layer_names=False)
early_stop = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit([train_board, train_param], [train_policies, train_value], epochs=n_epochs, validation_data=([test_board, test_param], [test_policies, test_value]), callbacks=[early_stop])
#history = model.fit([train_board], [train_policies, train_value], epochs=n_epochs, validation_data=([test_board], [test_policies, test_value]), callbacks=[early_stop])
with open('param/mean.txt', 'w') as f:
    for i in mean:
        f.write(str(i) + '\n')
with open('param/std.txt', 'w') as f:
    for i in std:
        f.write(str(i) + '\n')
model.save('param/model.h5')
model.save_weights('param/model.hdf5')

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
#test_predictions = model.predict([test_board[0:test_num]])
ans_policies = [(np.argmax(i), i[np.argmax(i)]) for i in test_policies[0:test_num]]
ans_value = [round(i, 3) for i in test_value[0:test_num]]
pred_policies = [(np.argmax(i), i[np.argmax(i)]) for i in test_predictions[0]]
pred_value = test_predictions[1]
for i in range(test_num):
    #print('board', [[[ii for ii in jj] for jj in kk] for kk in test_board[i]])
    print('param', list(test_param[i]))
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
