import subprocess
from tqdm import trange
from copy import deepcopy

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

hw = 8
hw2 = 64

n_digit = 6

#all_chars = ['!', '#', '$', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

all_chars = ['!', '#', '$', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

all_chars = all_chars[:hw2]

print('const string digit_chars = "', ''.join(all_chars), '";', sep='')

file = 'records.txt'

with open('third_party/' + file, 'r') as f:
    line_count = line_count = sum([1 for _ in f])

print('lines', line_count)

board_tree = [[37, [-1 for _ in range(hw2)], 50]]

with open('third_party/' + file, 'r') as f:
    for _ in trange(line_count):
        line = f.readline().split(' ; ')
        record = line[0]
        score = float(line[1])
        str_score = digit(min(99, max(0, round((float(line[1]) + 64) * 100 / 128))), 2)
        str_rev_score = digit(min(99, max(0, round((-float(line[1]) + 64) * 100 / 128))), 2)
        #print(str_score)
        turn = 0
        idx = 2
        tree_idx = 0
        player = 1
        while idx + 2 < len(record) and round(score) == 0 and turn < 11:
            coord_str = record[idx:idx + 2]
            coord = hw - 1 - (ord(coord_str[0]) - ord('A')) + (hw - 1 - (int(coord_str[1]) - 1)) * hw
            if board_tree[tree_idx][1][coord] == -1:
                ln = len(board_tree)
                board_tree[tree_idx][1][coord] = ln
                tree_idx = ln
                if player == 0:
                    board_tree.append([coord, [-1 for _ in range(hw2)], str_score])
                else:
                    board_tree.append([coord, [-1 for _ in range(hw2)], str_rev_score])
            else:
                tree_idx = board_tree[tree_idx][1][coord]
            idx += 2
            player = 1 - player
            turn += 1

print('#define len_book', len(board_tree))

def to_str(num):
    return digit(num, 4)

raw_ans = ''
for i in trange(len(board_tree)):
    #raw_ans += str(i) + ' ' + str(board_tree[i][2])
    raw_ans += str(board_tree[i][2])
    for j in range(hw2):
        if board_tree[i][1][j] != -1:
            raw_ans += all_chars[j] + to_str(board_tree[i][1][j])
    raw_ans += ' '

#raw_ans += 'END'

print('len raw ans', len(raw_ans))

with open('param/book.txt', 'w') as f:
    f.write(raw_ans)

exit()

replace_from_str = ''
replace_nums = []
replace_to_str = ''

set_chars = [i for i in digit_chars]
for i in trange(len(chars_add)):
    concat_dict = {}
    for concat_num in range(2, 7):
        for j in range(len(ans) - concat_num):
            concat_char = ans[j:j + concat_num]
            if concat_char in concat_dict:
                concat_dict[concat_char] += concat_num - 1
            else:
                concat_dict[concat_char] = concat_num - 1
    change = None
    mx = 0
    for key in concat_dict.keys():
        if mx < concat_dict[key]:
            mx = concat_dict[key]
            change = key
    ln_change = len(change)
    ln_ans = len(ans)
    replace_from_str = change + replace_from_str
    replace_nums.insert(0, ln_change)
    replace_to_str = chars_add[i] + replace_to_str
    #print(i, change, chars_add[i], ln_change, mx)
    new_ans = ''
    j = 0
    while True:
        if j >= ln_ans - ln_change:
            if j >= ln_ans:
                break
            new_ans += ans[j]
            j += 1
        elif ans[j:j + ln_change] == change:
            new_ans += chars_add[i]
            j += ln_change
        else:
            new_ans += ans[j]
            j += 1
    ans = deepcopy(new_ans)
    set_chars.append(chars_add[i])

print('// len', len(ans))
print('const string replace_from_str = "', replace_from_str, '";', sep='')
print('const string replace_to_str = "', replace_to_str, '";', sep='')
print('const int replace_nums[ln_repair] = {', sep='', end='')
for i in replace_nums[:len(replace_nums) - 1]:
    print(i, ', ', sep='', end='')
print(replace_nums[len(replace_nums) - 1], '};', sep='')

if input('sure?: ') != 'yes':
    exit()

with open(sys.argv[1], 'w') as f:
    flag = False
    for i in range(len(ans)):
        if i % 300 == 0:
            flag = False
            f.write('"')
        f.write(ans[i])
        if i % 300 == 299:
            flag = True
            f.write('"\n')
    if not flag:
        f.write('"')
    f.write(';\n')

print('done')