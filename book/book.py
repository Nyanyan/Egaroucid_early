import subprocess
from tqdm import trange
from copy import deepcopy

hw = 8
hw2 = 64
board_index_num = 38
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

all_chars = ['!', '#', '$', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

print(''.join(all_chars[:hw2]))

def to_str_record(record):
    res = ''
    for coord in record:
        #res += chr(coord % hw + ord('A'))
        #res += str(coord // hw + 1)
        res += all_chars[coord]
    return res

records = []

with open('early_stage.txt', 'r') as f:
    line_count = int(f.readline())
    print(line_count)
    for _ in trange(line_count):
        record = f.readline()
        records.append(record)

raw_val = []
with open('val.txt', 'r') as f:
    raw_val = list(f.read().splitlines())

record_val = [[] for _ in range(100)]
mx_ln = 0
val_idx = 0
for i in trange(len(records)):
    str_record = records[i]
    record = []
    idx = 0
    while idx + 1 < len(str_record):
        y = int(str_record[idx + 1]) - 1
        x = ord(str_record[idx]) - ord('A')
        record.append(y * hw + x)
        idx += 2
    while True:
        if ':' in raw_val[val_idx]:
            break
        val_idx += 1
    next_move_str = raw_val[val_idx][1:3]
    y = int(next_move_str[1]) - 1
    x = ord(next_move_str[0]) - ord('A')
    next_move = y * hw + x
    record.append(next_move)
    score = float(raw_val[val_idx].split()[0][3:])
    record_val[len(record)].append([record, score])
    mx_ln = max(mx_ln, len(record))
    val_idx += 1

best_route = []
for ln in range(2, mx_ln + 1):
    best_score = 1000
    f_best_route = [i for i in best_route]
    for record, score in record_val[ln]:
        if record[:len(f_best_route)] == f_best_route and abs(score) < best_score:
            best_score = abs(score)
            best_route = [i for i in record]

print('best_route', to_str_record(best_route))

book = {}
for i in range(1, mx_ln):
    book[to_str_record(best_route[:i])] = best_route[i]

for record, score in record_val[3]:
    book[to_str_record(record[:-1])] = record[-1]

for ln in range(4, mx_ln + 1):
    for record, score in record_val[ln]:
        str_record = to_str_record(record[:-3])
        if str_record in book.keys():
            if book[str_record] == record[-3]:
                book[to_str_record(record[:-1])] = record[-1]
    print(ln, len(book))


with open('param/book.txt', 'w') as f:
    for record in book.keys():
        f.write(record + ' ' + all_chars[book[record]] + ' ')

