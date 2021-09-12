chars = sorted(list(set(range(35, 123)) - set([92])))
print(chars)
num_s = 123
num_e = 126
ln = len(chars)

vals = []
with open('param/param.txt', 'r') as f:
    while True:
        try:
            vals.append(float(f.readline()))
        except:
            break

pattern_all = len(vals)

vals_variation = sorted(list(set(vals)))
around = [0 for _ in range(1000)]
err = 0.0
step = 0.0001
while len(around) > ln:
    err += step
    around = []
    i = 0
    while i < len(vals_variation):
        avg = vals_variation[i]
        cnt = 1
        for j in range(i + 1, len(vals_variation)):
            if vals_variation[j] - vals_variation[i] > err:
                break
            avg += vals_variation[j]
            cnt += 1
        around.append(avg / cnt)
        i += cnt
around.sort()
print(len(around), ln, err)
print(around)

res_arr = []
for i in range(pattern_all):
    val = vals[i]
    tmp = -1
    min_err = 1000.0
    for j, k in enumerate(around):
        if abs(val - k) < min_err:
            min_err = abs(val - k)
            tmp = j
    res_arr.append(chr(chars[tmp]))

super_compress = []
max_same = 0
for i in range(len(res_arr)):
    if len(super_compress):
        if ord(super_compress[-1]) >= num_s:
            if ord(super_compress[-1]) < num_e and super_compress[-2] == res_arr[i]:
                max_same = max(max_same, ord(super_compress[-1]) + 1 - num_s)
                super_compress[-1] = chr(ord(super_compress[-1]) + 1)
            else:
                super_compress.append(res_arr[i])
        else:
            if super_compress[-1] == res_arr[i]:
                super_compress.append(chr(num_s))
            else:
                super_compress.append(res_arr[i])
    else:
        super_compress.append(res_arr[i])

print('max_same', max_same)

with open('param/param_compressed.txt', 'w') as f:
    flag = False
    for i in range(len(super_compress)):
        if i % 300 == 0:
            flag = False
            f.write('"')
        f.write(super_compress[i])
        if i % 300 == 299:
            flag = True
            f.write('"\n')
    if not flag:
        f.write('"')
    f.write(';\n')
