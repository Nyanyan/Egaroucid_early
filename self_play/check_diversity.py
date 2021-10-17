import matplotlib.pyplot as plt


def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n


num = int(input())
with open('records/' + digit(num, 7) + '.txt', 'r') as f:
    records = f.readlines()
for i in range(len(records)):
    records[i] = records[i].split()[0]

len_records = len(records)

y = []

for i in range(2, 60):
    dct = {}
    for record in records:
        if record[:i] in dct:
            dct[record[:i * 2]] += 1
        else:
            dct[record[:i * 2]] = 0
    y.append(len(dct) / len_records)

x = range(2, 60)

plt.plot(x, y)
plt.plot(x, [1.0 for _ in range(len(x))])
plt.plot(x, [0.9 for _ in range(len(x))])
plt.show()

