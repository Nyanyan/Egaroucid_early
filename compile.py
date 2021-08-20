import subprocess


cmd = 'g++ ai.cpp -O3 -march=native -fexcess-precision=fast -funroll-loops -flto -mtune=native -o ai.out'
o = subprocess.run(cmd.split(), encoding='utf-8', stderr=subprocess.STDOUT, timeout=None)
print('------------------compiled------------------')