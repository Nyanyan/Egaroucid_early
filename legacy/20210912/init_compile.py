import subprocess


cmd = 'g++ ai_init.cpp -O3 -march=native -fexcess-precision=fast -funroll-loops -flto -mtune=native -o ai_init.out'
o = subprocess.run(cmd.split(), encoding='utf-8', stderr=subprocess.STDOUT, timeout=None)
print('------------------compiled------------------')