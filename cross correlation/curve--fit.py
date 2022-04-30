




"""
import random as r
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
def test(x,m,c):
    return m*x+c
x=[i for i in range(90)]
y=[i+50*r.randint(10,20) for i in x]
sig_fft=fftpack.fft(y)
freq=fftpack.fftfreq(len(y),d=10)
print(freq)
plt.plot(x,y)
plt.show()
"""