import numpy as np
import matplotlib.pyplot as plt

def show(img):
    npimg = img.numpy()
    plt.figure(figsize=(16,6))
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])