from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import Image

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

from IPython.display import clear_output

import matplotlib.pyplot as plt
import os 
from PIL import Image

import multiprocessing as mp

nobackup = '/nobackup/b700938/'

while 1==1:
    poolsize = int(input('macplotlib: Please define poolsize to be used for parallelised functions'))
    if isinstance(poolsize, int) == True:
        print('poolsize set to %d'%poolsize)
        break
    else:
        continue

def gif_plot(func, loop_data, duration, gif_name):
    '''When supplied with a plotting function, will reproduce that plot for a range of different inputs and
    combine all resulting plots in a looping gif. Loop data should be an array of indices at which the data plotted
    by func should indexed'''
    
    frames = []
            
    for i in range(len(loop_data)):
        
        func(i)
        plt.savefig(nobackup+'frame%d.png'%i, dpi=300);
        plt.close()
        frames.append(Image.open(nobackup+'frame%d.png'%i))
        os.remove('frame%d.png'%i)
        
    frames[0].save(gif_name, format='GIF',
                append_images=frames[1:], save_all=True, duration=duration,
                loop=0)
    
    return