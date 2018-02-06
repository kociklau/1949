import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

fpath = os.path.dirname(os.path.abspath(__file__)) + "/../benchmark/experiment_results.dat"
index_array= []
tvalue_array = []
threshold_array=[]
with open(fpath, "r") as f:
    #Skip header row of "hits|misses"
    f.readline()
    #Get number of hits and misses
    hits,misses = f.readline().split("|")
    print("hits = {0}\nmisses = {1}".format(hits,misses))
    #Get threshold for hits/misses
    threshold = f.readline().split("=")[1]
    print("threshold = {0}".format(threshold))
    #Get array size
    N = f.readline().split("=")[1]
    print("array size= {0}".format(N))
    #Get stride
    s = f.readline().split("=")[1]
    print("stride = {0}".format(s))
    #Get number of iterations
    numIter = f.readline().split("=")[1]
    print("numIterations = {0}".format(numIter))
    #Skip the header row of "arrayIndex|tvalue"
    line = f.readline()
    #Process file until "end" reached
    while (True):
        line = f.readline()
        if ("end" in line):
            break
        index, tvalue = line.split("|")
        index_array.append(int(index))
        tvalue_array.append(int(tvalue))
        threshold_array.append(int(threshold))
        #print("index_array[{0}] = {1}".format(index,tvalue))

fig = plt.figure()

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

ax.text(0.65, 0.70, 'N = {0}stride = {1}threshold = {2}numIterations\
= {3}hits = {4} misses = {5}'.format(N,s,threshold,numIter,hits,misses),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='green', fontsize=14)

ax.plot(index_array, tvalue_array)
#plt.plot(index_array, threshold_array)
plt.title('tvalue vs access index')
plt.ylabel('tvalues (clock cycles)')
plt.xlabel('array index (integer elements)')
plt.show()
