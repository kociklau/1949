import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

fpath = os.path.dirname(os.path.abspath(__file__)) + "\\..\\benchmark\\benchmark\\experiment_results.dat"
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
    print("threshold = {0}".format(s))
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


plt.plot(index_array, tvalue_array)
#plt.plot(index_array, threshold_array)
plt.title('tvalue vs access index')
plt.ylabel('tvalues (ms)')
plt.xlabel('array index (integer)')
plt.show()
