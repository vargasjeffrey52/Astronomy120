#Lab 1 Photon counting & the Statistics of Light

import numpy as np
import matplotlib.pyplot as plt

def read_data(file):
    """ opens file and turns the data into 32-bit dat

    """
    x = np.loadtxt(file, delimiter=',', dtype='int32')
    return x

def usefull_data(file):
    """ removes the un-usefull data (channel number = 2)"""
    
    clock_ticks = read_data(file)
    return clock_ticks[:,1]

def time_interval(file):
    time = usefull_data(file)
    dt = time[1:] - time[0:-1]
    return dt

def average(file,nstep):
    "this function will show the average average number of clock ticks for every 1000 clicks and then plot a point utill all the events are recorded (i.e 10k events)" 
#remember to ask about how to change the values from the y axis to show in scientific notation
    
    dt = time_interval(file)
    marr = np.array([])
    i = np.arange(dt.size)
    for j in i[0::nstep]:
        m = np.mean(dt[j:j+nstep])
        marr = np.append(marr,m)
    
    return marr

def stand_dev(marr):
#worked to get the overall standard dev
    mu = np.sum(marr)/np.float(marr.size)
    dev = np.sqrt(np.sum((marr-mu)**2)/(np.float(marr.size)-1.))
    return dev

def stand_dev_graph(file):
    
    #avesize = np.size(average(file,400))
    
    chunk = 10 # this is the starting value for the chunk space. NOTE: the cunks will increase by 10
    x = np.array([]) #this is an empty array that will later stor the information of 1/sqrt(chunk)
    devarr = np.array([])# this empty array will store the standard deviation for diffrent number of chunks as chunks increase
    
    while chunk <= 1000: # the while loop will run until it reaches 1000
        ave = average(file,chunk) # this is a function that will take in the file and the chunk number and average it out.
        #NOTE chunk will increase by 10 so average will change
        
        m = stand_dev(ave)#this will take the standard deviation of the average and later store that value onto devarr
        #NOTE: standard deviation is different becaue average changes according to the number of chuncks
        
        devarr= np.append(devarr, m)# stores diffrents standard deviation
        x =np.append(x, 1/np.sqrt(chunk))#stores the 1/sqrt(chunk) so that we can plot it later for figure 8
        #note x will be an array whose values differ because chunk is changing. Also, this will represent the solid line in fig8 
        chunk = chunk + 10 #increase chunk by 10
    print (np.size(devarr), np.size(ave))
    print (devarr)
    graph1 = plt.plot(devarr, 'o')
    plt.xlabel('number of events averaged')
    plt.ylabel('standard deviation of the mean [ticks]')    
        
    plt.figure(2)
    s = np.std(time_interval(file))
    s_over_sqrtN = s*x
    graph2 = plt.plot(x,devarr, 'o',x, s_over_sqrtN, 'g')
    #plt.xlabel(1/N**.5)
    #plt.ylabel('standard deviation of the mean [ticks]')
   
    
    return devarr, plt.show(graph1),plt.show(graph2)

