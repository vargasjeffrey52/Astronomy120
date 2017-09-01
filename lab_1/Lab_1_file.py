#Lab 1 Photon counting & the Statistics of Light Fall 2015

import numpy as np
import matplotlib.pyplot as plt
import doctest
import math



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

def graph(file):
    plt.figure(1)
    graph_1 = plt.plot(usefull_data(file))
    x_label = plt.xlabel('Event number')
    y_label = plt.ylabel('Clock tick')

    plt.figure(2)
    graph_2 = plt.plot(time_interval(file))
    x_label = plt.xlabel('Event number')
    y_label = plt.ylabel('Interval [clock ticks]')

    plt.figure(3)
    graph_3 = plt.plot(time_interval(file),',')
    x_label = plt.xlabel('Event number')
    y_label = plt.ylabel('Interval [clock ticks]')
    return plt.show(graph_1), plt.show(graph_2),plt.show(graph_3)
#________________every thing above is working may need to be clean up to make code more nice________________________________________________________


# some statistics



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



def graph_ave(file,nstep):
    
    """this graph is mean interval vs start index. ie. it is averaging chunks of  Nstep events and then it plots the the point.
    ex) nstep = 100 => average of 0-100 events then average of 101-200
    the average from 0-100 represents the first point...
    """
    
    marr = average(file,nstep)
    xaxis = np.linspace(0,10000,np.size(marr))
    ave_graph = plt.plot(xaxis, marr, 'o')
    plt.xlabel('Start index')
    plt.ylabel('Mean Interval [clock ticks]')
    print (np.size(marr))
    return plt.show(ave_graph)


def ave_lim(file):
# worked: it produces figure:5     
    dt = time_interval(file)
    marr = np.array([])
    i = np.arange(dt.size)
    nstep = 1000
    for j in i[0::nstep]:
        m = np.mean(dt[0:j+nstep])
        marr = np.append(marr,m)
    return marr

def ave_lim_graph(file):
    marr = ave_lim(file)
    xaxis = np.linspace(0,10000,np.size(marr))
    ave_graph = plt.plot(xaxis,marr, 'o')
    plt.xlabel('Number of intervals averaged')
    plt.ylabel('Mean Interval [clock ticks]')
    return marr, plt.show(ave_graph)

"""def stand_dev(file):
#worked to get the overall standard dev
    marr = average(file)
    mu = np.sum(marr)/np.float(marr.size)
    dev = np.sqrt(np.sum((marr-mu)**2)/(np.float(marr.size)-1.))
    return dev"""
def stand_dev(marr):
#worked to get the overall standard dev
    mu = np.sum(marr)/np.float(marr.size)
    dev = np.sqrt(np.sum((marr-mu)**2)/(np.float(marr.size)-1.))
    return dev

"""def stand_dev_graph(file):
    
    ave = average(file,10)
    i = np.arange(ave.size)
    nstep = 10
    devarr = np.array([])
    for j in i[0::nstep]:
        m = stand_dev(ave[0:j+nstep])
        devarr= np.append(devarr, m)
    print (np.size(devarr), np.size(ave))
    print (devarr)
    plot = plt.plot(devarr, 'o')
    plt.xlabel('number of events averaged')
    plt.ylabel('standard deviation of the mean [ticks]')
    return devarr, plt.show()"""


"""def stand_dev_graph(file):
    end = 400
    start = 100
    devarr = np.array([])
    nstep = 10
    
    avearr = np.array([])
    
    while start <= end:
        
        ave = average(file,start)
        print (ave)
        avearr = np.append(avearr,ave) 
        start = start + 10
        i = np.arange(avearr.size)
        for j in i[0::nstep]:
            m = stand_dev(avearr[0:j+nstep])
            devarr= np.append(devarr, m)
    print (np.size(devarr), np.size(ave))
    print (devarr)
    plot = plt.plot(devarr, 'o')
    plt.xlabel('number of events averaged')
    plt.ylabel('standard deviation of the mean [ticks]')
    return devarr, plt.show()"""
def stand_dev_graph(file):
    
    #avesize = np.size(average(file,400))
    
    chunk = 10 # this is the starting value for the chunk space. NOTE: the cunks will increase by 10
    x = np.array([]) #this is an empty array that will later stor the information of 1/sqrt(chunk)
    devarr = np.array([])# this empty array will store the standard deviation for diffrent number of chunks as chunks increase
    
    while chunk <= 400: # the while loop will run until it reaches 1000
        ave = average(file,chunk) # this is a function that will take in the file and the chunk number and average it out.
        #NOTE chunk will increase by 10 so average will change
        
        m = stand_dev(ave)#this will take the standard deviation of the average and later store that value onto devarr
        #NOTE: standard deviation is different becaue average changes according to the number of chuncks
        
        devarr= np.append(devarr, m)# stores diffrents standard deviation
        x =np.append(x, 1/np.sqrt(chunk))#stores the 1/sqrt(chunk) so that we can plot it later for figure 8
        #note x will be an array whose values differ because chunk is changing. Also, this will represent the solid line in fig8 
        chunk = chunk + 10 #increase chunk by 10
        
    xaxis = np.linspace(0,400,np.size(devarr))
    graph1 = plt.plot(xaxis,devarr, 'o')
    plt.xlabel('number of events averaged')
    plt.ylabel('standard deviation of the mean [ticks]')    
        
    plt.figure(2)
    s = np.std(time_interval(file))
    s_over_sqrtN = s*x
    graph2 = plt.plot(x,devarr, 'o',x, s_over_sqrtN, 'g')
    plt.xlabel('1/sqrt(N)')
    plt.ylabel('standard deviation of the mean [ticks]')
    #plt.ylabel(r'$\frac(1){sqrt(N)}$   
    
    return devarr, plt.show(graph1),plt.show(graph2)
def histogram(file):
    dt = time_interval(file)
    n = 100000
    #define the lower and upper bin edges and bin width
    bw = (dt.max()-dt.min())/(n-1.)
    bin1 = dt.min() + bw *np.arange(n)
    #define the array to hold the occurrence count
    bincount = np.array([])
    #loop trhough the bins
    for bin in bin1:
        count = np.where((dt >=bin) & (dt< bin +bw))[0].size
        bincount = np.append(bincount,count)
        binc = bin1 + 0.5*bw
    return bincount,binc

def histo_graph(file):
    bincount,binc = histogram(file)[0],histogram(file)[1]
    #compute bin centers for ploting
    #binc = bin1 + 0.5*bw
    plt.figure()
    plt.xlabel('Interval [ticks]')
    plt.ylabel('Frequency [Counts]')
    grpah1 = plt.plot(binc[0:50],bincount[0:50],drawstyle = 'steps-mid')
    
    return plt.show()


"""def histogram_2(file):
    tau = np.mean(histogram(file)[1])
    binc = histogram(file)[1]
    bincount = histogram(file)[0]
    dt = time_interval(file)
    exp = math.exp((-dt)/tau)
    p_dt = (1/tau) * exp
    plt.figure()
    plt.plot(binc[0:50],p_dt[0:50], 'g')
    return plt.show()"""


def histo_log(file):
    dt = time_interval(file)
    cut = np.where(dt>3000)
    dt_cut = dt[cut]
    n = 500
    #define the lower and upper bin edges and bin width
    bw = (dt_cut.max()-dt_cut.min())/(n-1.)
    bin1 = dt_cut.min() + bw *np.arange(n)
    #define the array to hold the occurrence count
    bincount = np.array([])
    #loop trhough the bins
    for bin in bin1:
        count = np.where((dt_cut >=bin) & (dt_cut< bin +bw))[0].size
        bincount = np.append(bincount,count)
        binc = bin1 + 0.5*bw

    print (np.size(bincount), np.size(binc))
    # testing the eponential line graph
    plt.figure(1)
    
    tau=np.mean(dt_cut.astype(np.float))
    print (tau)
    p=(20.*(10**9.)/(14.*tau))*np.exp(-binc/tau)#ask gerry whi he chose 18/15
    plt.plot(binc, p)
    
    plt.xlabel('Interval [ticks]')
    plt.ylabel('Frequency [Counts]')
    grpah1 = plt.plot(binc,bincount,drawstyle = 'steps-mid')

    plt.figure(2)
    plt.xlabel('Interval [Tick]')
    plt.ylabel('Frequency')
    plt.plot(binc[0:200], np.log10(bincount[0:200]), drawstyle='steps-mid')
    plt.plot(binc[0:200], np.log10(p[0:200]))
    

    return plt.show()

    
