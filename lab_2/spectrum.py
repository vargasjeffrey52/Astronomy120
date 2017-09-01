import sys
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 20})

def main(filename, bckgrndname ,end, digits):
    data=readlots(filename, end, digits)
    background=readlots(bckgrndname, end, digits)
    avgdata=pixelaverage(data)
    #print avgdata
    avgbackground=pixelaverage(background)
    #print avgbackground
    goodness=avgdata[1]-avgbackground[1]
    goodness=[avgdata[0],goodness]
    plt.figure(1)
    s=pixelgraph(goodness)
    plt.figure(2)
    t=pixelgraph(avgbackground)
    plt.figure(3)
    u=pixelgraph(avgdata)
    yesorno=raw_input('Find centroid (y or n)? ')
    if yesorno=="n":
        print 'ok'
    elif yesorno=="y":
        center=int(raw_input('center pixel? '))
        delta=int(raw_input('how far for each side?  '))
        mercurywavelen=np.array([435.8328, 576.9598, 579.0663])
        mercurycenter=np.array([[482,483], [1307,1308], [1329,1330]])
        mercurydelta=np.array([5, 5, 5])
        neonwavelen=np.array([585.24879, 587.28275, 594.48342, 614.30626, 616.35939, 621.72812, 626.64950])
        neoncenter=np.array([[1359,1360],[1378, 1379], [1418,1419], [1548, 1549], [1560, 1462] ,[1597, 1598], [1630, 1631]])
        neondelta=np.array([4, 4, 4, 4, 4, 4, 4])
        answer=raw_input('mercury or neon? (mercury or neon?) ')
        plt.figure(4)
        if answer=='mercury':
            centroids=centroid(goodness, mercurycenter, mercurydelta)
            plt.plot(mercurywavelen, centroids)
        elif answer=='neon':
            centroids=centroid(goodness, neoncenter, neondelta)
            plt.plot(neonwavelen, centroids)
        else:
            print 'plz wait while I start over since you told me BS'
            main(filename, bckgrndname, end, digits)
            #break
        #centroid(goodness, center, delta)
    else :
        print 'dumb answer'
    plt.show()
def centroid(goodness, center, delta):
    "Finds centroids of spikes given the center of the spike and how far to go on each side."
    pixels=goodness[0]
    intensity=goodness[1]
    end=len(delta)
    i=0
    centroids=np.array([])
    while i<end:
        loopcenters=center[i]
        spikepixels0=pixels[loopcenters[0]-delta[i]:loopcenters[0]+delta[i]]
        spikeintensity0=intensity[loopcenters[0]-delta[i]:loopcenters[0]+delta[i]]
        spikepixels1=pixels[loopcenters[1]-delta[i]:loopcenters[1]+delta[i]]
        spikeintensity1=pixels[loopcenters[1]-delta[i]:loopcenters[1]+delta[i]]
        numerator0=sum(spikepixels0*spikeintensity0)
        denominator0=sum(spikeintensity0)
        centroid0=numerator0/denominator0
        numerator1=sum(spikepixels1*spikeintensity1)
        denominator1=sum(spikeintensity1)
        centroid1=numerator1/denominator1
        centroid=np.mean(np.array([centroid0, centroid1]))
        print centroid
        centroids=np.append(centroids, centroid)
        i=i+1
    return centroids
def read(filename):
    raw=np.loadtxt(filename, comments='>',skiprows=16, dtype=np.int32)
    pixel=raw[:,0]
    intensity=raw[:,1]
    return [pixel, intensity]
def pixelgraph(data):
    #data=read(filename)
    #print data
    pixel=data[0]
    intensity=data[1]
    t=plt.plot(pixel,intensity)
    plt.xlabel('Pixel Number')
    plt.ylabel('Number of photons')
    #plt.show()
    return t
def readlots(filename, end, digits):
    "Reads more than one file at a time. requires the last characters in the file to be numbers that enumerates the data. Sll numbered files must have the same number of digits with 0 in front if neccesary"
    i=0
    suffix='0'*digits#First file must be the 0th file
    pixels=np.array([])#These arrays will hold the x and y values of the file, in this lab they are pixels and intensities.
    intensities=np.array([])
    while i<end:#slowly loop through all the fields
        rawelement=np.loadtxt(filename+suffix+'.txt', comments='>', skiprows=16, dtype=np.int32)#read a file
        pixels=np.append(pixels, rawelement[:,0])
        intensities=np.append(intensities,rawelement[:,1])
        suffixint=int(suffix)#turn the suffix into an integer to increase by 1
        i=suffixint+1#increase the index in the while loop. Should match suffix
        #print i
        almostnewsuffix=str(suffixint+1)#increase by 1 and turn into string
        zerosneeded=digits-len(almostnewsuffix)
        suffix='0'*zerosneeded+almostnewsuffix
        #print suffix
    return [pixels,intensities]
    
def pixelaverage(data):
    "Finds the average value for each pixel. Returns array of pixel numbers and average intesnsity for each"
    #data=readlots(filename, end, digits)#get all the data
    pixels=data[0]#seperate into pixels
    #print pixels
    intensities=data[1]#and intensiites
    #avgdata=np.array([[],[]])#array holds avg data
    i=0
    avgpixels=np.array([])
    avgdataintensities=np.array([])
    while i<=2047:
        places=np.where(pixels==i)[0]#find all the places with the same pixel value
        intensityavgforpixel=np.mean(intensities[places])
        avgpixels=np.append(avgpixels, i)
        #print avgdata[0]
        avgdataintensities=np.append(avgdataintensities, intensityavgforpixel)
        #print avgdata
        i=i+1
    avgdata=np.array([avgpixels, avgdataintensities])
    return avgdata
