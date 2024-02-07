#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 08:43:00 2020

@author: andreas
"""

import matplotlib as mpl
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
# change the following to %matplotlib notebook for interactive plotting
# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

#For multicolor text in plt title
import numpy as np
from numpy.linalg import eig, inv, svd
from math import atan2
import colorcet
import pandas as pd
import pims
import trackpy as tp
import glob
import os
import re # Regex tools
import sys
from scipy import stats,spatial,interpolate
from scipy.optimize import curve_fit
from scipy import ndimage as ndi
import numpy as np
from lmfit import Model, Parameters
import math
import skimage as ski
import skimage.filters
from sklearn.cluster import DBSCAN
import pyclesperanto_prototype as cle
import ffmpeg
plt.rcParams['animation.ffmpeg_path']='/Users/bioc1463/Desktop/ffmpeg'

### Silence warnings from np.sqrt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
## Silence annoying pandas warning on copys
pd.set_option('mode.chained_assignment', None)


sourceDir = '/Volumes/BackUp_1/To_do/Tracking/SprE/'

verbose = False
tp.quiet() # Turn off progress reports for best performance
forceAnalysis = True
forceSaveCompiledData = True
process3Donly = False

# Data parameters
getFPSandExposureFromFname = True #Tries to guess framespeed number from filename (i.e. 1on3off = )
defaultExposureTime = 50 #ms exposure time used if unable to find in filename (found by regex searching for \d+ms where \d+ is any number of digits)
#framesPerSec = 12.5 #This number is used if dynamicFramespeed is False

# Tracking parameters
trackingMemory = 5
TruncateTracks = 25
useClusterFiltering = False
useCellOutlineFiltering = True #This uses the celloutline file to segment and filter
lowerCanny = 1.2 #Settings for Canny thresholding
upperCanny = 5 #Settings for Canny thresholding
minTrackLen = 25 #Tracks below this length (in frames) are removed

#Movie parameters
saveMovies=True
forceMovie = True

#Settings for calculating MSDs
maxLagtime = 24
micronPerPixel = 0.1169
saveMSDData= True
forceSaveiMSDData = True #Saves a .csv file with the iMSD matrix generated with trackpy

#Mapping corrections 
#(3D channels are annoyingly not aligned so it is required to correct. Disable for 2D analysis!)
adjustChannelMapping = False
xCorrection=10
yCorrection=0

############# Initialize a bunch of functions needed to run the code ###########

def __fit_ellipse(x, y):
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S, C = np.dot(D.T, D), np.zeros([6, 6])
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
    U, s, V = svd(np.dot(inv(S), C))
    a = U[:, 0]
    return a

def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])

def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * (
        (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    down2 = (b * b - a * c) * (
        (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return atan2(2 * b, (a - c)) / 2

def fit_ellipse(x, y):
    """@brief fit an ellipse to supplied data points: the 5 params
        returned are:
        M - major axis length
        m - minor axis length
        cx - ellipse centre (x coord.)
        cy - ellipse centre (y coord.)
        phi - rotation angle of ellipse bounding box
        
        More info here: https://github.com/ndvanforeest/fit_ellipse
    @param x first coordinate of points to fit (array)
    @param y second coord. of points to fit (array)
    """
    a = __fit_ellipse(x, y)
    centre = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    M, m = ellipse_axis_length(a)
    # assert that the major axix M > minor axis m
    if m > M:
        M, m = m, M
    # ensure the angle is betwen 0 and 2*pi
    phi -= 2 * np.pi * int(phi / (2 * np.pi))
    return [M, m, centre[0], centre[1], phi]

def clusterFiltering(tsData):
    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=5, min_samples=50).fit(tsData[['x','y']].values)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    tsData['labels']=labels
    tsData=tsData.loc[tsData.labels > -1]
    
    goodClusters=[]
    #Loop through clusters and discard if convex hull area is too small
    for cluster in tsData.labels.unique():
        subData=tsData.loc[tsData.labels==cluster]
        db = DBSCAN(eps=framesPerSec/5, min_samples=int(len(subData.x)/100)).fit(subData[['x','y']].values)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
        #Suppress the annoying SettingWithCopyError
        pd.options.mode.chained_assignment = None
        subData.loc[:,"labels"]=labels
        subData=subData.loc[subData.labels > -1]
        
        hull=spatial.ConvexHull(subData[['x','y']])
        convexHullArea=hull.volume
        if convexHullArea > 100:
            goodClusters.append(subData)
            # plt.figure()
            # plt.scatter(subData.x,subData.y,c=subData.labels,s=1)
            # plt.title(convexHullArea)
            # plt.scatter(subData.x,subData.y,c=subData.labels,s=1)
            ew, eh, eCX, eCY, eRot=fit_ellipse(subData.x.to_numpy(), subData.y.to_numpy())
            print(ew, eh, eCX, eCY, eRot)
    tsData = pd.concat(goodClusters)
    # plt.scatter(tsData.x,tsData.y)
    return tsData

def cellOutlineFiltering(tsData, outlineFname, lowerCanny, upperCanny, showMask=False, cHullLim=0.7, useCHullAlways=True):
    #read images, normalise, and calculate params
    stack=ski.io.imread(outlineFname)
    stack=(stack/stack.max()) * 255
    stack=ski.filters.median(stack)
    IM_Mean_orig= np.mean(stack, axis=0)
    # plt.imshow(IM_Mean_orig)
    # plt.scatter(tsData.x,tsData.y, s=0.1)
    # # t = tp.link(tsData, search_range=120/framesPerSec, adaptive_stop=0.56, adaptive_step=0.99, memory=trackingMemory)
    # # t = tp.filter_stubs(t, 20)
    # plt.scatter(t.x,t.y, s=0.1)                            
    Adjustment = IM_Mean_orig/IM_Mean_orig.max()
    stack = stack * Adjustment
    lowerCanny = 1.2
    upperCanny = 5
    stack = [ski.feature.canny(im, lowerCanny, upperCanny) for im in stack]
   
    #Calculate the z projection (aka mean)
    IM_Mean= np.mean(stack, axis=0)
    # plt.imshow(IM_Mean)
    
    #Adaptive threshold with Otsu's method
    IM_Mean=IM_Mean > ski.filters.threshold_otsu(IM_Mean)
    # plt.imshow(IM_Mean)
    #Dilate to close incomplete edges
    IM_Mean=ndi.binary_dilation(IM_Mean)
    # IM_Mean=ndi.binary_dilation(IM_Mean)
    # plt.imshow(IM_Mean)
    #Fill the binary cell outlines
    IM_Mean=ndi.binary_fill_holes(IM_Mean)
    # plt.imshow(IM_Mean)
    #Erode to get rid of potential dilated dirt
    IM_Mean=ndi.binary_erosion(IM_Mean)
    IM_Mean=ndi.binary_erosion(IM_Mean)
    IM_Mean=ndi.binary_erosion(IM_Mean)
    IM_Mean=ndi.binary_dilation(IM_Mean)
    IM_Mean=ndi.binary_dilation(IM_Mean)

    # plt.imshow(IM_Mean)
    #define mask to be used in contour iteration/refinement
    mask=IM_Mean

    #Find and iterate through all contours in the mask to dump contours with 
    #wrong shape descriptors. Cells should appear as elongated ellipses so
    #aspect ratio of width/height and area should serve well as simple criteria
    contoursAll=ski.measure.find_contours(mask)
    finalMask = np.zeros(mask.shape) #Create an empty image to draw correct contours onto
    removedMask = np.zeros(mask.shape)
    aspectRatioList = []
    cleanedDataList = []
    for contour in contoursAll:

        #Fit ellipse to contour and calculate parameters
        x, y = contour[:,1], contour[:,0]    
        if len(x) < 8 or len(y) < 9:
            # print("contour too small to analyse, skipping!")
            continue
        ew, eh, eCX, eCY, eRot=fit_ellipse(x, y)
        area=np.pi*ew*eh
        # print(area)
        if ew < eh:
            aspectRatio = eh/ew
        else:
            aspectRatio = ew/eh
            
        if (aspectRatio < 2.5 and area < 250) or area < 100:
            aspectRatioList.append(aspectRatio)
            #This is likely to be a small bright spot which we need to get rid of to
            #increase performance of thresholding
            # print("Bright spot found")
            # spotNoiseFound=1
            #Don't add this to pruned, remove it from the input image and recalculate binary mask
            # IM_Mean[y.min().astype(int):y.max().astype(int),x.min().astype(int):x.max().astype(int)] = IM_Mean[y.min().astype(int):y.max().astype(int),x.min().astype(int):x.max().astype(int)]*np.invert(mask[y.min().astype(int):y.max().astype(int),x.min().astype(int):x.max().astype(int)])
            if aspectRatio > 2.5:
                rr, cc = ski.draw.polygon(y,x)
                removedMask[rr,cc]=1
            pass
        else:
            hull=spatial.ConvexHull(np.array(list(zip(x,y))))
            # convexHullPerimeter=hull.area
            convexHullArea=hull.volume
            polygonArea=0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
            
            # print("This contours is fine")
            # print("Area=",area)
            # print("aspectRatio=",aspectRatio)
            # print(polygonArea/convexHullArea)
            # eParams=[]
            # plt.figure(0)
            # plt.plot(x, y, color='blue', label='points')
            # plt.legend()
            # for i in range(0,len(x)-100):
            #     plt.figure()
            #     plt.scatter(x[i:i+100],y[i:i+100])
                
                
            #     ew, eh, eCX, eCY, eRot=fit_ellipse(x[i:i+100],y[i:i+100])
                
            #     if not (np.isnan(ew) or np.isnan(eh)) and ew+eh < 20:
                    
            #         eParams.append([ew, eh, eCX, eCY, eRot])
            #         ## plot the elliptical fit (troubleshooting)
            #         center, axes = (eCX, eCY), (ew, eh)
                            
            #         # generate points on the fitted ellipse
            #         R=np.linspace(0,6,100)
            #         a, b = axes
            #         xx = center[0] + ew * np.cos(R) * np.cos(eRot) - eh * np.sin(R) * np.sin(eRot)
            #         yy = center[1] + ew * np.cos(R) * np.sin(eRot) + eh * np.sin(R) * np.cos(eRot)
            #         plt.plot(xx, yy, '+', color='red', label='fitted ellipse', linewidth=2.0)
            #         plt.Rectangle((eCX, eCY),ew,eh, fc='blue',ec="red")
                    
                
            # plt.axes().set_aspect('equal', 'datalim')
            
            if polygonArea/convexHullArea < cHullLim or useCHullAlways:
                #Use convex hull instead since the canny edges likely didn't close well.
                # Calculate the interpolated x and y values to the same len as the 
                # original coordinates
                xHull=x[hull.vertices]
                yHull=y[hull.vertices]
                t = np.arange(len(xHull))
                ti = np.linspace(0, t.max(), len(x))
                xi = interpolate.interp1d(t, xHull, kind='linear')(ti)
                yi = interpolate.interp1d(t, yHull, kind='linear')(ti)
    
                # # Plot the interpolation result
                # fig, ax = plt.subplots()
                # ax.set_aspect('equal')
                # ax.scatter(xi, yi)
                # ax.scatter(xHull, yHull)
                # ax.scatter(x, y)
                # ax.margins(0.05)
                # plt.show()
                x=xi
                y=yi
            
            #Use the boundary to draw all pixels within as 1 onto the finalMask
            rr, cc = ski.draw.polygon(y,x)
            finalMask[rr,cc]=1
            
            goodPositions=list(zip(rr,cc))
            
            # #Have a look at the area around current mask:
            # pad=10
            # plt.imshow(finalMask[min(rr)-pad:max(rr)+pad,min(cc)-pad:max(cc)+pad]) #mask
            # plt.scatter(x-min(x)+pad,y-min(y)+pad) #xy outline of mask
            
            # # subData=tsData[(tsData.x > min(x)-pad) & (tsData.x < max(x)+pad) & (tsData.y > min(y)-pad) & (tsData.y < max(y)+pad)]#data around mask
            # # plt.scatter(subData.x-min(x)+pad,subData.y-min(y)+pad, s=0.1, c=subData.frame, cmap=colorcet.cm.isoluminant_cgo_70_c39_r)
            
            # subDataCleaned=cleanDataTmp[(cleanDataTmp.x > min(x)-pad) & (cleanDataTmp.x < max(x)+pad) & (cleanDataTmp.y > min(y)-pad) & (cleanDataTmp.y < max(y)+pad)]#data around mask
            # plt.scatter(subDataCleaned.x-min(x)+pad,subDataCleaned.y-min(y)+pad, s=0.1, c=subDataCleaned.frame, cmap=colorcet.cm.isoluminant_cgo_70_c39_r)
          
            indexer=[]
            for row in tsData.iterrows():
                tmpX=int(row[1].x)
                tmpY=int(row[1].y)   
                try:
                    if (tmpY,tmpX) in goodPositions: #If the localisation falls within the True og the mask, keep them, otherwise remove with False indexing
                        indexer.append(True)
                    else:
                        indexer.append(False)
                except:
                    indexer.append(False)
            if len(indexer) > 1:
                cleanDataTmp=tsData.loc[indexer] #If any localisations saved, store data.
                cleanDataTmp.drop_duplicates(subset='frame', keep=False, inplace=True) #Drops all frames with 2 or more localisations. It might be worth adding some sort of way to keep localisations when there is only 1 duplicate in many frames to help tracking a bit when there is a lot of noise.
                # ax.scatter(cleanDataTmp.x,cleanDataTmp.y, c=cleanDataTmp.frame)
                # plt.show()
                # if len(cleanDataTmp) > 000:
                #     sys.exit()
                cleanedDataList.append(cleanDataTmp)

        #### Testing 
        
        # cleanDataTmp=cleanedDataList[16]
        # t = tp.link(cleanDataTmp, search_range=120/framesPerSec, adaptive_stop=0.56, adaptive_step=0.99, memory=trackingMemory)
        # # subset to remove less than 25-frame track length
        # t1 = tp.filter_stubs(t, 50)
        # plt.scatter(t.x,t.y,c=t.particle)
        # plt.scatter(t1.x,t1.y,c=t1.particle)
        # tp.plot_traj(t)
        # tp.plot_traj3d(t1)
        
        
        #####
        
        # ## plot the elliptical fit (troubleshooting)
        # center, axes = (eCX, eCY), (ew, eh)
                
        # # generate points on the fitted ellipse
        # R=np.linspace(0,6,100)
        # a, b = axes
        # xx = center[0] + ew * np.cos(R) * np.cos(eRot) - eh * np.sin(R) * np.sin(eRot)
        # yy = center[1] + ew * np.cos(R) * np.sin(eRot) + eh * np.sin(R) * np.cos(eRot)
        # plt.figure(0)
        # plt.plot(x, y, color='blue', label='points')
        # plt.plot(xx, yy, '+', color='red', label='fitted ellipse', linewidth=2.0)
        # plt.Rectangle((eCX, eCY),ew,eh, fc='blue',ec="red")
        # plt.legend()
        # plt.axes().set_aspect('equal', 'datalim')

        # print("Number of coordinates:", len(contour), len(coords), len(coords1))

    if showMask == True:
        plt.imshow(finalMask)
    
    if len(cleanedDataList) > 1:
        tsData = pd.concat(cleanedDataList)
    elif len(cleanedDataList) == 1:
        tsData = cleanedDataList[0]
    else:
        tsData=pd.DataFrame(columns=tsData.columns) #Make an empty dataframe since all has been removed
        
    # indexer=[]
    # for row in tsData.iterrows():
    #     x=int(row[1].x)
    #     y=int(row[1].y)   
                             
    #     try:
    #         if finalMask[y,x]:
    #             indexer.append(True)
                
    #         else:
    #             indexer.append(False)
    #     except:
    #         indexer.append(False)
    # tsData=tsData.loc[indexer]

    return tsData, finalMask, IM_Mean_orig

def plotLocalisations(fname, images,df, verbose=True, s=0.1,cmap=colorcet.cm.glasbey, fps=6.25):
    '''
    Function that plot a timeseries with overlayed x,y points stored in df.

    Parameters
    ----------
    fname : string
        path where the output file is written to.
    images : list of 2D numpy arrays
        images plt.scatter will be overlayed on.
    df : pd.DataFrame
        dataframe containing x, y, frame.
    verbose : boolean, optional
        Used to turn on/off reporting of progress. The default is True.
    s : float, optional
        Used to control size of plt.scatter points. The default is 0.1.
    cmap : matplotlib.colors.LinearSegmentedColormap, optional
        sets colomap for the scatterplot. The default is cc.cm.glasbey.

    Returns
    -------
    None.

    '''
    ###################### Attenmpt to plot on the movie: ##########################
    metadata = dict(title="fromSegmentationScript", artist='Andreas Kjaer', comment='This movie is generated by the segmentation pipeline developed by Andreas Kjaer, 2021.')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    
    #open a video file to write the data to
    with writer.saving(fig,fname,100):
        #loop through the image frames
        for frameNum in range(len(images)):
            if verbose:
                if frameNum == int(len(images)/4):
                    print("25% complete")
                if frameNum == int(len(images)/2):
                    print("50% complete")
                if frameNum == int(len(images)*3/4):
                    print("75% complete")
            
            #reset axes (remove to get cummulative localisaton)
            plt.cla()
            frame = images[frameNum]
            #plot the image
            plt.imshow(frame)
            #plot the localisation, color by trackID
            plt.scatter(df.loc[df.frame == frameNum].x,
                        df.loc[df.frame == frameNum].y, 
                        s=s,cmap=cmap,
                        c=df.loc[df.frame == frameNum].particle.astype(int))
    
            #grab the frame
            writer.grab_frame()
    plt.close('all')   


for dataDir in os.listdir(sourceDir):
    #Testing purposes: dataDir = '20210522_sAK73_0_5_nM_JF646-Halo_multi_Agar_20ms_51deg_18%640_1on3off_3D_pos_3'
    # dataDir=os.listdir(sourceDir)[285]
    if process3Donly:
        if '3D' not in dataDir:# or '149' not in dataDir:
            continue #Skip dirs without 3D *it is not part of the analysis and is a result of wrong naming/wrong data
    if os.path.isdir(sourceDir + dataDir) and 'System' not in dataDir:
        os.chdir(sourceDir + dataDir)
        if os.path.exists(sourceDir + dataDir + "/compiledData.csv") and forceSaveCompiledData:
            os.remove(sourceDir + dataDir + "/compiledData.csv")
            print("Deleted old compiledData for sure")
        tifFiles = glob.glob("B_*.tif")
        print("Now working directory: " + dataDir)
        if len(tifFiles) != 0:
            compiledData = '' # Set so I can check if it is defined during the loop
            im = '' # Set so I can check if it is defined during the loop
            for tifFile in tifFiles:
                #Testing purposes: tifFile = 'B_sAK212_1nM_JF646-Halo_Agar_20ms_51deg_20%_640_constant-1.tif'
                # tifFile=tifFiles[0]
                if getFPSandExposureFromFname == True:
                    msMatches = re.findall("(\d+)ms", tifFile)
                    if len(msMatches) == 0:
                        exposureTime = defaultExposureTime/1000 #convert from ms to s
                    elif len(msMatches) > 1:
                        print("Warning! Found "+len(msMatches)+" exposure time candidates. Using the first one which is: "+msMatches[0])
                        exposureTime=int(msMatches[0])/1000 #time in s
                    else:
                        exposureTime=int(msMatches[0])/1000 #time in s

                    strobingFreq = re.findall("(\d+)on(\d+)off", tifFile)
                    
                    if len(strobingFreq[0]) !=2:
                        print("Unable to find strobing freq, assuming constant exposure!")
                        framesPerSec = (1/exposureTime)
                    else:
                        onTime=int(strobingFreq[0][0])
                        offTime=int(strobingFreq[0][1])
                        totalCycle=onTime+offTime

                        framesPerSec = (1/exposureTime)/(totalCycle/onTime)
                        
                #Open tif
                FramesOpened = False
                if verbose:
                    print("Beginning analysis of: "+ tifFile)
                    
                ############## Commence tracking #############
                if not os.path.exists(os.path.abspath(tifFile[:-4] + "_trackingDataTS.csv")) or forceAnalysis:
                    if verbose:
                        print("Performing linking")
                    #### load localisations from ThunderSTORM #######
                    try:
                        with open("B_tunderstormData_" + tifFile[2:-3] + "csv", "r") as file:
                            tsData = pd.read_csv(file)
                    except:
                        print("COULD NOT FIND TS DATA!, Aborting")
                        break
                    #Change values from nm to pixel (Seems to already be in pix sometimes)
                    if '[nm]' in tsData.columns.values[1]:
                        tsData.iloc[:,1] = tsData.iloc[:,1]/116.9
                        tsData.iloc[:,2] = tsData.iloc[:,2]/116.9
                        if tsData.columns.values[3] == 'z [nm]':
                            tsData.iloc[:,3] = tsData.iloc[:,3]/116.9

                    new_columns = tsData.columns.values
                    #Rename columns to trackpy compatible values
                    new_columns[1] = 'x'
                    new_columns[2] = 'y'
                    if tsData.columns.values[3] == 'z [nm]':
                        new_columns[3] = 'z'
                    tsData.columns  = new_columns
                    
                    if adjustChannelMapping:
                        tsData.x = tsData.x + xCorrection
                        tsData.y = tsData.y + yCorrection
                    
                    if useClusterFiltering:
                        tsData = clusterFiltering(tsData)
                    
                    
                    if useCellOutlineFiltering:
                        tmpList=glob.glob(sourceDir+dataDir+"/A*cell_outline*")
                        #outlineFname="A"+tifFile[1:-4]+"_cell_outline.tif"  ##legacy
                        outlineFname=tmpList[0]
                        # sys.exit()
                        tsData, finalMask, IM_Mean_orig = cellOutlineFiltering(tsData,outlineFname, lowerCanny, upperCanny, showMask=False)
                        # tmp = cellOutlineFiltering(tsData,outlineFname, showMask=True, useCHullAlways=True)
                        plt.figure()
                        plt.imshow(finalMask)
                        # plt.imshow(IM_Mean_orig)
                        plt.scatter(tsData.x,tsData.y,s=0.01)
                        # plt.scatter(t1.x,t1.y,s=0.1)
                        plt.savefig("finalMask_"+tifFile[2:-4]+".png")
                        plt.close()
                        plt.figure()
                        # plt.imshow(finalMask)
                        plt.imshow(IM_Mean_orig)
                        plt.scatter(tsData.x,tsData.y,s=0.01)
                        plt.savefig("IM_Mean_orig_"+tifFile[2:-4]+".png")
                        plt.close()

                    if tsData.shape[0] != 0:
                        
                        ############ Linking and tracking #################
                        # Track/link the localisations
                        #t = tp.link(tsData, search_range=120/framesPerSec, adaptive_stop=0.56, adaptive_step=0.99, memory=trackingMemory)
                        t = tp.link(tsData, search_range=6, adaptive_stop=0.56, adaptive_step=0.99, memory=trackingMemory)
                        # subset to remove less than 25-frame track length
                        t1 = tp.filter_stubs(t, minTrackLen)
                    else:
                        t=tsData
                        t1=tsData
                        t['particle']=0
                        t1['particle']=0
                                       
                    
                    t1['framesPerSec'] = framesPerSec
                    # Compare the number of particles in the unfiltered and filtered data.
                    print('Before:', t['particle'].nunique())
                    print('After:', t1['particle'].nunique())
                    if t1['particle'].nunique() == 0:
                        print("Aborting. No tracks found!")
                        with open("no_tracks_found", "w") as f:
                            f.write("No tracks found for" + dataDir)
                        break
                                ####### Save the tracking data ##########
                    with open(os.path.abspath(tifFile)[:-4] + "_trackingDataTS.csv", "w") as file:
                        t1.to_csv(file)
                else: #If tracking exist, open the data from file unless the user required re-tracking/forceAnalysis
                    print("Tracking data for :" + tifFile + "was found and loaded from file. Tracking analysis was not performed.")
                    with open(os.path.abspath(tifFile)[:-4] + "_trackingDataTS.csv", "r") as file:
                        t1 = pd.read_csv(file)
                ########### Write a movie with the tracks annotated as animated scatter #########
                if (not os.path.exists(os.path.abspath(tifFile)[:-4] + "_localisations.mp4") or forceMovie) and saveMovies:
                    if verbose:
                        print("Writing movie")
                    images = ski.io.imread(tifFile)
                    plotLocalisations(os.path.abspath(tifFile)[0:-4]+"_localisations.mp4", images,t, verbose=True, s=0.1,cmap=colorcet.cm.glasbey, fps=framesPerSec)
                else:
                    if verbose:
                        print("movie was already made (or set to not being produced) for: "+ tifFile + "Continuing the script without making a new one!")
                
                ########### calculate iMSD and export #############
                if saveMSDData:
                    if not os.path.exists(sourceDir + dataDir + "/iMSD.csv") or forceSaveiMSDData:
                        print("Now saving iMSD!")
                        iMSD = tp.imsd(t1, micronPerPixel, framesPerSec, maxLagtime)
                        with open(sourceDir + dataDir + "/iMSD.csv", "w") as file:
                            iMSD.to_csv(file)
                    else:
                        if verbose:
                            print("iMSD already exists, skipping!")
                
                ########### Append the tracking data to compiledData (Pandas dataframe accumulating all data from a directory) #############
                if not os.path.exists(sourceDir + dataDir + "/compiledData.csv") or forceSaveCompiledData:
                    if verbose:
                        print("Now saving compiledData!")
                    try:
                        
                        t1['particle'] = max(compiledData['particle']) + t1['particle'] + 100
                        compiledData = compiledData.append(t1, ignore_index=True)
                        if verbose: 
                            print("Appended compiledData successfully")
                    except:
                        compiledData = pd.DataFrame(data=t1)
                
            if str(compiledData) == '':
                if verbose:
                    print("Compiled data was not created for some reason???")
            else:
                if not os.path.exists(sourceDir + dataDir + "/compiledData.csv") or forceSaveCompiledData:
                    ####### Save the compiled tracking data ##########
                    with open(sourceDir + dataDir + "/compiledData.csv", "w") as file:
                        compiledData.to_csv(file)
                        if sum(t1.x == compiledData.x) == len(t1.x):
                            print("Correct compiled data saved for sure!")
                        else:
                            print("WRONG COMPILED DATA SAVED ?!?!")
                else:
                    if verbose:
                        print("Compiled data already found, skipping writing!\n")
            
