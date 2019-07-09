# this prog predicts X and Y with help of NN
import numpy as np #we use this lib for math functions
# from array import *
import csv
def ReadFileData (filename, userId, itemId, Xmin, Xmax , Ymin, Ymax):
    """ we set array data from file """
    
    f = open (filename)
    print (filename,"opened")
    r = csv.reader (f)
    
    for line in r:
        userId.append (line[0])
        itemId.append (line[1])
        Xmin.append (line[2])
        Ymin.append (line [3])
        Xmax.append (line [4])
        Ymax.append (line [5])
        #print (line)
        
    f.close()
    print ("Data file closed")
    return 1


def ReadFileAnswers (filename, itemIdTrue, XminTrue, XmaxTrue, YminTrue, YmaxTrue):
    """ we set array data from file with answers"""
    fa = open (filename)
    ra = csv.reader (fa)
    
    for linea in ra :
        itemIdTrue.append (linea[0])
        XminTrue.append (linea[1])
        YminTrue.append (linea [2])
        XmaxTrue.append (linea [3])
        YmaxTrue.append (linea [4])
        #print (linea)
    fa.close()
    print ("Data file with answers closed")
    return 1


def sigmoid (x, T) :
    """ this is sigmoida for our NN"""
    return 1 / (1 + np.exp ( - float (x) / T ) )

def GetXminTrue (itemNumber) :
    """ func find XminTrue for itemNumber"""
    buffer = 0
    for i in range  (944) :
        if itemIdTrue [i] == itemNumber : buffer = XminTrue [i]
    return buffer

def GetXmaxTrue (itemNumber) :
    """ func find XmaxTrue for itemNumber"""
    buffer = 0
    for i in range  (944) :
        if itemIdTrue [i] == itemNumber : buffer = XmaxTrue [i]
    return buffer

def GetYminTrue (itemNumber) :
    """ func find YminTrue for itemNumber"""
    buffer = 0
    for i in range  (944) :
        if itemIdTrue [i] == itemNumber : buffer = YminTrue [i]
    return buffer

def GetYmaxTrue (itemNumber) :
    """ func find YmaxTrue for itemNumber"""
    buffer = 0
    for i in range  (944) :
        if itemIdTrue [i] == itemNumber : buffer = YmaxTrue [i]
    return buffer



def SetFirst (FirstValue, W) :
    """ func determinze first weitht """
    for i in range ( 944 ) :
        if i == 0 : W.append  ( FirstValue )
        else : W.append ( 0)
        #print (W)
    return 1




DataFile1 ="train_data.csv"; DataFile2 ="train_answers.csv"
# we create  arrays to save data from files
userId = []; itemId = []; Xmin = []; Xmax = []; Ymin = []; Ymax = []
itemIdTrue = []; XminTrue = []; XmaxTrue = []; YminTrue = []; YmaxTrue = []

ReadFileData (DataFile1, userId, itemId, Xmin, Xmax, Ymin, Ymax)
ReadFileAnswers (DataFile2, itemIdTrue, XminTrue, XmaxTrue, YminTrue, YmaxTrue)

T = 100 #parametr so caled temperature
L = 0.2 # coefficient of learning
WUXmin = []; SetFirst (0.85, WUXmin)
WIXmin = []; SetFirst (0.91, WIXmin)
WUXmax = []; SetFirst (0.98, WUXmax)
WIXmax = []; SetFirst (0.975, WIXmax)
WUYmin= []; SetFirst (0.95, WUYmin)
WIYmin = []; SetFirst (0.938, WIYmin)
WUYmax = []; SetFirst (0.947, WUYmax)
WIYmax = []; SetFirst (0.965, WIYmax)
# set first parameters of weithts NN

ErrWUXmin = []; SetFirst (318000,ErrWUXmin)
ErrWUXmax = []; SetFirst (319000,ErrWUXmax)
ErrWIXmin = []; SetFirst (320000,ErrWIXmin)
ErrWIXmax = []; SetFirst (321000,ErrWIXmax)
ErrWUYmin = []; SetFirst (322000,ErrWUYmin)
ErrWUYmax = []; SetFirst (323000,ErrWUYmax)
ErrWIYmin = []; SetFirst (324000,ErrWIYmin)
ErrWIYmax = []; SetFirst (325000,ErrWIYmax)
# first errors

#XminForecast = []; SetFirst (150, XminForecast)
#XmaxForecast = []; SetFirst (160, XmaxForecast)
#YminForecast = []; SetFirst (170, YminForecast)
#YmaxForecast = []; SetFirst (166, YmaxForecast)
# first forecasts

for k in range (943) : #very important loop
    
    xWUXmin = WUXmin[k]* float (userId [k+1]) + WIXmin [k]* float (itemId [k+1])+ float (Xmin[k+1])
    xWUXmax = WUXmax[k]* float (userId [k+1]) + WUXmax [k]* float (itemId [k+1])+ float (Xmax[k+1])
    xWIXmin = WIXmin[k]* float (userId [k+1]) + WIXmin [k]* float (itemId [k+1])+ float (Xmin[k+1])
    xWIXmax = WIXmax[k]* float (userId [k+1]) + WIXmax [k]* float (itemId [k+1])+ float (Xmax[k+1])
    yWUYmin = WUYmin[k]* float (userId [k+1]) + WIYmin [k]* float (itemId [k+1])+ float (Ymin[k+1])
    yWUYmax = WUYmax[k]* float (userId [k+1]) + WUYmax [k]* float (itemId [k+1])+ float (Ymax[k+1])
    yWIYmin = WIYmin[k]* float (userId [k+1]) + WIYmin [k]* float (itemId [k+1])+ float (Ymin[k+1])
    yWIYmax = WIYmax[k]* float (userId [k+1]) + WIYmax [k]* float (itemId [k+1])+ float (Ymax[k+1])
    #print ("arg of sigmoida", xWUXmin, xWUXmax, xWIXmin, xWIXmax, yWUYmin, yWUYmax, yWIYmin, yWIYmax)
    tempXminTrue= GetXminTrue (itemId [k+1])
    tempXmaxTrue= GetXmaxTrue (itemId [k+1])
    tempYminTrue = GetYminTrue (itemId [k+1])
    tempYmaxTrue = GetYmaxTrue (itemId [k+1])
    #print (" min max true ", tempXminTrue, tempXmaxTrue, tempYminTrue , tempYmaxTrue)
    if tempXminTrue == 0 : print ("item ", itemId [k+1], " is not found ")
    if tempXmaxTrue == 0 : print ("item ", itemId [k+1], " is not found ")
    if tempYminTrue == 0 : print ("item ", itemId [k+1], " is not found ")
    if tempYmaxTrue == 0 : print ("item ", itemId [k+1], " is not found ")
    
    WUsigmoidXmin = sigmoid (xWUXmin, T)
    WUsigmoidXmax = sigmoid (xWUXmax, T)
    WIsigmoidXmin = sigmoid (xWIXmin, T)
    WIsigmoidXmax = sigmoid (xWIXmax, T)
    WUsigmoidYmin = sigmoid (yWUYmin, T)
    WUsigmoidYmax = sigmoid (yWUYmax, T)
    WIsigmoidYmin = sigmoid (yWIYmin, T)
    WIsigmoidYmax = sigmoid (yWUYmax, T)
    #print (" sigmoids ", WUsigmoidXmin, WUsigmoidXmax, WIsigmoidXmin, WIsigmoidXmax, WUsigmoidYmin, WUsigmoidYmax, WIsigmoidYmin, WIsigmoidYmax)

    ErrWUXmin [k+1] = 0.5*(float (tempXminTrue) - WUsigmoidXmin)**2
    ErrWUXmax [k+1] = 0.5*(float (tempXmaxTrue) - WUsigmoidXmax)**2
    ErrWIXmin [k+1] = 0.5*(float (tempXminTrue) - WIsigmoidXmin)**2
    ErrWIXmax [k+1] = 0.5*(float (tempXmaxTrue) - WIsigmoidXmax)**2
    ErrWUYmin [k+1] = 0.5*(float (tempYminTrue) - WUsigmoidYmin)**2
    ErrWUYmax [k+1] = 0.5*(float (tempYmaxTrue) - WUsigmoidYmax)**2
    ErrWIYmin [k+1] = 0.5*(float (tempYminTrue) - WIsigmoidYmin)**2
    ErrWIYmax [k+1] = 0.5*(float (tempYmaxTrue) - WIsigmoidYmax)**2
    #print (" errors ", ErrWUXmin [k], ErrWUXmax [k], ErrWIXmin [k], ErrWIXmax [k], ErrWUYmin [k], ErrWUYmax [k], ErrWIYmin [k], ErrWIYmax [k])
    #print (" errors ", ErrWUXmin [k+1], ErrWUXmax [k+1], ErrWIXmin [k+1], ErrWIXmax [k+1], ErrWUYmin [k+1], ErrWUYmax [k+1], ErrWIYmin [k+1], ErrWIYmax [k+1])

    if ErrWUXmin [k+1]   <  ErrWUXmin [k] :
        WUXmin [k+1] = WUXmin [k] - (L*float(Xmin [k+1])* WUsigmoidXmin*(1-WUsigmoidXmin))/T
    else : WUXmin [k+1] = WUXmin [0]; #print (" weight ", WUXmin [k+1])
    if ErrWUXmax [k+1]   <  ErrWUXmax [k] :
        WUXmax [k+1] = WUXmax [k] - (L*float(Xmax [k+1])* WUsigmoidXmax*(1-WUsigmoidXmax))/T
    else : WUXmax [k+1] = WUXmax [0]; #print (" weight ", WUXmax [k+1])

    if ErrWIXmin [k+1]   <  ErrWIXmin [k] :
        WIXmin [k+1] = WIXmin [k] - (L*float(Xmin [k+1])* WIsigmoidXmin*(1-WIsigmoidXmin))/T
    else : WIXmin [k+1] = WIXmin [0]; #print (" weight ", WIXmin [k+1])
    if ErrWIXmax [k+1]   <  ErrWIXmax [k] :
        WIXmax [k+1] = WIXmax [k] - (L*float(Xmax [k+1])* WIsigmoidXmax*(1-WIsigmoidXmax))/T
    else : WIXmax [k+1] = WIXmax [0]; #print (" weight ", WIXmax [k+1]) 

    if ErrWUYmin [k+1]   <  ErrWUYmin [k] :
        WUYmin [k+1] = WUYmin [k] - (L*float(Ymin [k+1])* WUsigmoidYmin*(1-WUsigmoidYmin))/T
    else : WUYmin [k+1] = WUYmin [0]; #print (" weight ", WUYmin [k+1])
    if ErrWUYmax [k+1]   <  ErrWUYmax [k] :
        WUYmax [k+1] = WUYmax [k] - (L*float(Ymax [k+1])* WUsigmoidYmax*(1-WUsigmoidYmax))/T
    else : WUYmax [k+1] = WUYmax [0]; #print (" weight ", WUYmax [k+1])
    
    if ErrWIYmin [k+1]   <  ErrWIYmin [k] :
        WIYmin [k+1] = WIYmin [k] - (L*float(Xmin [k+1])* WIsigmoidYmin*(1-WIsigmoidYmin))/T
    else : WIYmin [k+1] = WIYmin [0]; #print (" weight ", WIYmin [k+1])
    if ErrWIYmax [k+1]   <  ErrWIYmax [k] :
        WIYmax [k+1] = WIYmax [k] - (L*float(Ymax [k+1])* WIsigmoidYmax*(1-WIsigmoidYmax))/T
    else : WIYmax [k+1] = WIYmax [0]; #print (" weight ", WIYmax [k+1])

  
    if k == 0 :
        print (k, "step", WUXmin [0], WUXmax [0], WIXmin [0], WIXmax [0], WUYmin [0], WUYmax [0], WIYmin [0], WIYmax [0])
    if k == 100 :
        print (k, "step", WUXmin [100], WUXmax [100], WIXmin [100], WIXmax [100], WUYmin [100], WUYmax [100], WIYmin [100], WIYmax [100])
    if k == 200 :
        print (k, "step", WUXmin [200], WUXmax [200],  WIXmin [200], WIXmax [200], WUYmin [200], WUYmax [200], WIYmin [200], WIYmax [200])
    if k == 300 :
        print (k, "step", WUXmin [300], WUXmax [300], WIXmin [300], WIXmax [300], WUYmin [300], WUYmax [300], WIYmin [300], WIYmax [300])
    if k == 400 :
        print (k, "step", WUXmin [400], WUXmax [400], WIXmin [400], WIXmax [400], WUYmin [400], WUYmax [400], WIYmin [400], WIYmax [400])
    if k == 500 :
        print (k, "step", WUXmin [k], WUXmax [k], WIXmin [k], WIXmax [k], WUYmin [k], WUYmax [k], WIYmin [k], WIYmax [k])
    if k == 600 :
        print (k, "step", WUXmin [k], WUXmax [k], WIXmin [k], WIXmax [k], WUYmin [k], WUYmax [k], WIYmin [k], WIYmax [k])
    if k == 700 :
        print (k, "step", WUXmin [k], WUXmax [k], WIXmin [k], WIXmax [k], WUYmin [k], WUYmax [k], WIYmin [k], WIYmax [k])
    if k == 800 :
        print (k, "step", WUXmin [k], WUXmax [k], WIXmin [k], WIXmax [k], WUYmin [k], WUYmax [k], WIYmin [k], WIYmax [k])
    if k == 900 :
        print (k, "step", WUXmin [k], WUXmax [k], WIXmin [k], WIXmax [k], WUYmin [k], WUYmax [k], WIYmin [k], WIYmax [k])
    if k == 942 :
        print (k, "step", WUXmin [k], WUXmax [k], WIXmin [k], WIXmax [k], WUYmin [k], WUYmax [k], WIYmin [k], WIYmax [k])
    
input("Print something to continue")
