import sys
import os
droppedFile = sys.argv[1]
fo = open(droppedFile, "r")
lines = fo.readlines()
count = 0
sampleno=1
printing = False
internal = 0
temp=[]
datestore=""
timestore=""
ch4=False
co2= False
no2=False
start= False
filename="Filename= "+droppedFile
import pandas as pd
df = pd.DataFrame(columns=['CH4', 'CO2', 'N2O', 'Time','Date'])


for line in lines:
    if "[Header]" in line and start:
        temp.append(str(timestore).ljust(15))
        temp.append(str(datestore).ljust(15))
        timestore=""
        datestore=""
        print(temp)
        df.loc[len(df)] = temp
        temp=[]
        printing = False
        gastype=0
       # temp.append(str(sampleno).ljust(15))
        sampleno+=1
            
    
    if "[Peak Table(Ch1)]" in line:
        internal=0
        printing= True
        ch4=True
        co2=False
        no2=False

    if "[Peak Table(Ch2)]" in line:
        internal=0
        ch4= False
        co2= True
        no4= False
        
    if "[Peak Table(Ch3)]" in line:
        internal=0
        ch4=False
        co2=False
        no4=True
            

    if printing :
        if internal == 1:
            peaks= line.split()
            peakno = int(peaks[3])
            if co2:
                co2peak= int(peaks[3])
            if ch4:
                ch4peak= int(peaks[3])
            if no2:
                no2peak= int(peaks[3])
            if "0" in line:
                temp.append(str(0).ljust(10))
            
        if internal>=3 and peakno!=0:
            date= line.split()
            insertval= date[4]
            if ch4:
                if(float(date[1])<4.10 or float(date[1])>4.30):
                    insertval="0"
                    #print("co2 val is "+ date[1])
                if(ch4peak==1):
                    temp.append(insertval.ljust(10))
                ch4peak-=1
            elif co2:
                if(float(date[1])<7.70 or float(date[1])>7.90):
                    insertval="0"
                    #print("co2 val is "+ date[1])
                if(co2peak==1):
                    temp.append(insertval.ljust(10))
                co2peak-=1
            elif no2:
                if(float(date[1])<10.70 or float(date[1])>10.80):
                    insertval="0"
                    #print("co2 val is "+ date[1])
                if(no2peak==1):
                    temp.append(insertval.ljust(10))
                no2peak-=1
            else:
                temp.append(insertval.ljust(10))
            peakno-=1

        internal+=1

    if "Output Date" in line:
        date= line.split()
        datestore=date[2]
            
        
    if "Output Time" in line:
        timearr= line.split()
        timing= timearr[2]
        label= timearr[3]
        time= timing+label
            
        timestore=time


        
    count+=1    
    start=True   
        
temp.append(str(timestore).ljust(15))
temp.append(str(datestore).ljust(15))
  

df.loc[len(df)] = temp

import numpy as np
df.index = np.arange(1, len(df)+1)




df.to_csv("trial.csv", index=True)
