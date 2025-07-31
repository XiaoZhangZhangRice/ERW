# -*- coding: utf-8 -*-
"""

This script is for bulk processing Licor data to calculate flux estimates
from raw LICOR data and start/end time estimates.

Other files needed:
    - LICOR Data file
    - CSV table of time estimates

@author: jrlamb - Aug 6 2024
"""

# %% USER INPUTS

# STEP 1: Chamber geometry
chamberVolume = 900 # [cm3], the volume of the chamber, tubes, and licor
                    # (licor internal volume is ~20cm3)
surfaceArea = 450  # [cm2], the surface area that is emitting flux
                    # (small mason jar surface area is 38.3cm2)
                    
# STEP 2: Files

licorFilePath = "../../DataMamiraua/2024May/diurnalTreeCH4/licor.data"

infoFilePath = "../../DataMamiraua/2024May/diurnalTreeCH4/treeChambersMay25.csv"

outputFilePath = "../../DataMamiraua/2024May/diurnalTreeCH4/processedExpTreeChambersMay25.csv"

# STEP 3: Flux options

durationEstimate = 180 # [seconds], duration of each flux measurement
startCut = 90 # [seconds] how much to remove from the front for wiggles


# %% NUTS AND BOLTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skl

# %% LOAD LICOR DATA
residLim = 0.025

liDF = pd.read_table(licorFilePath, sep='\t', skiprows=(0,1,2,3,4,6)) # load file
liDF = liDF[liDF['RESIDUAL'] < residLim] # remove points with high residuals
liDF['CH4'] *= 0.001 #convert ppb to ppm


# %% PREPARE TIMES
''' EDIT AS NEEDED
The goal here is to make a DF with a column for unixStartEstimate and unixEndEstimate
for each chamber measurement
'''

infoDF = pd.read_csv(infoFilePath)
outputDF = infoDF.copy()

# a function to take the plot number and recorded time, and make a unix time
def makeUnixStart(row, colID):
    date = 20240525 + int(row['plot']) # get the right day
    dateTime = str(date) + 'T' + row[colID] # add the day to the time
    timeStamp = pd.to_datetime(dateTime).timestamp() # convert to tiemstamp
    return timeStamp + startCut
# Extract the start times
infoDF['AMH1Start'] = infoDF.apply(makeUnixStart, axis=1, colID="T1AM")
infoDF['AMH2Start'] = infoDF.apply(makeUnixStart, axis=1, colID="T2AM")

infoDF['PMH1Start'] = infoDF.apply(makeUnixStart, axis=1, colID="T1PM")
infoDF['PMH2Start'] = infoDF.apply(makeUnixStart, axis=1, colID="T2PM")

# Make the end time estimate (+ startCut)

infoDF[['AMH1End', 'AMH2End', 'PMH1End', 'PMH2End']] = infoDF[['AMH1Start', 'AMH2Start', 'PMH1Start', 'PMH2Start']] + durationEstimate - startCut

# %% FUNCTIONS TO CALCULATE FLUX

def getFluxesExp(row, startRow, endRow):
    start = row[startRow]
    end = row[endRow]
    
    # load CH4 data
    CH4 = liDF[(liDF['SECONDS'] >= start) & (liDF['SECONDS'] <= end)]['CH4']
    
    # load times and normalize
    times = liDF[(liDF['SECONDS'] >= start) & (liDF['SECONDS'] <= end)]['SECONDS']
    times = times-times.iloc[0] + startCut
    
    # calculate first derivative
    dCdT = np.gradient(CH4)
    
    # plt.plot(CH4, dCdT, 'b.', markersize=3)
    
    
    # check linearity of dCdT vs CH4
    corr = np.corrcoef(CH4, dCdT)[0,1]
    
    # get linear fit of dCdT vs CH4
    m, b = np.polyfit(CH4, dCdT, 1)

    
    # get intercepts
    dCdTMax = b
    internalC = -1 * b/m
    

    # plt.plot(CH4, dCdT, 'b.', markersize=3)
    # plt.plot([0, internalC], [dCdTMax, 0], 'k.-')
    # plt.show()
        
    
    # row['fluxLinear'] = np.mean(dCdT)
    # row['fluxExp'] = dCdTMax
    # row['internalC'] = internalC
    # row['fluxLinearity'] = corr
    return dCdTMax

def getFluxesLin(row, startRow, endRow):
    start = row[startRow]
    end = row[endRow]
    
    # load CH4 data
    CH4 = liDF[(liDF['SECONDS'] >= start) & (liDF['SECONDS'] <= end)]['CH4']
    
    # load times and normalize
    times = liDF[(liDF['SECONDS'] >= start) & (liDF['SECONDS'] <= end)]['SECONDS']
    times = times-times.iloc[0] + startCut
    
    # calculate first derivative
    dCdT = np.gradient(CH4)
    
    return np.mean(dCdT)

# %% RUN GET FLUXES
infoDF['AMH1Flux'] = infoDF.apply(getFluxesExp, axis=1, startRow='AMH1Start',
                                  endRow='AMH1End')
infoDF['AMH2Flux'] = infoDF.apply(getFluxesExp, axis=1, startRow='AMH2Start',
                                  endRow='AMH2End')
infoDF['PMH1Flux'] = infoDF.apply(getFluxesExp, axis=1, startRow='PMH1Start',
                                  endRow='PMH1End')
infoDF['PMH2Flux'] = infoDF.apply(getFluxesExp, axis=1, startRow='PMH2Start',
                                  endRow='PMH2End')

# %% CONVERT FLUXES TO PROPER UNITS
ppm2Moles = (1/1000000) * chamberVolume/22400 # [ppm/s] to [mols/s]
ppm2Mass = ppm2Moles * 16.04 # [mols/s] to [g/s] by CH4 16 g/mol
ppm2flux = 1000*(ppm2Mass*3600)/(surfaceArea/10000) # [g/s] to [g/m2-d]

infoDF[['AMH1Flux', 'AMH2Flux', 'PMH1Flux', 'PMH2Flux']] *= ppm2flux

# %% Linear VS EXPONENTIAL FLUX ANALYSIS

# infoDF = infoDF[infoDF['fluxLinearity']<-0.0]

# plt.plot(infoDF['fluxLinear'], infoDF['fluxExp'], '.')
# plt.plot([0,max(infoDF['fluxExp'])],[0,max(infoDF['fluxExp'])], 'k-')


# p = np.polyfit(infoDF['fluxLinear'], infoDF['fluxExp'], 1)
# print(p)
# plt.plot(infoDF['fluxLinear'], np.polyval(p, infoDF['fluxLinear']), 'r-')

# plt.show()

# plt.hist(infoDF['fluxLinearity'])
# plt.show()


# %% SAVE FLUXES TO OUTPUT

outputDF[['AMH1Flux', 'AMH2Flux', 'PMH1Flux', 'PMH2Flux']] = infoDF[['AMH1Flux', 'AMH2Flux', 'PMH1Flux', 'PMH2Flux']]

outputDF.to_csv(outputFilePath, index=False)
