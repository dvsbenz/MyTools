#Davis Benz
#CalibrateAPS
#06/18/2021

import subprocess, time
import re
import numpy as np
import numpy.random as rn
import os, os.path
import datetime

##################################################
################ FUNCTIONS #######################
##################################################
######## Setup Functions ##########
def IntializeNodes():
    raw   = subprocess.check_output('LinuxTestTool -c can1 -n 70 opstate', shell = True) #check current state of the cassette node
    match = re.search('(?<=moduleState: )(.*)', raw)
    state = match.group(0)
    if state != '4':              
        raw = subprocess.check_output('LinuxTestTool -c can1 -n 70 modconf 0 0 0 0', shell = True)
        raw = subprocess.check_output('LinuxTestTool -c can1 -n 70 opstate 4', shell = True)
        raw = subprocess.check_output('LinuxTestTool -c can1 -n 70 opstate', shell = True)
        print('Cassette Node')
        print(raw)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
######## Control Functions ########
def PingCassetteNode(valvePositions):
    #ping the cassette to set valve positions/read out feedback
    nodeString  = 'PT0144 -c can1 -n 70 ' + valvePositions + ' 0 0'   #valves change postion throughout the run, allows for a position input everytime the node is pinged
    nodeRawData = subprocess.check_output(nodeString, shell = True)   #Ping Cassette Node and save the message
    match       = re.search('(?<=loadCellCounts: )(.*)', nodeRawData) #regular expression search for load cell counts
    countsAPS   = match.group(0)

    return countsAPS

def CloseValves():
    #function to the close all 3 pinch valves
    position = '1 1 1'
    nodeString = 'PT0144 -c can1 -n 70 ' + position +' 0 0'
    raw = subprocess.check_output(nodeString, shell = True)
    #print(raw)

    return position
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def InitializeCalDataFile(machine):
    #Intialize Files
        #calibration specific data
    calInfo = open('CalInfo_M_' + machine + '.txt', 'a+')

    return calInfo

def WriteData(calInfo, machine, forceGain, calData):
    calInfo.write('Machine, ' + machine +'\n')
    calInfo.write('APS Positive Gain [N/count], ' + str(forceGain) +'\n')
    calInfo.write('500g, 0g\n')

    for k in range(len(calData[0,:])):
        calInfo.write(str(calData[0, k]) + ', ' + str(calData[1, k]) + '\n')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Setup
mass      = np.array([500.0, 0.0]) #Masses Used for Calibration
numCounts = 20
calData   = np.array([np.zeros(numCounts), np.zeros(numCounts)])  #Array that aps counts will be stored in
counts    = np.zeros(len(mass))                                   #vector with average count for each mass
force     = (mass/1000) * 9.81                                    #vector with the calibration forces in Newtons

#IntializeNodes()                                                 #Initialize the Cassette Node
valvePos = CloseValves()                                          #set valves to the closed position

print('')
print('APS Calibration') 
print('')

machine  = raw_input('enter machine number and press "Enter": ')
print('')

calInfo = InitializeCalDataFile(machine)

for i in range(len(mass)):
    print('place ' + str(mass[i]) + 'g mass on the APS')
    raw_input('press "Enter" when the mass is in place')
    for j in range(numCounts):
        countsAPS    = PingCassetteNode(valvePos)
        calData[i,j] = countsAPS
        time.sleep(0.25)
    counts[i] = np.int(np.mean(calData[i,:]))
    print('')

# print(counts)
# print(calData)

forceGain = (force[1]-force[0])/(counts[1]-counts[0])

# print('Gain Force = ' + str(gainForce) + ' [N/count]')
# print('Gain Press = ' + str(gainPres) + '[mmHg/count]')

WriteData(calInfo, machine, forceGain, calData)

print('Calibration Complete')
print('')
