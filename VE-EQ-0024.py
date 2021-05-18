#Davis Benz
#VE-EQ-0024
#01/28/2020

import subprocess, time
import re
import numpy as np
from numpy import genfromtxt
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

    raw   = subprocess.check_output('LinuxTestTool -c can1 -n 30 opstate', shell = True) #check current state of the centrifuge node
    match = re.search('(?<=moduleState: )(.*)', raw)
    state = match.group(0)
    if state != '4':
        raw = subprocess.check_output('LinuxTestTool -c can1 -n 30 modconf 0 0 0 0', shell = True) #address timeout error
        raw = subprocess.check_output('LinuxTestTool -c can1 -n 30 opstate 4', shell = True)       #set centrifuge node to operating state
        raw = subprocess.check_output('LinuxTestTool -c can1 -n 30 opstate', shell = True)         #check current state of the centrifuge node
        print('Centrifuge Node')
        print(raw)

    raw   = subprocess.check_output('LinuxTestTool -c can1 -n 51 opstate', shell = True) #check current state of the centrifuge node
    match = re.search('(?<=moduleState: )(.*)', raw)
    state = match.group(0)
    if state != '4':
        raw = subprocess.check_output('LinuxTestTool -c can1 -n 51 modconf 0 0 0 0', shell = True) #address timeout error
        raw = subprocess.check_output('LinuxTestTool -c can1 -n 51 opstate 4', shell = True)       #set centrifuge node to operating state
        raw = subprocess.check_output('LinuxTestTool -c can1 -n 51 opstate', shell = True)         #check current state of the centrifuge node
        print('Draw Pump Node')
        print(raw)
    raw = subprocess.check_output('PT0141C -c can1 -n 51 50 50 3500 0 5 0', shell = True)

def PressureTestRange():

    presVals  = np.array([-350.0, -250.0, -150.0, -50, 0.0, 200.0, 400.0, 600.0, 800.0, 1000.0])
    #presVals  = np.array([600.0, 1000.0])
    setPoints = rn.permutation(presVals) #randomize order of that points are tested at to eliminate bias error

    return setPoints
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
######## Control Functions ########
def GetRefPres():
    #convert ADC counts to calibrated pressure value
    #refCounts = float(subprocess.check_output('cat /sys/bus/iio/devices/iio:device0/in_voltage4_raw', shell = True)) #read in counts from BBB ADC and convert from a string to a float
    #shift pressure tansducer by X.X(8.5Psi for verification) psi so readings never show negative pressure
        #transducer wont output negative voltage
    #pressure transducer to ADC cal curve
        # Pressure [Psi] = 0.008732 - 0.027870
    #full equation to go from counts to mmHg 
        # Pressure [mmHg] = ((G*counts - yInt) - pShift)*51.714933

    refPres = 0.635350*float(subprocess.check_output('cat /sys/bus/iio/devices/iio:device0/in_voltage1_raw', shell = True)) - 444.910249
    
    return refPres

def PingCassetteNode(valvePositions):
    #ping the cassette to set valve positions/read out feedback
    nodeString  = 'PT0144 -c can1 -n 70 ' + valvePositions + ' 0 0'             #valves change postion throughout the run, allows for a position input everytime the node is pinged
    nodeRawData = subprocess.check_output(nodeString, shell = True)               #Ping Cassette Node and save the message
    match       = re.search('(?<=loadCellCounts: )(.*)', nodeRawData)         #regular expression search for load cell counts
    countsAPS   = match.group(0)
    match       = re.search('(?<=cassetteDoorStatus: Position)(.*)', nodeRawData) #regular expression search for latch position feedback
    latchPos    = match.group(0)

    return countsAPS, latchPos

def CloseValves():
    #function to the close all 3 pinch valves
    position = '1 1 1'
    nodeString = 'PT0144 -c can1 -n 70 ' + position +' 0 0'
    raw = subprocess.check_output(nodeString, shell = True)
    #print(raw)

    return position

def OpenValves():
    #function to the open all 3 pinch valves
    position = '0 0 0'
    nodeString = 'PT0144 -c can1 -n 70 ' + position +' 0 0'
    raw = subprocess.check_output(nodeString, shell = True)
    #print(raw)

    return position

def SetValves(valvePositions):
    #function to the set the valve positions
    nodeString  = 'PT0144 -c can1 -n 70 ' + valvePositions + ' 0 0'         #valve change postion throughout the run, allows for a position input 
    raw         = subprocess.check_output(nodeString, shell = True) #Ping Cassette Node, keep all three pinch valve opens for this test
    print(raw)

def SetPumpSpeed(output):
    #set the pump speed based on the PID output
    if output > 400:  #check that we don't push a value to the pump node that is larger than the input range for the pump speed 
        output = 400  #limit the pump max output
    elif output < -400:
        output = -400

    raw = subprocess.check_output('PT0141 -c can1 -n 51 ' + str(output) + ' 1 0 0', shell = True)

def SetPumpRacePosition(pumpSpeed, position):
    raw = subprocess.check_output('PT0141 -c can1 -n 51 ' + str(int(pumpSpeed)) + ' ' + str(position) + ' 0 0', shell = True)
    #print(raw)

def PingCentrifugeNode(compressor):
    #ping the cassette to set valve positions/read out feedback
    nodeString  = 'PT0139 -c can1 -n 30 0 0 0 0 ' + compressor + ' 0 0 0 0 0'             #valves change postion throughout the run, allows for a position input everytime the node is pinged
    nodeRawData = subprocess.check_output(nodeString, shell = True)               #Ping Cassette Node and save the message
    match       = re.search('(?<=hpPressure: )(.*)', nodeRawData)         #regular expression search for load cell counts
    hpCounts    = float(match.group(0))

    return hpCounts

def GetHighPressure(hpCounts):

    highPressure = 0.0332 * hpCounts - 12.0

    return highPressure

def PressurizeTank(hpCounts, highPressure, compressor):
    hpCounts     = PingCentrifugeNode(compressor)
    highPressure = GetHighPressure(hpCounts)
    print('Pressurizing High Pressure System')
    while highPressure <= 90.0:
        compressor = '1'
        hpCounts     = PingCentrifugeNode(compressor)
        highPressure = GetHighPressure(hpCounts)
    compressor = '0'
    hpCounts   = PingCentrifugeNode(compressor)

    return hpCounts, highPressure, compressor
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
######## Timing Functions #########
def getCurrentTime(timing):
    #timing variable will be an array that contains various timing parameters
    #timing
    #timing[0]: t0, start of the test
    #timing[1]: lastTime1, previous timing iteration, used for calculating intervals
    #timing[2]: lastTime2, previous timing iteration, used for calculating intervals
    #timing[3]: lastTime3, previous timing iteration, used for calculating intervals
    #timing[4]: now, the current time
    #timgin[5]: dt1, interval duration between now and lastTime1
    #timing[6]: dt2, interval duration between now and lastTime2
    #timing[7]: dt3, interval duration between now and lastTime3
    #Get current time
    timing[4] = time.time() - timing[0] #get current time
    timing[5] = timing[4] - timing[1]   #get dt1 
    timing[6] = timing[4] - timing[2]   #get dt2
    timing[7] = timing[4] - timing[3]   #get dt2
    return timing

def setLastTime(timing, intNum):
    #timing variable will be an array that contains various timing parameters
    #timing
    #timing[0]: t0, start of the test
    #timing[1]: lastTime1, previous timing iteration, used for calculating intervals
    #timing[2]: lastTime2, previous timing iteration, used for calculating intervals
    #timing[3]: lastTime3, previous timing iteration, used for calculating intervals
    #timing[4]: now, the current time
    #timgin[5]: dt1, interval duration between now and lastTime1
    #timing[6]: dt2, interval duration between now and lastTime2
    #timing[7]: dt3, interval duration between now and lastTime3
    #set lastTime(Num) to now
    timing[intNum] = timing[4]

    return timing
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
########## PID Functions ##########
def PID(setPoint, ref, K, error, errBuffer, timing, output, freqPID, APS):
    #PID controller for the motor
        #setpoint: point we're trying to get to. In this case cassette internal pressure
        #     ref: actual value that the setpoint is compared to, In this case, actual pressure
        #       K: array of current PID constants
        #  timing: array of timing parameters
        # freqPID: target rate to run PID function
    #Main
    #uses dt1
    timing = getCurrentTime(timing)

    #want to run the PID Controller over a regular interval. This allows to simplify the math and get rid of the divide operator on the de(t)/dt term
    if timing[5] >= freqPID: #check if timing as hit the PID interval
        #Calculate Error
        error[0] = setPoint - ref  #current error
        error[1] = error[1] + error[0] #error sum
        dErr     = error[0] - error[2]

        #PID control Output
        output = K[0]*error[0] + K[1]*error[1] + K[2]*dErr

        #remeber some variables
        error[2] = error[0]
        timing   = setLastTime(timing, 1)

        #Update Error Buffer
        errBuffer = AppendBuffer(error[0], errBuffer)

        print('PresSet = ' + "{0:.1f}".format(setPoint) + 'mmHg, Pres = ' + "{0:.1f}".format(ref) + 'mmHg, Counts = ' + APS +', t = ' + "{0:.2f}".format(timing[4]) + 's, dt = ' + "{0:.2f}".format(timing[5]) + 's')
    
    return output, timing, error, errBuffer

def SetPIDParams(setPoints):
    #define the PID params for each setPoint in the test
    l          = len(setPoints)   #number of columns in the PID array... P, I, & D value for each set point
    PID_Params = np.zeros([l+1, 3]) #initialize a lx3 array for the PID constants

    for i in range(l):
        #loop through the setPoints and determine the PID constants for that setpoint
        if setPoints[i] == -350.0:                
            PID_Params[i, :] = np.array([0.215, 0 ,0])

        elif setPoints[i] == -250.0:             
            PID_Params[i, :] = np.array([0.165, 0, 0])

        elif setPoints[i] == -150.0:              
            PID_Params[i, :] = np.array([0.145, 0, 0])

        elif setPoints[i] == -50:               
            PID_Params[i, :] = np.array([0.13, 0, 0])

        elif setPoints[i] == 0.0:                
            PID_Params[i, :] = np.array([0.125, 0,0 ]) #take some adjustment

        elif setPoints[i] == 200:               
            PID_Params[i, :] = np.array([0.11, 0.0, 0])

        elif setPoints[i] == 400:            
            PID_Params[i, :] = np.array([0.09, 0, 0])

        elif setPoints[i] == 600:            
            PID_Params[i, :] = np.array([0.075, 0, 0]) #could be better

        elif setPoints[i] == 800 :            
            PID_Params[i, :] = np.array([0.06, 0, 0])

        elif setPoints[i] == 1000 :            
            PID_Params[i, :] = np.array([0.05, 0, 0])

    return PID_Params
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
######### Ring Buffer #############
#need to comment this section to better understand what is happening here
#needs comments
    #class (better understand python classes)
    #self  (better understand self in python)
#can probably write the AppendBuffer() and GetBufferAvg() functions better
#do those rewrite after learning more about python classes and (self)
class RingBuffer:
   def __init__(self, size):
       self.data = [None for i in range(size)]

   def append(self, x):
       self.data.pop(0)
       self.data.append(x)

   def get(self):
       return self.data

def AppendBuffer(val, Buffer):
    Buffer.append(val)
    return Buffer

def GetBufferAvg(Buffer):
    mean = np.mean(Buffer.get())
    return mean
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
######### File Functions ##########
def GetTestInfo():
    executor         = raw_input('enter name and press "Enter": ')
    date             = raw_input('enter date (MM/DD/YYYY) and press "Enter": ')
    machine          = raw_input('enter machine number (1J000XX) and press "Enter": ')
    dlogNum          = raw_input('enter machine dlog number (XXXXXX.dlog) and press "Enter": ')
    presTransCalNum  = raw_input('enter pressure transducer calibration serial number (XX-XXXXXXX) and press "Enter": ')
    presTransCalDate = raw_input('enter pressure transducer calibration due date (MM/DD/YYYY) and press "Enter": ')
    cassetteLotNum   = raw_input('enter cassette lot number and press "Enter": ')
    posGainPsi       = raw_input('enter machine positive APS gain and press "Enter": ')
    negGainPsi       = raw_input('enter machine negative APS gain and press "Enter": ')
    posGain          = float(posGainPsi) * 51.7149 #[mmHg/Psi]
    negGain          = float(negGainPsi) * 51.7149 #[mmHg/Psi]

    return executor, date, machine, dlogNum, presTransCalNum, presTransCalDate, cassetteLotNum, posGain, posGainPsi, negGain, negGainPsi

def InitializeGeneralDataFile(executor, date, machine, dlogNum, presTransCalNum, presTransCalDate, cassetteLotNum, posGain, posGainPsi, negGain, negGainPsi):
    #Intialize Files
        #general data file

    generalData = open('pressureData.txt', 'a+') #start new file or append to an exsisting. Appending to an exsisting file is undersireable   
    #Write Header Information
        #general data file
    generalData.write(executor + ', ' + date + ', VE-EQ-0024-B, APS Accuracy Test\n') #Name, Date, Test #
    generalData.write(machine + ', ' + dlogNum + '\n')                                #Machine #, dlog #
    generalData.write('Pressure Transducer Info, ' + presTransCalNum + ', ' + presTransCalDate +'\n')
    generalData.write('Cassette Lot Number, ' + cassetteLotNum + '\n')
    generalData.write('Positive Gain [mmHg/count], ' + str(posGain) + ', Positive Gain [Psi/count], ' + posGainPsi + '\n')
    generalData.write('Negative Gain [mmHg/count], ' + str(negGain) + ', Negative Gain [Psi/count], ' + negGainPsi + '\n')
    generalData.write('Time [s], Reference Pressure [mmHg], APS Counts, Set Point, Take Cal Data, Door Position, Pump Speed\n')   #Column Information
    
    return generalData

def InitializeCalDataFile(executor, date, machine, dlogNum, presTransCalNum, presTransCalDate, cassetteLotNum, posGain, posGainPsi, posOffset, negGain, negGainPsi, negOffset):
    #Intialize Files
        #calibration specific data
    calData = open('calData.txt', 'a+')     
    #Write Header Information
        #calibration data file
    calData.write(executor + ', ' + date + ', VE-EQ-0024-B, APS Accuracy Test\n') #Name, Date, Test #
    calData.write(machine + ', ' + dlogNum + '\n')                                #Machine #, dlog #
    calData.write('Pressure Transducer Info, ' + presTransCalNum + ', ' + presTransCalDate +'\n')
    calData.write('Cassette Lot Number, ' + cassetteLotNum + '\n')
    calData.write('Positive Gain [mmHg/count], ' + str(posGain) + ', Positive Gain [Psi/count], ' + posGainPsi + '\n')
    calData.write('Positive Offset [mmHg], ' + str(posOffset) + '\n')
    calData.write('Negative Gain [mmHg/count], ' + str(negGain) + ', Negative Gain [Psi/count], ' + negGainPsi + '\n')
    calData.write('Negative Offset [mmHg], ' + str(negOffset) + '\n')
    calData.write('Reference Pressure [mmHg], APS Counts\n')

    return calData

def WriteGeneralData(generalData, time, pressure, APS, setPoint, writeCal, doorPos, pumpSpeed):
    #write data to the general data file
    #time, pressure, and setPoint come in as floats. APS counts is a string

    generalData.write(str(time) + ', ' + str(pressure) + ', ' + APS + ' ,' +  str(setPoint) + ', ' + str(writeCal) + ', ' + doorPos + ', ' + str(pumpSpeed) + '\n')

    return generalData

def WriteCalData(calData, calPointsRef, calPointsAPS, testLen, numCalPoints):
    #write data to the calibration data file
    #tpressure comes in as floats. APS counts is a string
    for k in range(testLen):
        for l in range(numCalPoints):
            calData.write(str(calPointsRef[k,l]) + ', ' + str(calPointsAPS[k, l]) + '\n')

    return calData
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##### Test Protocol Functions #####
def TakeCalPoints(APSCounts, refPres, timing, calPointInt, i, j, numCalPoints, testLen, writeCal, calPointsAPS, calPointsRef):
    #write a calibration point every second
    #main
    if writeCal == True:      #System has reached steady state and is ready to take cal points?
        #uses dt2
        timing = getCurrentTime(timing)

        if timing[6] > calPointInt: #Check to see if 1 second has passed? Take a cal point every second 
            if j < numCalPoints:    #check if system has reached the end of the calpoints Arrays, columns

                calPointsAPS[i, j] = np.int(APSCounts) #update the APS calpoint array   
                calPointsRef[i, j] = refPres   #update the ref calpoint array
                j = j + 1                      #iterate to the next cal point
            
            else: #we've taken the last calibration point
            
                print('Current Cal Data')
                print(calPointsAPS)
                writeCal = False #flip state on write cal since the system is moving to a new calibration point            
                j = 0            #reset calibration counter

                if i < testLen:  #check if system has reached the end of the calpoints Arrays, rows

                    i = i + 1    #move to the next setpoint
            #uses dt2          
            setLastTime(timing, 2)

    return i, j, writeCal, calPointsAPS, calPointsRef

def CheckTestEnd(i, testLen, test):
    if i == testLen:
        test = False

    return test

def CheckSteadyState(ErrorBandSS, timing, checkSSInterval, errBuffer, writeCal):
    #wait for 5 seconds to elapse and check average error
    #uses dt3
    timing = getCurrentTime(timing)
    #Main
    if timing[7] > checkSSInterval: #has the check interval passed?      
        errMean = GetBufferAvg(errBuffer)  #calculate average buffer error
        print('Mean error = ' + "{0:.1f}".format(errMean) + '[mmHg]')
        if np.abs(errMean) < ErrorBandSS:  #check if average error is less than 10 (value subject to change based on testing)
            writeCal = True              #set the system to write cal datapoints

        #uses dt3
        timing = setLastTime(timing, 3)

    return timing, writeCal
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

##################################################
################## SETUP #########################
#####################i#############################
#File Setup
executor, date, machine, dlogNum, presTransCalNum, presTransCalDate, cassetteLotNum, posGain, posGainPsi, negGain, negGainPsi = GetTestInfo()
generalData = InitializeGeneralDataFile(executor, date, machine, dlogNum, presTransCalNum, presTransCalDate, cassetteLotNum, posGain, posGainPsi, negGain, negGainPsi) #setup the data file

#Generate Set Points and corresponding PID parameters
setPoints = PressureTestRange()     #Randomize Pressure Point order
K         = SetPIDParams(setPoints) #set the PID parameters
testLen   = len(setPoints)          #Get number of pressure points that are being tested

########## variables ###############
#timing variable will be an array that contains various timing parameters
#timing
#timing[0]: t0, start of the test
#timing[1]: lastTime1, previous timing iteration, used for calculating intervals
#timing[2]: lastTime2, previous timing iteration, used for calculating intervals
#timing[3]: lastTime3, previous timing iteration, used for calculating intervals
#timing[4]: now, the current time
#timgin[5]: dt1, interval duration between now and lastTime1
#timing[6]: dt2, interval duration between now and lastTime2
#timing[7]: dt3, interval duration between now and lastTime3
timing    = np.zeros(8)
#~~~~~#
#PID
    #Error
        #Error variable will be an array that contains various error parameters
error = np.zeros(3)
#[0] => current error
#[1] => error Sum: sum of error over a period of time
#[2] => last error: error from the previous iteration
    #Sample Rate
freqPID = 0.30
        #change the constants to match the constants for the first setpoint
#SetTunings(pidParams[0,0] , pidParams[0,1], pidParams[0,2])
    #Output
output  = 0.0   #PID motor control output

#Error FIFO Buffer setup
bufSize      = 20                   #Size fo the FIFO buffer
bufSize2     = 50
errBuffer    = RingBuffer(bufSize)  #error FIFO buffer
countsBuffer = RingBuffer(bufSize2) #error FIFO buffer
presBuffer   = RingBuffer(bufSize2) #error FIFO buffer
intErr       = 200.0                #Intial Error
intCounts    = 0    
intPres      = 0
for j in range(bufSize):          #initialize the buffer with error values outside of the steady state bands
    errBuffer    = AppendBuffer(intErr, errBuffer) #fill buffer with the intial error
for j in range(bufSize2):
    countsBuffer = AppendBuffer(intCounts, countsBuffer) 
    presBuffer = AppendBuffer(intPres, presBuffer) 
#~~~~~#
#Protocal Variables
highPressure    = 0.0 #Psi
compressor      = '0'
hpCounts        = 0
creepCheck      = 300.0                            #time letting the cassette sit loaded with nothing happening. want to look at preload creep
primeDuration   = 60.0                              #duration of cassette prime
checkSSInterval = 5.0                               #interval betweem steady state checks
calPointInt     = 2.0                               #interval between cal points
test            = True                              #run the test while loop
return0         = True                              #run the return to 0 mmHg while loop
i               = 0                                 #setPoints counter
j               = 0                                 #calibration points counter
writeCal        = False                             #protocol has reached a steady state and is ready to save cal data
numCalPoints    = 10                                #Number of calibration points to take at each pressure point
calPointsAPS    = np.zeros((testLen, numCalPoints)) #Array to store calibration APS data prior to writing it
calPointsRef    = np.zeros((testLen, numCalPoints)) #Array to store calibration APS data prior to writing it
ErrorBandSS     = 5.0                               #Error bound for the primary section of the test.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

##################################################
################### MAIN #########################
##################################################

#######Pre Test Setup#####
#set nodes to operating mode
IntializeNodes()
#open valves to allow for loading the cassette and pump
valvePositions = OpenValves()
SetPumpRacePosition(0, 0)

#pressuize Tank
hpCounts, highPressure, compressor = PressurizeTank(hpCounts, highPressure, compressor)

#get the latch position
APS, latch = PingCassetteNode(valvePositions) 

#Check if the door is closed and prompt the user to open the door
print('Test Start')
while latch == '1': 
    print('Open Door and Load Cassette to Continue')
    time.sleep(2.5)
    APS, latch = PingCassetteNode(valvePositions)

#start logging data while the door is open and the cassette is being loaded. 
#Set t0
print('Load Cassette')
timing[0] = time.time()
timing    = getCurrentTime(timing)
while latch == '2':
    #print('waiting for door, t = ' + "{0:.2f}".format(timing[4]))
    timing     = getCurrentTime(timing) #Start logging time
    pressure   = GetRefPres()
    APS, latch = PingCassetteNode(valvePositions) 
    WriteGeneralData(generalData, timing[4], pressure, APS, 0.0, writeCal, latch, output) #write data

#wait for a period of time to look at creep/preload on the soft cassette.
print('Cassette Loaded')
SetPumpRacePosition(0, 1)
timing = setLastTime(timing, 1) #reset lastTime 1 (timing [1]) prior to this timing check while loop
timing = getCurrentTime(timing)
while timing[5] < creepCheck:
    print('creep check, t = ' + "{0:.2f}".format(timing[4]))
    timing       = getCurrentTime(timing)
    APS, latch   = PingCassetteNode(valvePositions)
    pressure     = GetRefPres()
    countsBuffer = AppendBuffer(float(APS), countsBuffer)
    presBuffer   = AppendBuffer(pressure, presBuffer)
    WriteGeneralData(generalData, timing[4], pressure, APS, 0.0, writeCal, latch, output) #write data

#Calculate Offset
countsMean = GetBufferAvg(countsBuffer)
presMean   = GetBufferAvg(presBuffer)
posOffset  = 0 - posGain * countsMean #assume pressure prior to pump starting is 0mmHg, Calculate Offset from that
negOffset  = 0 - negGain * countsMean #assume pressure prior to pump starting is 0mmHg, Calculate Offset from that

#Prime Cassette
print('Priming Cassette')
timing = setLastTime(timing, 1) #reset lastTime 1 (timing [1]) prior to this timing check while loop
timing = getCurrentTime(timing)
while timing[5] < primeDuration:
    print('Prime, t = ' + "{0:.2f}".format(timing[4]))
    timing     = getCurrentTime(timing)
    pressure   = GetRefPres()
    APS, latch = PingCassetteNode(valvePositions)
    WriteGeneralData(generalData, timing[4], pressure, APS, 0.0, writeCal, latch, output) #write data
    SetPumpSpeed(-65)
#~~~~~~~~~~~~~~~~~~~~~~~~#

####Pressure Protocol#####
SetPumpSpeed(0)
print('pump paused')
valvePositions = CloseValves()
print('valves closed')
print(setPoints)
print('Setpoint = ' + str(setPoints[i]))

while test == True:
    #Get Values
    APS, latch                        = PingCassetteNode(valvePositions)
    pressure                          = GetRefPres()
    output, timing, error, errBuffer  = PID(setPoints[i], pressure, K[i, :], error, errBuffer, timing, output, freqPID, APS)
    
    #Change speed based on PID output
    SetPumpSpeed(output)

    #Write General Pressure Data
    generalData = WriteGeneralData(generalData, timing[4], pressure, APS, setPoints[i], writeCal, latch, output) #write to the general data file
   
    #Write Calibration Data Check
    timing, writeCal                           = CheckSteadyState(ErrorBandSS, timing, checkSSInterval, errBuffer, writeCal)
    i, j, writeCal, calPointsAPS, calPointsRef = TakeCalPoints(APS, pressure, timing, calPointInt, i, j, numCalPoints, testLen, writeCal, calPointsAPS, calPointsRef) #Write Cal data after reaching steady state, change setpoint, change test state
    test                                       = CheckTestEnd(i, testLen, test)

SetPumpSpeed(0)
print('Test End')
#valvePositions = OpenValves()

#stuff finalizing the test
print('Main While Loop Exited')
generalData.close()   #close the general pressure data file
writeCal = False      #make sure writeCal is set to False
#process changes 

K           = np.array([0.125, 0, 0])
ErrorBandSS = 30.0                  #open up the error band to allow the system reach a steady state faster

while return0 == True:       #return cassette pressure to 0 mmHg so that the system doesn't leave a cassette pressurized
    #get values
    pressure                          = GetRefPres()
    output, timing, error, errBuffer  = PID(0.0, pressure, K, error, errBuffer, timing, output, freqPID, APS)

    #set pump speed
    SetPumpSpeed(output)

    #check for steady state
    timing, writeCal = CheckSteadyState(ErrorBandSS, timing, checkSSInterval, errBuffer, writeCal)
    if writeCal == True:  #check for change in writeCal
        return0 = False #on change in state exit while loop

print('Test Ramp Down Loop Exited')
SetPumpSpeed(0)                   #turn off pump
OpenValves()                      #close the pinch valves
SetPumpRacePosition(0, 0)
calData = InitializeCalDataFile(executor, date, machine, dlogNum, presTransCalNum, presTransCalDate, cassetteLotNum, posGain, posGainPsi, posOffset, negGain, negGainPsi, negOffset)        
calData = WriteCalData(calData, calPointsRef, calPointsAPS, testLen, numCalPoints)
calData.close()                          #close the general pressure data file
print('#######################')
print('#### TEST COMPLETE ####')
print('#######################')
#~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
