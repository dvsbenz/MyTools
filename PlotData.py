##############################################
##############Libraries#######################
##############################################
import numpy as np              #import numpy library
from numpy import genfromtxt
from numpy import loadtxt
import scipy as sp              #import scipy library
from scipy import stats   
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import matplotlib.pyplot as plt #import matplotlib
import fnmatch                  #import fnmatch libary, allows for unix shell-style wildcards
import os, os.path              #import os library, usefull functions for pathnames
import re                       #import re library, provides regular expression matching similar to perl
import random as rn
import csv      

##############################################
################Functions#####################
##############################################
####### File System Movement Functions #######
def GetFileNames():
	#get the name of the files to loop through
	directories = [] #initialize array
	for fname in os.listdir():    #go through everything in the base directory
		if fname.endswith('.py'): #check for anything that ends with .py and exclude it
			continue              
		else:
			directories.append(fname) #grab only the directories that have data files

	return directories

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
######## Data file Handling Functions ########
def LoadData(fileName): 
	#load Data from text file
	#first 2 lines are header info
		#name/test/test sub type/date
		#Time [s], Reference Pressure [mmHg], PID SetPoint [mmHg], APS Counts
	data_temp = np.genfromtxt(fileName, skip_header = 7, dtype = float, delimiter = ",")
	#get number of data points
	arryShape = data_temp.shape;
	colmLen   = arryShape[0] - 1
	#split 2d array into vectors for each data column
	time      = data_temp[0:colmLen, 0] #[s]
	pres      = data_temp[0:colmLen, 1] #reference pressure [mmHg]
	APS       = data_temp[0:colmLen, 2] #current set point [mmHg]
	SP        = data_temp[0:colmLen, 3] #APS loadcell counts
	DP        = data_temp[0:colmLen, 5] #Door Position Output
	pumpSpeed = data_temp[0:colmLen, 6] #Pump Speed Output

	data_temp = np.genfromtxt(fileName, skip_header = 7, dtype = str, delimiter = ",")
	SS        = data_temp[0:colmLen, 4]

	return time, pres, APS, SP, SS, DP, pumpSpeed

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
############## Ploting Functions #############
def PlotData(xaxis, yaxis, lines, axLabels, linLabels, title):
	#plot the data
	xlims, ylims = GenPlotLims(xaxis, yaxis) #automatically generate the axis limits
	plt.figure()
	for i in range(len(xaxis)):              #add
		plt.plot(xaxis[i], yaxis[i], lines[i], linewidth = 1.0, mew = 1.0, label = linLabels[i])
	plt.title(title)
	plt.xlabel(axLabels[0])
	plt.ylabel(axLabels[1])
	plt.axis([xlims[0], xlims[1], ylims[0], ylims[1]])
	plt.legend(loc = 'best')

def SubPlotData(xaxis1, xaxis2, yaxis1, yaxis2, lines, axLabels, linLabels, title):
	xlims1, ylims1 = GenPlotLims(xaxis1, yaxis1)
	xlims2, ylims2 = GenPlotLims(xaxis2, yaxis2)
	plt.figure()
	plt.subplot(3, 1, 1)
	for i in range(len(xaxis1)):
		plt.plot(xaxis1[i], yaxis1[i], lines[0][i], linewidth = 1.0, mew = 1.0, label = linLabels[0][i])
	plt.title(title[0])
	plt.xlabel(axLabels[0][0])
	plt.ylabel(axLabels[0][1])
	plt.axis([xlims1[0], xlims1[1], ylims1[0], ylims1[1]])
	plt.legend(loc = 'best')

	plt.subplot(3, 1, 2)
	for i in range(len(xaxis2)):
		plt.plot(xaxis2[i], yaxis2[i], lines[1][i], linewidth = 1.0, mew = 1.0, label = linLabels[1][i])
	plt.title(title[1])
	plt.xlabel(axLabels[1][0])
	plt.ylabel(axLabels[1][1])
	plt.axis([xlims2[0], xlims2[1], ylims2[0], ylims2[1]])
	plt.legend(loc = 'best')

def GenPlotLims(xaxis, yaxis):
	#x axis lims
	xlimsax = np.zeros((2,len(xaxis))); xlims = np.zeros(2)
	for i in range(len(xaxis)):
		xlimsax[0,i] = np.amin(xaxis[i]); #lower limits
		xlimsax[1,i] = np.amax(xaxis[i]); #upper limits
	xlims[0] = np.amin(xlimsax[0,0:len(xaxis)]); 
	xlims[1] = np.amax(xlimsax[1,0:len(xaxis)]);

	#y axis lims
	ylimsax = np.zeros((2,len(yaxis))); ylims = np.zeros(2)
	for i in range(len(yaxis)):
		ylimsax[0,i] = np.amin(yaxis[i]); #lower limits
		ylimsax[1,i] = np.amax(yaxis[i]); #upper limits
	ylims[0] = np.amin(ylimsax[0,0:len(yaxis)]); 
	ylims[1] = np.amax(ylimsax[1,0:len(yaxis)]);

	#extend beyond 10% of y range
	dy  = ylims[1]-ylims[0]
	ext = dy*0.05
	ylims[0] = ylims[0] - ext; ylims[1] = ylims[1] + ext
	return xlims, ylims
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
############### Timing Functions #############
def GetTimeInd(time, sec):
	ind = 0
	for i in range(len(time)):
		if time[i] > sec:
			ind = i
			break
	return i
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

##############################################
#####################MAIN#####################
##############################################
#number of tests to loop through
fileNames  = GetFileNames()
preLoad    = np.zeros([len(fileNames)])
creepSlope = np.zeros([len(fileNames)])

decayTimes = np.array([10, 30, 60, 90, 120, 150, 180])
APSDecay   = np.zeros(len(decayTimes))
APSDecayt  = np.zeros(len(decayTimes))

for i in range(len(fileNames)):
	####### Get Data ################
	time, pres, APS, SP, SS, DP, pumpSpeed = LoadData(fileNames[i])
	#################################
	for j in range(len(DP)):
		if DP[j] == 2  and DP[j+1] == 1:
			latchInd = j
			break
	tLatch = time[latchInd]

	for j in range(len(pumpSpeed)):
		if pumpSpeed[j] > 0.0 or pumpSpeed[j] < 0.0:
			indDec = j-1
			break
	indDec = GetTimeInd(time, time[indDec] - 60)

	for j in range(len(decayTimes)):
		APSDecay[j]  = APS[GetTimeInd(time, tLatch + decayTimes[j])]
		APSDecayt[j] = time[GetTimeInd(time, tLatch + decayTimes[j])]


	# ############ raw Data ##########
	xaxis1    = [time]
	xaxis2    = [time, time]
	yaxis1    = [pres]
	yaxis2    = [APS, DP]
	lines     = [['b--'],['r--', 'g--']]
	axLabels  = [['time [s]', 'Pressure [mmHg]'], ['time [s]', 'counts']]
	linLabels = [['Reference Pressure [mmHg]'], ['APS Counts', 'Door Sensor Change']]
	title1    = 'Reference Pressure Plot'
	title2    = 'APS Counts'
	title     = [title1, title2]
	SubPlotData(xaxis1, xaxis2, yaxis1, yaxis2, lines, axLabels, linLabels, title)
	# ###################################

	#xaxis     = [time, time[latchInd], time[indDec], APSDecayt]
	#yaxis     = [APS, APS[latchInd], APS[indDec], APSDecay]
	#lines     = ['b--', 'rx', 'rx', 'kx']
	#axLabels  = ['time [s]', 'counts']
	#linLabels = ['APS counts', 'Latch Switch', 'Pump Start', 'Decay Measurements']
	#title     = 'APS Counts'
	#PlotData(xaxis, yaxis, lines, axLabels, linLabels, title)

	#xaxis     = [time]
	#yaxis     = [APS]
	#lines     = ['b--']
	#axLabels  = ['time [s]', 'counts']
	#linLabels = ['APS counts']
	#title     = 'APS Counts'
	#PlotData(xaxis, yaxis, lines, axLabels, linLabels, title)



	# ############# Plot Data and Error ############
	# #Plot1 = data
	# xaxis1           = [time, time]
	# yaxis1           = [pres, APSDual]
	# #Plot2 = percent error
	# xaxis2           = [time, errT, errT]
	# yaxis2           = [error, posErr, NegErr]
	# #Plot3 = error difference [mmHg]
	# xaxis3           = [time, errT, errT]
	# yaxis3           = [diff, ErrDiffP, ErrDiffN]
	# lines            = [['b-', 'r--'], ['b--', 'g--', 'r--'], ['b--', 'g--', 'r--']]
	# axLabels         = [['time [s]', 'Pressure [mmhg]'], ['time [s]', 'Error [%]'],  ['time [s]', 'Error [mmHg]']]
	# linLabels        = [['Calibrated Pressure Reference [mmHg]', 'APS [mmHg]', 'Curve Type'], ['Percent Error', 'Pos Err Spec', 'Neg Err Spec'], ['Error [mmHg]', 'Percent Full Scale Error [mmHg]', 'Percent Full Scale Error [mmHg]']]
	# titleTop         = 'Cassette ' + str(i + 1) + ' Dual Curve Error'
	# titleMiddle      = 'Percent Error'
	# titleBot         = 'Full Scale Error'
	# title            = [titleTop, titleMiddle, titleBot]
	# SubPlotData(xaxis1, xaxis2, xaxis3, yaxis1, yaxis2, yaxis3, lines, axLabels, linLabels, title)
	# ###############################################
plt.show()