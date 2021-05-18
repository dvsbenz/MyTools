##############################################
##############Libraries#######################
##############################################
import numpy as np              #import numpy library
from numpy import genfromtxt
from numpy import loadtxt
import scipy as sp              #import scipy library
from scipy import stats
from scipy.stats import norm   
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
	data_temp = np.genfromtxt(fileName, skip_header = 9, dtype = float, delimiter = ",")
	#get number of data points
	arryShape = data_temp.shape;
	colmLen   = arryShape[0] - 1
	#split 2d array into vectors for each data column
	pres      = data_temp[0:colmLen, 0] #reference pressure [mmHg]
	APS       = data_temp[0:colmLen, 1] #APS loadcell counts

	return pres, APS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
############## Ploting Functions #############
def PlotData(xaxis, yaxis, lines, axLabels, linLabels, title):
	#plot the data
	xlims, ylims = GenPlotLims(xaxis, yaxis) #automatically generate the axis limits
	plt.figure()
	for i in range(len(xaxis)):              #add
		plt.plot(xaxis[i], yaxis[i], lines[i], linewidth = 1.0, mew = 1.0, label = linLabels[i])
	plt.tick_params(labelsize = 18)
	plt.title(title, fontsize = 22)
	plt.xlabel(axLabels[0], fontsize = 20)
	plt.ylabel(axLabels[1], fontsize = 20)
	plt.axis([xlims[0], xlims[1], ylims[0], ylims[1]])
	#plt.legend(loc = 'best')

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
########### Regression Functions #############
def linearRegression(xaxis, yaxis, axLabels, title, plotData):
	#order Data
	xaxis, yaxis = OrderData(xaxis, yaxis)
	#perform least squares linear regression
	linReg      = stats.linregress(xaxis, yaxis) #slope, intercept, r-value, p-value, stderr
	linRegCurve = xaxis*linReg[0]+linReg[1]      #y=mx+b => y=linReg[0]+linReg[1]
	# print('Linear Regression')
	# print('Cal Eqn = ' + str.format('{0:.5f}', linReg[0]) + '*V + ' + str.format('{0:.5f}', linReg[1]))
	# print('R^2 = ' + str.format('{0:.5f}', linReg[2]) + '\n')
	# print('')
	#setup plot
	if plotData == True:
		pxaxis    = [xaxis, xaxis]
		pyaxis    = [yaxis, linRegCurve]
		lines     = ['r.', 'b--']
		linLabels = ['data', 'Regression']
		PlotData(pxaxis, pyaxis, lines, axLabels, linLabels, title)
		xann      = np.amax(xaxis)-(np.amax(xaxis)*0.3)
		yann      = np.amax(yaxis)-(np.amax(yaxis)*0.1)
		plt.annotate('y = ' + str.format('{0:.5f}', linReg[0]) + 'x + ' + str.format('{0:.5f}',linReg[1]) + '\n' + 'r^2 = ' + str.format('{0:.5f}', linReg[2]), xy =(xann, yann), xytext = (xann, yann))

	return linReg
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
############ Array Manipulation ##############
def OrderData(xaxis, yaxis):
	xaxis, yaxis = zip(*sorted(zip(xaxis, yaxis)))
	xaxis = np.array(xaxis)
	yaxis = np.array(yaxis)
	return xaxis, yaxis

def GetPostiveNeg(xaxis, yaxis):
	totalLen = len(yaxis)
	count = 0
	for i in range(len(yaxis)):
		if yaxis[i] <= 0:
			count = count +1
	yaxisNeg = np.zeros(count)
	xaxisNeg = np.zeros(count)

	count = 0
	for i in range(len(yaxis)):
		if yaxis[i] >= 0:
			count = count + 1
	yaxisPos = np.zeros(count)
	xaxisPos = np.zeros(count)

	count = 0
	for i in range(len(yaxis)):
		if yaxis[i] <= 0:
			yaxisNeg[count] = yaxis[i]
			xaxisNeg[count] = xaxis[i]
			count = count + 1

	count = 0
	for i in range(len(yaxis)):
		if yaxis[i] >= -0:
			yaxisPos[count] = yaxis[i]
			xaxisPos[count] = xaxis[i]
			count = count + 1

	return xaxisPos, yaxisPos, xaxisNeg, yaxisNeg
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#################################################################################
############################### Plot Cal Eqns ###################################
#################################################################################
#number of tests to loop through
fileNames = GetFileNames()
#initialize variables
linReg     = np.zeros([len(fileNames), 5])
linRegPos  = np.zeros([len(fileNames), 5])
linRegNeg  = np.zeros([len(fileNames), 5])
countsAll  = [[]]*len(fileNames)
counts1000 = [[]]*len(fileNames)
counts350  = [[]]*len(fileNames)
preloadDelta = np.array([-136, -9, -101, -8, 22, -209, 18, -58, -86, -136, -235, 12, 21, 104, -277, -81, -102, -6, -290, -146, -26, -15, 7, 6, -38, -10, -128, -139, -97, 51, -46, 9, -42, 103, -18, 33, -82, -197, -86, -183, 16, -136, -104, -84, -156, -144, 27, -117, -44, 12, -128, -144, -269, 54, -61, -65, -64, -113, -10, 22])


for i in range(len(fileNames)):
	####### Get Data ################
	pres, APS                        = LoadData(fileNames[i])   #full data set
	countsAll[i]                     = APS                      #logging counts to see if either of extreme of the ADC register is pinged
	APSPos, presPos, APSNeg, presNeg = GetPostiveNeg(APS, pres) #sort by positive and negative pressure values
	#################################

	########## Collect Counts #######
	#1000mmhg
	for j in range(len(pres)):
		if pres[j] < 1000 + 15 and pres [j] > 1000 - 15:
			counts1000[i].append(APS[j])
		if pres[j] < -350 + 15 and pres[j] > -350 - 15:
			counts350[i].append(APS[j])
	
	####### linear regression #######
	axLabels        = ['Node ADC Counts', 'Pressure [mmHg]']
	title           = 'Soft Cassette Housing V4, Test ' + str(i+1) + ', Calibration Curve'
	linReg[i, :]    = linearRegression(APS, pres, axLabels, title, plotData = False)
	linRegPos[i, :] = linearRegression(APSPos, presPos, axLabels, title, plotData = False)
	linRegNeg[i, :] = linearRegression(APSNeg, presNeg, axLabels, title, plotData = False)
	#################################

	#### Plot all 3 Regressions #####
	xaxis     = [APS, APS, APSPos, APSNeg]
	yaxis     = [pres, linReg[i, 0]*APS + linReg[i, 1], linRegPos[i, 0]*APSPos + linRegPos[i, 1], linRegNeg[i, 0]*APSNeg + linRegNeg[i, 1]]
	lines     = ['r.', 'b--', 'g--', 'r--']
	axLabels  = ['ADC Counts', 'Gain [mmHg/count]']
	linLabels = ['Raw Data', 'Full Cal Curve', 'Postivie Cal Curve', 'Negative Cal Curve']
	title     = 'Cassette ' + str(i+1) + ' Pos & Neg Cal Curve Comparison'
	PlotData(xaxis, yaxis, lines, axLabels, linLabels, title)
	# xann      = np.amax(xaxis)-(np.amax(xaxis)*0.3)
	# yann      = np.amax(yaxis)-(np.amax(yaxis)*0.1)
	# plt.annotate('y = ' + str.format('{0:.5f}', linRegPos[0]) + 'x + ' + str.format('{0:.5f}',linRegPos[1]) + '\n' + 'r^2 = ' + str.format('{0:.5f}', linRegPos[2]), xy =(xann, yann), xytext = (xann, yann))
	#################################


#Gather Stats and plot enesemble of basic cal curves
numReg    = len(fileNames)
counts    = np.array([800, 2300])
slopes    = np.zeros(numReg)
slopesPos = np.zeros(numReg)
slopesNeg = np.zeros(numReg)
yInt      = np.zeros(numReg)
yIntPos   = np.zeros(numReg)
yIntNeg   = np.zeros(numReg)
r2        = np.zeros(numReg)
r2Pos     = np.zeros(numReg)
r2Neg     = np.zeros(numReg)
xaxis     = [[]]*(numReg)
yaxis     = [[]]*(numReg)
lines     = [[]]*(numReg)
linLabels = [[]]*(numReg) 
colors    = ['r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-', 'r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-', 'r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-','r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-', 'r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-', 'r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-','r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-', 'r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-', 'r--','r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-', 'r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-', 'r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-','r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-', 'r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-', 'r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-','r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-', 'r--', 'b--', 'g--', 'y--', 'm--', 'c--', 'k--', 'r-', 'b-', 'g-', 'r--']
for i in range(numReg):
	xaxis[i]      = counts
	linLabels[i]  = 'Curve ' + str(i+1)
	slopes[i]     = linReg[i, 0]
	slopesPos[i]  = linRegPos[i, 0]
	slopesNeg[i]  = linRegNeg[i, 0]
	yInt[i]       = linReg[i, 1]
	yIntPos[i]    = linRegPos[i, 1]
	yIntNeg[i]    = linRegNeg[i, 1]
	r2[i]         = linReg[i, 2]
	r2Pos[i]      = linRegPos[i, 2]
	r2Neg[i]      = linRegNeg[i, 2]
	yaxis[i]      = linReg[i, 0]*counts + linReg[i, 1]
	lines[i]      = colors[i]
axLabels         = ['Node ADC Counts', 'Pressure [mmHg]']
title            = 'Calibration Curves'
PlotData(xaxis, yaxis, lines, axLabels, linLabels, title)

#Plot ensemble of Dual Cal Curves
numReg    = len(fileNames)
countsNeg = np.array([800, 1320])
countsPos = np.array([1210, 2300])
xaxis     = [[]]*(numReg*2)
yaxis     = [[]]*(numReg*2)
lines     = [[]]*(numReg*2)
linLabels = [[]]*(numReg*2) 
cnt = 0
for i in range(numReg):
	xaxis[i]     = countsNeg
	#yaxis[i]     = linRegNeg[i, 0]*countsNeg + linRegNeg[:, 1]
	yaxis[i]     = linRegNeg[i, 0]*countsNeg - 1245*linRegNeg[i, 0]
	linLabels[i] = 'Curve ' + str(i+1) + 'Negative Curve'
	lines[i]     = colors[i]
	cnt          = cnt + 1
for i in range(numReg):
	xaxis[i + cnt]     = countsPos
	#yaxis[i + cnt]     = linRegPos[i, 0]*countsPos + linRegPos[:, 1]
	yaxis[i + cnt]     = linRegPos[i, 0]*countsPos - 1245*linRegPos[i, 0]
	linLabels[i + cnt] = 'Curve ' + str(i+1)  + 'Positive Curve'
	lines[i]           = colors[i]
axLabels               = ['Node ADC Counts', 'Pressure [mmHg]']
title                  = 'Calibration Curves'
PlotData(xaxis, yaxis, lines, axLabels, linLabels, title)

#Print Stats to terminal
print('##################################')
print('######## Stats Positive ##########')
print('##################################')
slopeMean = np.mean(slopesPos); meanPos = slopeMean
slopeStd  = np.std(slopesPos)
slopeMax  = np.amax(slopesPos); imax = np.argmax(slopesPos)
slopeMin  = np.amin(slopesPos); imin = np.argmin(slopesPos)
print('Slope Positive')
print('mean = ' + str(slopeMean) + '... std = ' +str(slopeStd))
print('max  = ' + str(slopeMax) + ' ... ' + fileNames[imax])
print('min  = ' + str(slopeMin) + ' ... ' + fileNames[imin])
print('Resolution = ' + str(1/slopeMean) + ' counts/mmHg')
intMean = np.mean(yIntPos)
intStd  = np.std(yIntPos)
intMax  = np.amax(yIntPos)
intMin  = np.amin(yIntPos)
print('y Intercept Positive')
print('mean = ' + str(intMean) + '... std = ' +str(intStd))
print('max  = ' + str(intMax))
print('min  = ' + str(intMin))
r2Mean = np.mean(r2Pos)
r2Std  = np.std(r2Pos)
print('r2 Positive')
print('mean = ' + str(r2Mean) + '... std = ' +str(r2Std))
print()
print('##################################')
print('######## Stats Negative ##########')
print('##################################')
slopeMean = np.mean(slopesNeg)
slopeStd  = np.std(slopesNeg)
slopeMax  = np.amax(slopesNeg)
slopeMin  = np.amin(slopesNeg)
print('Slope Negative')
print('mean = ' + str(slopeMean) + '... std = ' +str(slopeStd))
print('max  = ' + str(slopeMax))
print('min  = ' + str(slopeMin))
print('Resolution = ' + str(1/slopeMean) + ' counts/mmHg')
intMean = np.mean(yIntNeg)
intStd  = np.std(yIntNeg)
intMax  = np.amax(yIntNeg)
intMin  = np.amin(yIntNeg)
print('y Intercept Nega+tive')
print('mean = ' + str(intMean) + '... std = ' +str(intStd))
print('max  = ' + str(intMax))
print('min  = ' + str(intMin))
r2Mean = np.mean(r2Neg)
r2Std  = np.std(r2Neg)
print('r2 Negative')
print('mean = ' + str(r2Mean) + '... std = ' +str(r2Std))
print('##################################')
print('############# Counts #############')
print('##################################')
ccTop = 0
ccBot = 0 
countsMax = np.zeros(len(fileNames))
countsMin = np.zeros(len(fileNames))
for i in range(len(fileNames)):
	countsMax[i] = np.amax(countsAll[i])
	countsMin[i] = np.amin(countsAll[i])
	if countsMin[i] == 0 or countsMin[i] == 1:
		ccBot = ccBot + 1
	if countsMax[i] == 4095:
		ccTop = ccTop + 1
print('Max')
print('Mean = ' + str(np.mean(countsMax)) + ' ... std = ' + str(np.std(countsMax)))
print(' Max = ' + str(np.amax(countsMax)))	
print(' Min = ' + str(np.amin(countsMax)))
print('Min')
print('Mean = ' + str(np.mean(countsMin)) + ' ... std = ' + str(np.std(countsMin)))
print(' Max = ' + str(np.amax(countsMin)))	
print(' Min = ' + str(np.amin(countsMin)))
print('Sum')	
print(' Max Counts = ' + str(np.amax(countsAll)))
print(' Min Counts = ' + str(np.amin(countsAll)))
print('Num Cas Top = ' + str(ccTop))
print('Num Cas Bot = ' + str(ccBot))
print('.....')
print('1000 mmHg')
print('Mean = ' + str(np.mean(counts1000)) + ' ... std = ' + str(np.std(counts1000)))
print(' Max = ' + str(np.amax(counts1000)))	
print(' Min = ' + str(np.amin(counts1000)))
print('')
print('-350 mmHg')
print('Mean = ' + str(np.mean(counts350)) + ' ... std = ' + str(np.std(counts350)))
print(' Max = ' + str(np.amax(counts350)))	
print(' Min = ' + str(np.amin(counts350)))
print('')

SlopesOrdered, CasNumLowHigh = OrderData(slopesPos, fileNames)

#create normal probability plot
n = np.zeros(len(SlopesOrdered))
for i in range(len(SlopesOrdered)):
	n[i] = i + 1

fi = (n-0.375)/(len(SlopesOrdered)+0.25)
zVal = norm.ppf(fi)

linRegNormProb = np.zeros(5)
linRegNormProb = linearRegression(SlopesOrdered, zVal, axLabels, title, plotData = False)

print('n = ' + str(len(SlopesOrdered)) + '... 0.9784 ... 0.9850')
print('r^2 = ' + str(linRegNormProb[2]))
print('')

xaxis     = [SlopesOrdered, SlopesOrdered]
yaxis     = [zVal, linRegNormProb[0]*SlopesOrdered + linRegNormProb[1]]
lines     = ['r.', 'b--']
axLabels  = ['Positive Gains [mmHg]', 'Z-Value']
linLabels = ['Normal Probability Curve', 'Linear Regression']
title     = 'Normal Probability Curve'
PlotData(xaxis, yaxis, lines, axLabels, linLabels, title)

curveList1 = open('curveList1.txt', 'a+') #start new file or append to an exsisting. Appending to an exsisting file is undersireable 
curveList2 = open('curveList2.txt', 'a+')  

for i in range(len(SlopesOrdered)):
 	print(str(SlopesOrdered[i]) + ' . . . ' + CasNumLowHigh[i])
 	curveList1.write(str(SlopesOrdered[i]) + '\n')
 	curveList2.write(CasNumLowHigh[i] + '\n')

curveList1.close()
curveList2.close()

# print('Low End Gain')
# print(SlopesOrdered[0:5])
# print(CasNumLowHigh[0:5])
# print('High End Gain')
# print(SlopesOrdered[len(SlopesOrdered)-6:len(SlopesOrdered)-1])
# print(CasNumLowHigh[len(SlopesOrdered)-6:len(SlopesOrdered)-1])
# print('All Slopes')

# plt.figure()
# plt.hist(slopesPos, 20)
# plt.title('Gain Histogram', fontsize = 22)
# plt.xlabel('Gain [Counts/mmHg]', fontsize = 20)
# plt.ylabel('Frequency', fontsize = 20)

# slopeRatio = np.divide(slopesPos, slopesNeg)
# print('Slope Ratio Stats')
# print('Mean = ' + str(np.mean(slopeRatio)) + ' . . . Std = ' + str(np.std(slopeRatio)))
# print(' Min = ' + str(np.amin(slopeRatio)))
# print(' Max = ' + str(np.amax(slopeRatio)))
# plt.figure()
# xaxis     = [np.arange(1, 11, 1)]
# yaxis     = [slopeRatio]
# lines     = ['b.']
# axLabels  = ['Cassette Number', 'Slope Ratio']
# linLabels = ['Slope Ratio']
# title     = 'Slope Ratio'
# PlotData(xaxis, yaxis, lines, axLabels, linLabels, title)



plt.show()