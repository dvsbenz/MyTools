##############################################
##############Libraries#######################
##############################################
import numpy as np              #import numpy library
from numpy import genfromtxt
from numpy import loadtxt
import numpy.random as rn
import scipy as sp              #import scipy library
from scipy import stats   
from scipy.optimize import curve_fit
from scipy.signal import medfilt
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt #import matplotlib
import fnmatch                  #import fnmatch libary, allows for unix shell-style wildcards
import os, os.path              #import os library, usefull functions for pathnames
import re                       #import re library, provides regular expression matching similar to perl
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

def GetTimeInd(time, sec):
	ind = 0
	for i in range(len(time)):
		if time[i] > sec:
			ind = i
			break
	return i

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

def SubPlotData(xaxis1, xaxis2, xaxis3, yaxis1, yaxis2, yaxis3, lines, axLabels, linLabels, title):
	xlims1, ylims1 = GenPlotLims(xaxis1, yaxis1)
	xlims2, ylims2 = GenPlotLims(xaxis2, yaxis2)
	xlims3, ylims3 = GenPlotLims(xaxis3, yaxis3)
	plt.figure()
	plt.subplot(3, 1, 1)
	for i in range(len(xaxis1)):
		plt.plot(xaxis1[i], yaxis1[i], lines[0][i], linewidth = 1.0, mew = 1.0, label = linLabels[0][i])
	plt.tick_params(labelsize = 18)
	plt.title(title[0], fontsize = 22)
	plt.xlabel(axLabels[0][0], fontsize = 20)
	plt.ylabel(axLabels[0][1], fontsize = 20)
	plt.axis([xlims1[0], xlims1[1], ylims1[0], ylims1[1]])
	plt.legend(loc = 'best')

	plt.subplot(3, 1, 2)
	for i in range(len(xaxis2)):
		plt.plot(xaxis2[i], yaxis2[i], lines[1][i], linewidth = 1.0, mew = 1.0, label = linLabels[1][i])
	plt.tick_params(labelsize = 18)
	#plt.title(title[1], fontsize = 22)
	plt.xlabel(axLabels[1][0], fontsize = 20)
	plt.ylabel(axLabels[1][1], fontsize = 20)
	plt.axis([xlims2[0], xlims2[1], ylims2[0], ylims2[1]])
	plt.legend(loc = 'best')

	plt.subplot(3, 1, 3)
	for i in range(len(xaxis3)):
		plt.plot(xaxis3[i], yaxis3[i], lines[2][i], linewidth = 1.0, mew = 1.0, label = linLabels[2][i])
	plt.tick_params(labelsize = 18)
	#plt.title(title[2], fontsize = 22)
	plt.xlabel(axLabels[2][0], fontsize = 20)
	plt.ylabel(axLabels[2][1], fontsize = 20)
	plt.axis([xlims3[0], xlims3[1], ylims3[0], ylims3[1]])
	plt.legend(loc = 'best')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
############ Calibration Equations ###########
def CalibrationEqn(data, m, b):
	y = m*data + b
	return y

def AdjustIntercept(data, pres, m):
	b = pres - m*data
	return b
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
########## Dual Cal Eqn Algorithm ############
def CalcPresDual(data, mPos, mNeg, bPos, bNeg, offset):
	bufferSize = 10
	presBuffer = RingBuffer(bufferSize)
	pres       = np.zeros(len(data))
	curveType  = np.zeros(len(data))
	typePos    = 500
	typeNeg    = -250

	for i in range(bufferSize):
		if offset >= 0:
			posEqn       = True
			presPos      = CalibrationEqn(data[i], mPos, bPos)
			presBuffer.append(presPos)    #fill buffer with the inital values using positive cal equation
			pres[i]      = presPos
			curveType[i] = typePos
		else:
			posEqn       = False
			presNeg      = CalibrationEqn(data[i], mNeg, bNeg)
			presBuffer.append(presNeg)    #fill buffer with the inital values using neg cal equation
			pres[i]      = presNeg
			curveType[i] = typeNeg


	for i in range(len(data) - bufferSize):
		presPos = CalibrationEqn(data[i + bufferSize], mPos, bPos)
		presNeg = CalibrationEqn(data[i + bufferSize], mNeg, bNeg)
		if posEqn == True:
			pres[i + bufferSize] = presPos
			curveType[i + bufferSize] = typePos
			presBuffer.append(presPos)
			if presPos < 0:
				grad = np.mean(np.gradient(presBuffer.get()))
				mean = np.mean(presBuffer.get())
				if grad < 0 and mean < 0:
					posEqn = False
		if posEqn == False:
			pres[i + bufferSize] = presNeg
			curveType[i + bufferSize] = typeNeg
			presBuffer.append(presNeg)
			if presNeg > 0:
				grad = np.mean(np.gradient(presBuffer.get()))
				mean = np.mean(presBuffer.get())
				if grad > 0 and mean > 0:
					posEqn = True
	return pres, curveType

class RingBuffer:
    def __init__(self, size):
        self.data = [None for i in range(size)]

    def append(self, x):
        self.data.pop(0)
        self.data.append(x)

    def get(self):
        return self.data

def AppendBuffer():
    global error
    global errBuffer
    errBuffer.append(error)

def GetBufferAvg():
    global errBuffer
    errMean = np.mean(errBuffer.get())
    return errMean
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
############ Error Calculations ##############
def CalcError(sensor, reference):
	#calculate the error between the calibrated reference and sthe sensor
	diff      = reference - sensor
	error     = np.abs(diff/reference)*100

	return diff, error 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##############################################
#####################MAIN#####################
##############################################
fileNames     = GetFileNames()

mPosAvg     = 1.089
mPosMax     = 1.171
mPosMin     = 1.044
bPosInt     = -781.923345

mNegAvg     = 0.631627
mNegMax     = 0.658379
mNegMin     = 0.607019
bNegInt     = -740.838562

mPosGold    = 0.659117
mNegGold    = 0.612607

mPosStd     = 0.009274

Gf          = 0.009089

FtP         = 73.052448

gainRatio   = 1.1

#mPos        = Gf*FtP
mPos        = mPosAvg
mNeg        = mPos/gainRatio
assumedPres = 0.0 #Asume pressure is 0mmHg prior to intial draw

print()
print('         Sensor Gain = ' + str(mPos))
print('Deviations from Mean = ' + str(np.abs((mPos - mPosAvg)/mPosStd)))
print()

tNom          = 1.5                       #time slice for unloaded counts
countsNom     = np.zeros(len(fileNames))  #check unloaded counts at start of each run
countsPreload = np.zeros(len(fileNames))
preloadDelta  = np.zeros(len(fileNames)) #delta in preload after decay

PF          = 1
AT          = 0
errReq      = [6.0, 12.0]
errReq2     = 50
numPres     = 15
setPoints   = np.zeros((len(fileNames), numPres))
presDiff    = np.zeros((len(fileNames), numPres))
errSect     = np.zeros((len(fileNames), numPres))
maxPresDiff = np.zeros(len(fileNames))
posErrSect  = []
negErrSect  = []
presDiffHi  = []

gateStartInd  = np.zeros(numPres)
gateStartTime = np.zeros((numPres,2))
gateEndInd    = np.zeros(numPres)
gateEndTime   = np.zeros((numPres,2))
gate          = [-350, 1000]
gateErrPos    = [0, 6]
gateErrNeg    = [0, 12]
gateErrPres   = [-50, 50]

for i in range(len(fileNames)):
	#Load Data
	time, pres, APS, SP, SS, DP, pumpSpeed = LoadData(fileNames[i])

	#Calc Sensor Nominal counts and sensor preload after decay
	indNom       = GetTimeInd(time, tNom)
	countsNom[i] = np.round(np.mean(APS[0:indNom]), decimals = 0)
	for j in range(len(DP)):
		if DP[j] == 2  and DP[j+1] == 1:
			latchInd = j+1
			break
	for j in range(len(pumpSpeed)):
		if pumpSpeed[j] > 0.0 or pumpSpeed[j] < 0.0:
			indDec = j-1
			break
	tdecay = time[indDec] - 60
	for l in range(len(time)):
		if time[l] > tdecay:
			indDec = l-1
			break
	countsPreload[i] = np.round(np.mean(APS[indDec - 25 : indDec]), decimals = 0)
	preloadDelta[i]  = np.round(countsPreload[i] - countsNom[i], decimals = 0)

	#zero system prior to run
	bPos = AdjustIntercept(countsPreload[i], assumedPres, mPos)
	bNeg = AdjustIntercept(countsPreload[i], assumedPres, mNeg)
	######### Dual ERROR ##############
	#Data using basic cal curve
	APSDual, curveType = CalcPresDual(APS, mPos, mNeg, bPos, bNeg, preloadDelta[i])
	#Calc Error Basic
	diff, error = CalcError(APSDual, pres)
	############# Generate Gates #################
	#Gates
	gsc = 0
	gec = 0
	for j in range(len(SS)-1):
		if SS[j] == ' False' and SS[j + 1] == ' True':
			#gate start
			gateStartInd[gsc]     = j
			gateStartTime[gsc, 0] = time[j]
			gateStartTime[gsc, 1] = time[j]
			gsc = gsc + 1
		elif SS[j] == ' True' and SS[j+1] == ' False':
			#gate end
			gateEndInd[gec]     = j
			gateEndTime[gec, 0] = time[j]
			gateEndTime[gec, 1] = time[j]
			gec = gec + 1
	gateEndInd[numPres - 1] = len(SS) - 1
	#Error
	PF       = 1
	sections = 0
	for j in range(numPres):
		setPoints[i, j] = np.mean(SP[int(gateStartInd[j]):int(gateEndInd[j])])   #setpoints for some pos/neg specific statistics
		presDiff[i, j]  = np.mean(diff[int(gateStartInd[j]):int(gateEndInd[j])])  #mean pressure difference at section
		errSect[i, j]   = np.mean(error[int(gateStartInd[j]):int(gateEndInd[j])]) #mean error at section
		presDiffHi.append(presDiff[i, :])
		if setPoints[i, j] >= 0:
			if errSect[i, j] < 1.00*errReq[0]:      #throw out super high error caused by small differences near 0
				posErrSect.append(errSect[i, j])
			if errSect[i, j] >=  errReq[0] and np.abs(presDiff[i, j]) > errReq2:
				PF = 0
				sections = sections + 1
		else:
			if errSect[i, j] < 1.00*errReq[1]:      #throw out super high error caused by small differences near 0
				negErrSect.append(errSect[i, j]) 
			if errSect[i, j] >=  errReq[1] and np.abs(presDiff[i, j]) > errReq2:
				PF = 0
				sections = sections + 1
	if PF == 0:
		print('Test ' + str(i+1) + ' Failed ..... ' + str(sections) + ' sections failed')
		AT = 1
	############# Plot Data and Error ############
	#scale Data for ploting
	for j in range(len(error)):
		if error[j] > errReq[1] + 1:
			error[j] = errReq[1] + 1
		if diff[j] > errReq2 + 10:
			diff[j] = errReq2 + 10
		elif diff[j] < (errReq2 * -1) - 10:
			diff[j] = (errReq2 * -1) - 10
	errT     = [time[0], time[len(time)-1]]
	posErr   = [errReq[0], errReq[0]]   #6%
	NegErr   = [errReq[1], errReq[1]]   #12%
	ErrDiffP = [errReq2, errReq2]       #+/- 20mmHg
	ErrDiffN = [errReq2*-1, errReq2*-1] #+/- 20mmHg
	#Plot1 = data
	xaxis1           = [time, time]
	yaxis1           = [pres, APSDual]
	#Plot2 = percent error
	xaxis2           = [time, errT, errT]
	yaxis2           = [error, posErr, NegErr]
	#Plot3 = error difference [mmHg]
	xaxis3           = [time, errT, errT]
	yaxis3           = [diff, ErrDiffP, ErrDiffN]
	lines            = [['b-', 'r--', 'g--'], ['b--', 'g--', 'r--'], ['b--', 'g--', 'r--']]
	axLabels         = [['time [s]', 'Pressure [mmhg]'], ['time [s]', 'Error [%]'],  ['time [s]', 'Error [mmHg]']]
	linLabels        = [['Calibrated Pressure Reference [mmHg]', 'APS [mmHg]', 'Curve Type'], ['Percent Error', 'Pos Err Spec', 'Neg Err Spec'], ['Error [mmHg]', 'Percent Full Scale Error [mmHg]', 'Percent Full Scale Error [mmHg]']]
	titleTop         = 'Cassette ' + str(i + 1) + ' Dual Curve Error'
	titleMiddle      = 'Percent Error'
	titleBot         = 'Full Scale Error'
	title            = [titleTop, titleMiddle, titleBot]
	SubPlotData(xaxis1, xaxis2, xaxis3, yaxis1, yaxis2, yaxis3, lines, axLabels, linLabels, title)
	# for j in range(18):
	# 	plt.subplot(3, 1, 1)
	# 	plt.plot(gateStartTime[j,:], gate, 'm-', linewidth = 1.0, mew = 1.0)
	# 	plt.plot(gateEndTime[j,:], gate, 'k-', linewidth = 1.0, mew = 1.0)
	# 	plt.subplot(3, 1, 2)
	# 	plt.plot(gateStartTime[j,:], gateErr, 'm-', linewidth = 1.0, mew = 1.0)
	# 	plt.plot(gateEndTime[j,:], gateErr, 'k-', linewidth = 1.0, mew = 1.0)
	# 	plt.subplot(3, 1, 3)
	# 	plt.plot(gateStartTime[j,:], gatePres, 'm-', linewidth = 1.0, mew = 1.0)
	# 	plt.plot(gateEndTime[j,:], gatePres, 'k-', linewidth = 1.0, mew = 1.0)
	# ##############################################

if AT == 0:
	print()
	print('All Test Passed')

print()
print('Preload Info')
print('counts nominal')
print('mean = ' + str(np.mean(countsNom)) + ' ... std = ' + str(np.std(countsNom)))
print(' max = ' + str(np.amax(countsNom)))
print(' min = ' + str(np.amin(countsNom)))
print('Preload Counts')
print('mean = ' + str(np.mean(preloadDelta)) + ' ... std = ' + str(np.std(preloadDelta)))
print(' max = ' + str(np.amax(preloadDelta)))
print(' min = ' + str(np.amin(preloadDelta)))
print()
print('Error Info')
print('      Number of Cassettes = ' + str(len(fileNames)))
print('      Number of Pressures = ' + str(numPres))
print('Total Number of Pressures = ' + str(len(fileNames) * numPres))
print()
print('Error Stats')
print('Overall Stats')
print('mean = ' + str(np.mean(errSect)) + ' ... std = ' + str(np.std(errSect)))
print(' max = ' + str(np.amax(errSect)))
print(' min = ' + str(np.amin(errSect)))
print()
print('Positive Error')
print('mean = ' + str(np.mean(posErrSect)) + ' ... std = ' + str(np.std(posErrSect)))
print(' max = ' + str(np.amax(posErrSect)))
print(' min = ' + str(np.amin(posErrSect)))
print()
print('Negative Error')
print('mean = ' + str(np.mean(negErrSect)) + ' ... std = ' + str(np.std(negErrSect)))
print(' max = ' + str(np.amax(negErrSect)))
print(' min = ' + str(np.amin(negErrSect)))
print()
print('Pres Diff Stats')
print('mean = ' + str(np.mean(presDiff)) + ' ... std = ' + str(np.std(presDiff)))
print(' max = ' + str(np.amax(presDiff)))
print(' min = ' + str(np.amin(presDiff)))
cassCounter  = 0
groupCounter = 0
presDiff2    = []
order        = np.arange(0,len(fileNames),1)
order        = rn.permutation(order)
for i in range(len(fileNames)):
	for j in range(numPres):
		presDiff2.append(presDiff[order[i],j])
	cassCounter = cassCounter + 1
	if cassCounter > 3:
		p, cv, cl = sp.stats.anderson(presDiff2)
		# plt.figure()
		# res = stats.probplot(presDiff2, plot = plt)
		#print('p = ' + str(p) + ' ..... ' + 'cv = ' + str(cv[0]))
		if p < cv[0]:
			print('Group ' + str(groupCounter + 1) + ', Anderson Test: Accept Normality')
		else:
			#Perform Shapiro-Wilks Normality Test
			w, ps = sp.stats.shapiro(presDiff2)
			if ps >= 0.05:
				print('Group ' + str(groupCounter + 1) + ', Shapiro Test: Accept Normality')
			else:
				#Trasnsform data with yeo-johnson normality transformation
				pt = PowerTransformer(method = 'yeo-johnson')
				presDiff_PT = np.array(presDiff2)
				presDiff_PT = presDiff_PT.reshape(-1, 1)
				pt.fit(presDiff_PT)
				presDiff_t = pt.transform(presDiff_PT)
				presDiff_t2 = []
				for k in range(np.size(presDiff_t)):
					presDiff_t2.append(presDiff_t[k, 0])
				pt, cvt, clt = sp.stats.anderson(presDiff_t2)
				#print('p = ' + str(p) + ' ..... ' + 'cv = ' + str(cv[0]))
				if pt < cvt[0]:
					print('Group ' + str(groupCounter + 1) + ', Anderson Test Transformed: Accept Normality')
				else:
					wt, pst = sp.stats.shapiro(presDiff_t2)
					if pst >= 0.05:
						print('Group ' + str(groupCounter + 1) + ', Shapiro Test Transformed: Accept Normality')
					else:
						print('Group ' + str(groupCounter + 1) + ', Not a Normal Distribuition')
		groupCounter = groupCounter + 1
		cassCounter  = 0
		presDiff2 = []

# plt.figure()
# plt.tick_params(labelsize = 18)
# plt.title('Positive Error Histogram', fontsize = 22)
# plt.xlabel('Error [%]', fontsize = 20)
# plt.ylabel('Frequency', fontsize = 15)
# plt.hist(posErrSect, bins = 50)

# plt.figure()
# plt.tick_params(labelsize = 18)
# plt.title('Negative Error Histogram', fontsize = 22)
# plt.xlabel('Error [%]', fontsize = 20)
# plt.ylabel('Frequency', fontsize = 15)
# plt.hist(negErrSect, bins = 50)

# plt.figure()
# plt.tick_params(labelsize = 18)
# plt.title('Pressure Difference Histogram', fontsize = 22)
# plt.xlabel('Pressure [mmHg]', fontsize = 20)
# plt.ylabel('Frequency', fontsize = 15)
# plt.hist(presDiffHi, bins = 50)

plt.show()