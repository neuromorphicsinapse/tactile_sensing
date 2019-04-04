# -*- coding: utf-8 -*-
'''
#-------------------------------------------------------------------------------
# NATIONAL UNIVERSITY OF SINGAPORE - NUS
# SINGAPORE INSTITUTE FOR NEUROTECHNOLOGY - SINAPSE
# Singapore
# URL: http://www.sinapseinstitute.org
#-------------------------------------------------------------------------------
# Neuromorphic Engineering Group
#-------------------------------------------------------------------------------
# Description: GUI to help analyzing data
#-------------------------------------------------------------------------------
'''
#-------------------------------------------------------------------------------
#Paths
import os, sys, glob
sys.path.append('../framework/libraries/general')
sys.path.append('../framework/libraries/texture_recognition')
sys.path.append('../framework/libraries/neuromorphic')
#-------------------------------------------------------------------------------
#PyQt libraries
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import QFileDialog #file management
#-------------------------------------------------------------------------------
#Libraries
import numpy as np
import scipy.signal as sig
from copy import copy
from dataprocessing import *
import texture
from tactileboard import *
import spiking_neurons as spkn
#-------------------------------------------------------------------------------
#matplotlib libraries for pyqt
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------
Ui_MainWindow, QMainWindow = loadUiType('formSpikeAnalysis_gui.ui')
#-------------------------------------------------------------------------------
class CONSTS():
    NOFILTER = 0
    HIGHPASS = 1
    LOWPASS = 2
    BANDPASS = 3
    FILE_PREFIX = ''
    SELECT_TEXTURE = 0
    SELECT_FORCE = 1
    SELECT_PALPATION = 2
    RAW = 0
    DERIVATIVE = 1
    NTAXELS = 16
    NPATCH = 5
class FormSpikeAnalysis(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(FormSpikeAnalysis,self).__init__()
        self.setupUi(self)
        self.setupGUI()
        #matplotlib
        #raw signal
        self.rawFigure = plt.figure()
        self.rawCanvas = FigureCanvas(self.rawFigure)
        self.rawToolbar = NavigationToolbar(self.rawCanvas,self)
        self.vlayoutSignal.addWidget(self.rawToolbar)
        self.vlayoutSignal.addWidget(self.rawCanvas)
        self.rawAxes = self.rawFigure.add_subplot(1,1,1)
        #processed signal
        self.procFigure = plt.figure()
        self.procCanvas = FigureCanvas(self.procFigure)
        self.procToolbar = NavigationToolbar(self.procCanvas,self)
        self.vlayoutSignal.addWidget(self.procToolbar)
        self.vlayoutSignal.addWidget(self.procCanvas)
        self.procAxes = self.procFigure.add_subplot(1,1,1)
        #spikes
        #taxel inputs
        self.inputSpkFigure = plt.figure()
        self.inputSpkCanvas = FigureCanvas(self.inputSpkFigure)
        self.inputSpkToolbar = NavigationToolbar(self.inputSpkCanvas,self)
        self.vlayoutSpikes.addWidget(self.inputSpkToolbar)
        self.vlayoutSpikes.addWidget(self.inputSpkCanvas)
        #results from the simulation
        self.simulResultsFigure = plt.figure()
        self.simulResultsCanvas = FigureCanvas(self.simulResultsFigure)
        self.simulResultsToolbar = NavigationToolbar(self.simulResultsCanvas,self)
        self.vlayoutSpikes.addWidget(self.simulResultsToolbar)
        self.vlayoutSpikes.addWidget(self.simulResultsCanvas)
        #variables
        #tactile signal
        self.rawTactile = []
        self.procTactile = []
        #filter
        self.filterType = CONSTS.NOFILTER
        #current
        self.currentType = CONSTS.RAW
#-------------------------------------------------------------------------------
    def setupGUI(self):
        #button events
        self.btnLoadSignal.clicked.connect(self.doLoad)
        self.btnApply.clicked.connect(self.doApply)
        self.btnRun.clicked.connect(self.runIzhikevich)
        self.btnExport.clicked.connect(self.doExportSpikes)
        #filter
        #populate the combobox
        self.cbFilter.addItems(['No filter','High-pass','Low-pass','Band-pass'])
        self.cbFilter.currentIndexChanged.connect(self.doChangeFilter)
        self.doChangeFilter(CONSTS.NOFILTER) #initial condition --> no filter
        self.tbPoles.setText('4')
        self.tbHighPass.setText('1')
        self.tbLowPass.setText('10')
        self.cbInputCurrent.addItems(['Raw','Derivative'])
        self.cbInputCurrent.currentIndexChanged.connect(self.doChangeInputCurrent)
        self.cbSensorPatch.addItems(['FFC1','FFC2','FFC3','FFC4','FFC5'])
        self.cbSensorPatch.currentIndexChanged.connect(self.doSelectPatch)
        #normalization
        self.chNormalize.stateChanged.connect(self.doNormalize)
        self.doNormalize()
        #default values for normalization
        self.tbBaseline.setText('60')
        self.tbVmin.setText('0')
        self.tbVmax.setText('1')
        #cutting the signal
        self.chCut.stateChanged.connect(self.doCut)
        self.tbTstart.setText('6')
        self.tbTend.setText('28')
        self.doCut()
        #handling plots
        self.cbPlot1.setDisabled(True)
        self.cbPlot2.setDisabled(True)
        self.cbPlot1.addItems(['all taxels','individual taxels'])
        self.cbPlot1.currentIndexChanged.connect(self.doPlotInputs)
        self.cbPlot2.addItems(['membrane voltage','rastergram'])
        self.cbPlot2.currentIndexChanged.connect(self.plotSpikes)
#-------------------------------------------------------------------------------
    def doChangeFilter(self,i):
        if i == CONSTS.NOFILTER:
            self.tbPoles.setDisabled(True)
            self.tbLowPass.setDisabled(True)
            self.tbHighPass.setDisabled(True)
            self.filterType = CONSTS.NOFILTER
        elif i == CONSTS.HIGHPASS:
            #enable high-pass only
            self.tbPoles.setDisabled(False)
            self.tbHighPass.setDisabled(False)
            self.tbLowPass.setDisabled(True)
            self.filterType = CONSTS.HIGHPASS
        elif i == CONSTS.LOWPASS:
            #enable low-pass only
            self.tbPoles.setDisabled(False)
            self.tbHighPass.setDisabled(True)
            self.tbLowPass.setDisabled(False)
            self.filterType = CONSTS.LOWPASS
        elif i == CONSTS.BANDPASS:
            self.tbPoles.setDisabled(False)
            self.tbLowPass.setDisabled(False)
            self.tbHighPass.setDisabled(False)
            self.filterType = CONSTS.BANDPASS
            #enable both low-pass and high-pass
#-------------------------------------------------------------------------------
    def doCut(self):
        if self.chCut.isChecked():
            self.tbTstart.setDisabled(False)
            self.tbTend.setDisabled(False)
        else:
            self.tbTstart.setDisabled(True)
            self.tbTend.setDisabled(True)
#-------------------------------------------------------------------------------
    def doNormalize(self):
        if self.chNormalize.isChecked():
            self.tbBaseline.setDisabled(False)
            self.tbVmin.setDisabled(False)
            self.tbVmax.setDisabled(False)
        else:
            self.tbBaseline.setDisabled(True)
            self.tbVmin.setDisabled(True)
            self.tbVmax.setDisabled(True)
#-------------------------------------------------------------------------------
    def doSelectPatch(self):
        self.rawTactile = self.tactiledata[self.cbSensorPatch.currentIndex()]
        self.plotRaw()
#-------------------------------------------------------------------------------
    def doApply(self):
        '''
        apply the pre-processing stages
        '''
        #select the proper sensor patch according to the GUI
        # self.doSelectPatch()
        #copy the raw signals
        self.procTactile = [self.rawTactile[:,k] for k in range(TBCONSTS.NTAXELS)]
        #-----------------------------------------------------------------------
        #filtering
        if self.filterType is not CONSTS.NOFILTER:
            ret = self.createFilter()
            #error when creating the filter
            if ret is False:
                self.showErrorMsgBox('Filter could not be created. Please, check the parameters.')
                return False
            else: #the filter was created
                ret = self.applyFilter()
                if ret is False:
                    self.showErrorMsgBox('Filter could not be applied to the signal. Please, check the parameters.')
                    return False
        #-----------------------------------------------------------------------
        #normalization
        if self.chNormalize.isChecked():
            ret = self.normalize()
            if ret is False:
                self.showErrorMsgBox('Error while normalizing the signals. Please, check the parameters')
                return False
        #-----------------------------------------------------------------------
        #splitting the signal
        if self.chCut.isChecked():
            ret = self.segment()
            if ret is False:
                self.showErrorMsgBox('Error while segmenting the signals. Please, check the parameters')
                return False
        #-----------------------------------------------------------------------
        #plot the processed signal
        self.plotProc()
        #-----------------------------------------------------------------------
#-------------------------------------------------------------------------------
    def showErrorMsgBox(self,strmsg):
        '''
        display an error message box with the specified text
        '''
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText(strmsg)
        msg.setWindowTitle("Error")
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
#-------------------------------------------------------------------------------
    def doLoad(self):
        '''
        load a specified tactile signal from the texture recognition
        experiment
        '''
        #open a save file dialog
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        #open the dialog and receives the filename
        diag = QFileDialog()
        filename,_ = diag.getOpenFileName(None,'Title','','Tactile data (*.txt)')
        #saves the poses
        #if clicked 'Cancel', then string will be False
        if filename:
            print('file to be loaded: ' + filename)
            self.filename = filename
            #load the signal
            resp = self.loadTactile()
            if resp:
                # msg = QMessageBox()
                # msg.setIcon(QMessageBox.Information)
                # msg.setText("File has been loaded successfully!")
                # # msg.setInformativeText("There was an error while loading the file.")
                # msg.setWindowTitle("Info")
                # msg.setStandardButtons(QMessageBox.Ok)
                self.plotRaw()
            else:
                self.showErrorMsgBox('Unable to load file. Please, check if the file is not corrupted.')
#-------------------------------------------------------------------------------
    def loadTactile(self):
        '''
        load tactile signal from a given file
        '''
        try:
            if self.chTexture.isChecked():
                print('here')
                tactile_signals = np.loadtxt(self.filename)
                self.tactiledata = [[] for k in range(TBCONSTS.NPATCH)]
                for k in range(1):
                    self.tactiledata[k] = np.zeros((len(tactile_signals),TBCONSTS.NTAXELS))
                    for j in range(TBCONSTS.NTAXELS):
                        self.tactiledata[k][:,j] = tactile_signals[:,j]
                self.rawTactile = self.tactiledata[0]
                self.rawTime = np.arange(0,len(self.rawTactile)) * (1.0/TBCONSTS.SAMPFREQ)
                return True
            else:
                tactile_signals = np.loadtxt(self.filename)
                self.tactiledata = [[] for k in range(TBCONSTS.NPATCH)]
                for k in range(TBCONSTS.NPATCH):
                    self.tactiledata[k] = np.zeros((len(tactile_signals),TBCONSTS.NTAXELS))
                    aux_taxel = 0
                    for j in range(k,80,5):
                        self.tactiledata[k][:,aux_taxel] = tactile_signals[:,j]
                        aux_taxel += 1
                self.rawTactile = self.tactiledata[0]
                self.rawTime = np.arange(0,len(self.rawTactile)) * (1.0/TBCONSTS.SAMPFREQ)
                return True
        except:
            return False
#-------------------------------------------------------------------------------
    def plotRaw(self):
        '''
        plot the raw tactile signal
        '''
        self.rawAxes.clear()
        self.rawAxes.plot(self.rawTime,self.rawTactile[:,0:TBCONSTS.NTAXELS])
        self.rawAxes.set_xlim([self.rawTime[0],self.rawTime[-1]])
        # self.rawAxes = self.rawFigure.add_subplot(2,1,2)
        # self.rawAxes.plot(self.rawTime,self.rawTactile[:,-1])
        # self.rawAxes.set_xlim([self.rawTime[0],self.rawTime[-1]])
        self.rawAxes.set_xlabel('Time (s)')
        self.rawCanvas.draw()
#-------------------------------------------------------------------------------
    def plotProc(self):
        '''
        plot the processed signal
        '''
        self.procAxes.clear()
        self.procTime = np.arange(0,len(self.procTactile[0])) * (1.0/TBCONSTS.SAMPFREQ)
        [self.procAxes.plot(self.procTime,x) for x in self.procTactile]
        self.procAxes.set_xlim([self.procTime[0],self.procTime[-1]])
        self.procAxes.set_xlabel('Time (s)')
        self.procCanvas.draw()
#-------------------------------------------------------------------------------
    def createFilter(self):
        '''
        create the pre-processing filter
        '''
        try:
            #get the number of poles
            self.npoles = float(self.tbPoles.text())
            #create a high pass filter
            if self.filterType is CONSTS.HIGHPASS:
                self.highPassFc = float(self.tbHighPass.text())
                wn = self.highPassFc / (TBCONSTS.SAMPFREQ/2.0)
                self.filtb, self.filta = sig.butter(self.npoles,wn,'high')
            #create a low pass filter
            elif self.filterType is CONSTS.LOWPASS:
                self.lowPassFc = float(self.tbLowPass.text())
                wn = self.lowPassFc / (TBCONSTS.SAMPFREQ/2.0)
                self.filtb, self.filta = sig.butter(self.npoles,wn,'low')
            #create a bandpass filter
            elif self.filterType is CONSTS.BANDPASS:
                self.highPassFc = float(self.tbHighPass.text())
                self.lowPassFc = float(self.tbLowPass.text())
                wnh = self.highPassFc / (TBCONSTS.SAMPFREQ/2.0)
                wnl = self.lowPassFc / (TBCONSTS.SAMPFREQ/2.0)
                self.filtb, self.filta = sig.butter(self.npoles,[wnh,wnl],'bandpass')
            return True #return true --> success!
        except:
            return False #return false --> error!
#-------------------------------------------------------------------------------
    def applyFilter(self):
        self.procTactile = [sig.filtfilt(self.filtb,self.filta,x) for x in self.procTactile]
#-------------------------------------------------------------------------------
    def normalize(self):
        '''
        normalize the tactile signals
        '''
        try:
            ts = int(0); te = int(1*TBCONSTS.SAMPFREQ)
            #take the baseline value
            baseline = [np.mean(x[ts:te]) for x in self.procTactile]
            #take the minimum value
            self.vmin = float(self.tbVmin.text())
            #take the maximum value
            self.vmax = float(self.tbVmax.text())

            #take the percentage value
            self.baselinePercentage = float(self.tbBaseline.text()) / 100.0

            self.procTactile = [convscale(self.procTactile[k],baseline[k],baseline[k]*self.baselinePercentage,self.vmin,self.vmax) for k in range(TBCONSTS.NTAXELS)]
            # self.procTactile = [self.procTactile[k]-baseline[k] for k in range(TBCONSTS.NTAXELS)]

            #
            return True
        except:
            return False
#-------------------------------------------------------------------------------
    def segment(self):
        '''
        segment the signal into the desired interval
        '''
        try:
            tstart = float(self.tbTstart.text())
            tend = float(self.tbTend.text())
            tstart = int(tstart * TBCONSTS.SAMPFREQ)
            tend = int(tend * TBCONSTS.SAMPFREQ)
            if tstart >= tend:
                return False
            if tstart < 0:
                return False
            if tend > len(self.rawTime):
                return False

            self.procTactile = [x[tstart:tend] for x in self.procTactile]
        except:
            return False
#-------------------------------------------------------------------------------
    def doChangeInputCurrent(self,i):
        if i == CONSTS.RAW:
            self.currentType = CONSTS.RAW
        elif i == CONSTS.DERIVATIVE:
            self.currentType = CONSTS.DERIVATIVE
#-------------------------------------------------------------------------------
    def runIzhikevich(self):
        #retrieve the parameters
        self.a = float(self.tbIza.text())
        self.b = float(self.tbIzb.text())
        self.c = float(self.tbIzc.text())
        self.d = float(self.tbIzd.text())
        self.GF = float(self.tbGain.text())
        #neurons
        self.neurons = [spkn.model.izhikevich(a=self.a,b=self.b,c=self.c,d=self.d,name=str(k)) for k in range(TBCONSTS.NTAXELS)]
        #currents
        #determine if raw input current should be used or its derivative (FA-I)
        if self.currentType == CONSTS.DERIVATIVE:
            self.currents = [np.abs(np.insert(np.diff(x),0,0)) for x in self.procTactile]
        else:
            self.currents = [x for x in self.procTactile]
        #if normalize option is chosen, normalize the individual taxel
        #with respect to its maximum first, then multiply by GF
        if self.chIzNormalize.isChecked():
            self.currents = [(x/np.max(x))*self.GF for x in self.currents]
        else: #otherwise, just multiply directly by GF
            self.currents = [x*self.GF for x in self.currents]
        #other parameters
        self.t0 = 0
        self.tf = len(self.currents[0])
        self.dt = 1
        #create the simulation object
        self.simulObj = spkn.simulation(self.dt,self.t0,self.tf,self.currents,self.neurons)
        #run the simulation
        self.simulObj.optIzhikevich()
        #rastergram
        self.rasters = spkn.analysis.raster(self.simulObj)
        #plot the results
        self.doPlotInputs(0)
        self.plotSpikes(0)
        #enable controls
        self.cbPlot1.setDisabled(False)
        self.cbPlot1.setCurrentIndex(0)
        self.cbPlot2.setDisabled(False)
        self.cbPlot2.setCurrentIndex(0)
#-------------------------------------------------------------------------------
    def spikes2matrix(self):
        spikeMatrix = np.zeros((self.tf,len(self.simulObj.neurons)))
        for k in range(len(self.simulObj.neurons)):
            spikeMatrix[self.simulObj.spikes[k],k] = 1
        return spikeMatrix
#-------------------------------------------------------------------------------
    def doExportSpikes(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        #open the dialog and receives the filename
        diag = QFileDialog()
        name,_ = diag.getSaveFileName(self,'Save File','','Spikes data (*.txt)')
        if name:
            f = open(name,'w')
            spkmat = self.spikes2matrix()
            for i in range(len(spkmat)):
                for k in range(len(self.simulObj.neurons)):
                    f.write(str(spkmat[i,k]) + ' ')
                f.write('\n')
            f.close()
#-------------------------------------------------------------------------------
    def doPlotInputs(self,i):
        #clear the figure containing the inputs
        self.inputSpkFigure.clear()
        if i == 0:
            #plot inputs
            self.inputSpkAxes = self.inputSpkFigure.add_subplot(1,1,1)
            for k in range(TBCONSTS.NTAXELS):
                self.inputSpkAxes.plot(self.simulObj.timev,self.simulObj.I[k])
            self.inputSpkAxes.set_xlim([self.simulObj.timev[0],self.simulObj.timev[-1]])
            self.inputSpkAxes.set_xlabel('Time (ms)')
        elif i==1:
            #plot inputs
            for k in range(TBCONSTS.NTAXELS):
                self.inputSpkAxes = self.inputSpkFigure.add_subplot(4,4,k+1)
                self.inputSpkAxes.plot(self.simulObj.timev,self.simulObj.I[k])
                self.inputSpkAxes.set_xlim([self.simulObj.timev[0],self.simulObj.timev[-1]])
        self.inputSpkCanvas.draw()
#-------------------------------------------------------------------------------
    def plotSpikes(self,i):
        #clear the figure containing the results of the simulation
        self.simulResultsFigure.clear()
        if i == 0:
            #plot results
            for k in range(TBCONSTS.NTAXELS):
                self.simulResultsAxes = self.simulResultsFigure.add_subplot(4,4,k+1)
                self.simulResultsAxes.plot(self.simulObj.timen[k],self.simulObj.vneurons[k])
                self.simulResultsAxes.set_xlim([self.simulObj.timev[0],self.simulObj.timev[-1]])
        elif i == 1:
            self.simulResultsAxes = self.simulResultsFigure.add_subplot(1,1,1)
            for k in range(TBCONSTS.NTAXELS):
                self.simulResultsAxes.scatter(self.rasters.xvals[k],self.rasters.yvals[k],marker='|',color='k')
            self.simulResultsAxes.set_ylim([0,TBCONSTS.NTAXELS])
            self.simulResultsAxes.set_xlim([self.simulObj.timev[0],self.simulObj.timev[-1]])
            self.simulResultsAxes.set_xlabel('Time (ms)')
            self.simulResultsAxes.set_ylabel('Neuron id')
        self.simulResultsCanvas.draw()
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Run the app
if __name__ == '__main__':
    import sys
    from PyQt5 import QtCore, QtGui, QtWidgets
    app = QtWidgets.QApplication(sys.argv)
    main = FormSpikeAnalysis()
    main.show()
    sys.exit(app.exec_())
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
