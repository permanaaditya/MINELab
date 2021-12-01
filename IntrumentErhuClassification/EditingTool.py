import os
import sys
import shutil
import cv2
import math
import time
import numpy as np
import argparse
import util.Util as Util
import pyaudio
import wave
import threading
import time
import subprocess
from datetime import datetime
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio

from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QAction, QStatusBar, QPushButton, QLabel, QSlider, QTextEdit, QFileDialog, QVBoxLayout, QStyle, QMessageBox, QTableWidget, QLineEdit, QTableWidgetItem, QComboBox
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QDir, Qt, QUrl, QSize

class EditingToolWindow(QtWidgets.QMainWindow) :
    def __init__(self) :
        super(EditingToolWindow, self).__init__()
        uic.loadUi('util/editingtool.ui', self)
        self.keyPressEvent = self.MainKeyPressed

        self.filePath = ""
        self.directoryPath = ""
        self.directoryOutputPath = ""
        self.baseName = ""
        self.pathRGB_1 = None
        self.pathRGB_2 = None
        self.pathDepthVideo = None
        self.pathDepth = None
        self.pathAudio = None
        self.pathDataTable = None

        self.inputRGB_1 = []
        self.inputRGB_2 = []
        self.inputDepthVideo = []
        self.inputDepth = []
        self.inputAudio = None
        self.inputAudioPlayer = None
        self.inputAudioSize = 0

        self.isPlaying = False
        self.frameCount = 0
        self.frameIndex = 0
        self.FPS = 30

        self.lastSelect = 0
        self.tableLoading = False

        self.classesTone = []

        self.initUI()
        self.show()

    def initUI(self) :
        self.setWindowIcon(QtGui.QIcon('util/icon.png'))

        self.centralwidget = self.findChild(QWidget, 'centralwidget')

        self.outputLabel = self.findChild(QLabel, 'outputLabel')
        self.outputLabel.setStyleSheet("background-color: black")
        self.outputLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.tableWidget = self.findChild(QTableWidget, 'tableWidget')
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.tableWidget.itemChanged.connect(self.changeCell)
        self.tableWidget.keyPressEvent = self.KeyPressed
        self.tableWidget.itemSelectionChanged.connect(self.tableSelectionChanged)

        self.dataFrame = self.findChild(QLabel, 'dataFrame')
        self.dataFrame.setAlignment(QtCore.Qt.AlignCenter)
        self.dataSlider = self.findChild(QSlider, 'dataSlider')
        self.dataSlider.valueChanged.connect(self.dataSliderChange)
        self.dataTimer = self.findChild(QLabel, 'dataTimer')
        self.dataTimer.setAlignment(QtCore.Qt.AlignCenter)

        self.playBtn = self.findChild(QPushButton, 'playBtn')
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.setFocus()

        self.stopBtn = self.findChild(QPushButton, 'stopBtn')
        self.stopBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))

        self.playBtn.clicked.connect(self.playData)
        self.stopBtn.clicked.connect(self.stopData)

        self.startFrameNumber = self.findChild(QLineEdit, 'startFrameNumber')
        self.startFrameNumber.setFixedWidth(80)
        self.startFrameNumber.setAlignment(QtCore.Qt.AlignRight)
        self.endFrameNumber = self.findChild(QLineEdit, 'endFrameNumber')
        self.endFrameNumber.setFixedWidth(80)
        self.endFrameNumber.setAlignment(QtCore.Qt.AlignRight)
        self.onlyInt = QtGui.QIntValidator()
        self.startFrameNumber.setValidator(self.onlyInt)
        self.endFrameNumber.setValidator(self.onlyInt)
        self.playRangeBtn = self.findChild(QPushButton, 'playRangeBtn')
        self.playRangeBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        self.playRangeBtn.clicked.connect(self.playRange)
        self.cutBtn = self.findChild(QPushButton, 'cutBtn')
        self.cutBtn.clicked.connect(self.cutData)

        self.startFrameNumber.installEventFilter(self)
        self.endFrameNumber.installEventFilter(self)
        self.playBtn.installEventFilter(self)
        self.stopBtn.installEventFilter(self)
        self.playRangeBtn.installEventFilter(self)
        self.cutBtn.installEventFilter(self)
        self.tableWidget.installEventFilter(self)

        self.actionBtnOpen = self.findChild(QAction, 'actionBtnOpen')
        self.actionBtnOpen.triggered.connect(self.actionBtnOpenEvent)
        self.actionBtnSave = self.findChild(QAction, 'actionBtnSave')
        self.actionBtnSave.triggered.connect(self.actionBtnSaveEvent)
        self.actionBtnExit = self.findChild(QAction, 'actionBtnExit')
        self.actionBtnExit.triggered.connect(self.actionExitBtnEvent)

        self.statusbar = self.findChild(QStatusBar, 'statusbar')

        if os.path.exists("util/classes.txt") :
            file = open("util/classes.txt", "r")
            self.classesTone = [line.strip() for line in file.readlines()]

            combobox = QComboBox()
            for tone in self.classesTone:
                combobox.addItem(tone)
            combobox.currentIndexChanged.connect(self.changeCombobox)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, self.tableWidget.columnCount() - 1, combobox)

    def MainKeyPressed(self, event):
        if event.key() == QtCore.Qt.Key_S:
            self.playData();
        elif event.key() == QtCore.Qt.Key_D:
            self.stopData()
        elif event.key() == QtCore.Qt.Key_R:
            self.playRange()

    def initData(self) :
        print("Files :")
        print("RGB 1 : {}".format(self.pathRGB_1))
        print("RGB 2 : {}".format(self.pathRGB_2))
        print("DepthVideo : {}".format(self.pathDepthVideo))
        print("Depth : {}".format(self.pathDepth))
        print("Audio : {}".format(self.pathAudio))

        input = cv2.VideoCapture(os.path.join(self.directoryPath, self.pathRGB_1))
        properties = Util.getVideoProperties(input)
        self.frameCount = properties['Count']
        print("RGB 1 frame count : {}".format(self.frameCount))

        while input.isOpened() :
            res, frame = input.read()
            if res is False :
                break
            self.inputRGB_1.append(frame)
        input.release()

        input = cv2.VideoCapture(os.path.join(self.directoryPath, self.pathRGB_2))
        properties = Util.getVideoProperties(input)
        print("RGB 2 frame count : {}".format(properties['Count']))

        while input.isOpened() :
            res, frame = input.read()
            if res is False :
                break
            self.inputRGB_2.append(frame)
        input.release()

        input = cv2.VideoCapture(os.path.join(self.directoryPath, self.pathDepthVideo))
        properties = Util.getVideoProperties(input)
        print("Depth video frame count : {}".format(properties['Count']))

        while input.isOpened() :
            res, frame = input.read()
            if res is False :
                break
            self.inputDepthVideo.append(frame)
        input.release()

        depthData = np.load(os.path.join(self.directoryPath, self.pathDepth))["arr_0"]
        print("Depth frame count : {}".format(len(depthData)))

        for data in depthData :
            self.inputDepth.append(data)

        self.inputAudio = AudioSegment.from_mp3(os.path.join(self.directoryPath, self.pathAudio))
        self.inputAudioSize = len(self.inputAudio)

        self.frameIndex = 0
        self.dataSlider.setRange(0, self.frameCount - 1)
        self.changeFrameStatus()
        self.showFirstData()

        self.statusbar.showMessage("Opened : {}".format(self.baseName))

    def showFirstData(self) :
        if self.frameCount > 0 :
            self.show_image()

    def playData(self) :
        if self.inputAudio is None :
            self.actionBtnOpenEvent()
            return

        if self.isPlaying :
            return

        self.isPlaying = True

        if self.inputAudio is not None :
            currentTime = int(self.inputAudioSize * (self.frameIndex / self.frameCount))
            self.inputAudioPlayer = _play_with_simpleaudio(self.inputAudio[currentTime:])

        startTime = time.time()
        currentLocation = self.frameIndex
        while self.frameCount > 0 and self.isPlaying and 0 <= self.frameIndex and self.frameIndex < self.frameCount :
            self.show_image()
            self.changeFrameStatus()
            self.frameIndex = math.floor(currentLocation + (time.time() - startTime) * 1000 / self.inputAudioSize * self.frameCount)

        if self.isPlaying :
            self.isPlaying = False
            self.frameIndex = 0
            self.changeFrameStatus()

    def playRange(self) :
        if self.isPlaying or self.startFrameNumber.text() == '' or self.endFrameNumber.text() == "":
            return

        start = int(self.startFrameNumber.text())
        end = int(self.endFrameNumber.text())
        if start < 0 or start >= end :
            return

        self.isPlaying = True
        self.frameIndex = start

        if self.inputAudio is not None :
            currentTime = int(self.inputAudioSize * (start / self.frameCount))
            endTime = int(self.inputAudioSize * (end / self.frameCount))
            self.inputAudioPlayer = _play_with_simpleaudio(self.inputAudio[currentTime:endTime])

        startTime = time.time()
        currentLocation = self.frameIndex
        while self.frameCount > 0 and self.isPlaying and 0 <= self.frameIndex and self.frameIndex < end + 1 and self.frameIndex < self.frameCount:
            self.show_image()
            self.changeFrameStatus()
            self.frameIndex = math.floor(currentLocation + (time.time() - startTime) * 1000 / self.inputAudioSize * self.frameCount)

        if self.isPlaying :
            self.isPlaying = False
            self.frameIndex = start
            self.changeFrameStatus()

    def stopData(self) :
        self.isPlaying = False

        if self.inputAudioPlayer is not None :
            self.inputAudioPlayer.stop()

    def dataSliderChange(self, position) :
        if self.frameCount > 0 :
            self.frameIndex = position
            self.show_image()
            self.changeFrameStatus()

            if self.lastSelect == 1 :
                self.startFrameNumber.setText(str(self.frameIndex))
            elif self.lastSelect == 2 :
                self.endFrameNumber.setText(str(self.frameIndex))

    def show_image(self) :
        maxwidth, maxheight = self.outputLabel.width(), self.outputLabel.height()

        depth = cv2.applyColorMap(cv2.convertScaleAbs(self.inputDepth[self.frameIndex], alpha=0.03), cv2.COLORMAP_JET)
        rgb_depth = np.concatenate((self.inputRGB_2[self.frameIndex], depth), axis=1)

        h1, w1 = self.inputRGB_1[self.frameIndex].shape[:2]
        h2, w2 = rgb_depth.shape[:2]

        image = np.zeros((h1 + h2, w2, 3), np.uint8)

        image[: h1, int(w1/2) : int(w1/2) + w1] = self.inputRGB_1[self.frameIndex]
        image[h1 : h1 + h2, : w2] = rgb_depth

        f1 = maxwidth / image.shape[1]
        f2 = maxheight / image.shape[0]
        f = min(f1, f2)
        dim = (int(image.shape[1] * f), int(image.shape[0] * f))

        image = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        image = image.scaled(dim[0], dim[1], Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.outputLabel.setPixmap(QtGui.QPixmap.fromImage(image))

        cv2.waitKey(1)


    def changeFrameStatus(self) :
        if self.frameCount > 0 :
            self.dataSlider.setValue(self.frameIndex)
            self.dataFrame.setText("{}/{}".format(self.frameIndex, self.frameCount - 1))

            currentTime = int(self.inputAudioSize * (self.frameIndex / self.frameCount))
            self.dataTimer.setText("{}/{}".format(Util.hhmmss(currentTime), Util.hhmmss(self.inputAudioSize)))

    def actionBtnSaveEvent(self) :
        if self.frameCount == 0 :
            return

        reply = QMessageBox.question(self, 'Message', 'Are you sure you want to save?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            ''
        else:
            return

        self.statusbar.showMessage("Saving data. [{}]".format(datetime.now().strftime("%H:%M:%S")))

        os.makedirs(self.directoryOutputPath, exist_ok=True)
        for filename in os.listdir(self.directoryOutputPath):
            file_path = os.path.join(self.directoryOutputPath, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        outputPath = os.path.join(self.directoryOutputPath, self.pathRGB_1)
        output = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'MP4V'), self.FPS, (self.inputRGB_1[0].shape[1], self.inputRGB_1[0].shape[0]))
        for data in self.inputRGB_1 :
            output.write(data)
        output.release()

        outputPath = os.path.join(self.directoryOutputPath, self.pathRGB_2)
        output = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'MP4V'), self.FPS, (self.inputRGB_2[0].shape[1], self.inputRGB_2[0].shape[0]))
        for data in self.inputRGB_2 :
            output.write(data)
        output.release()

        outputPath = os.path.join(self.directoryOutputPath, self.pathDepthVideo)
        output = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'MP4V'), self.FPS, (self.inputDepthVideo[0].shape[1], self.inputDepthVideo[0].shape[0]))
        for data in self.inputDepthVideo :
            output.write(data)
        output.release()

        outputPath = os.path.join(self.directoryOutputPath, self.pathDepth)
        np.savez(outputPath, self.inputDepth)

        outputPath = os.path.join(self.directoryOutputPath, self.pathAudio)
        self.inputAudio.export(outputPath, format="wav")

        outputPath = os.path.join(self.directoryOutputPath, self.baseName + ".txt")
        file = open(outputPath,"w")

        for i in range(self.tableWidget.rowCount()) :
            emptyColFound = True
            line = ""
            for j in range(self.tableWidget.columnCount()) :
                if j < self.tableWidget.columnCount() - 1:
                    if self.tableWidget.item(i , j) is not None and self.tableWidget.item(i , j).text() != "":
                        emptyColFound = False
                        line += self.tableWidget.item(i , j).text().strip() + " "
                    else :
                        line += "- "
                else :
                    line += str(self.tableWidget.cellWidget(i, j).currentIndex())
            line += "\n"
            if not emptyColFound :
                file.write(line)

        file.close()

        QMessageBox.warning(self, "Message", "Data is saved!")
        self.statusbar.showMessage("Data is saved. [{}]".format(datetime.now().strftime("%H:%M:%S")))


    def startFrameNumberPress(self) :
        print("startFrameNumberPress")

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.FocusIn:
            if obj is self.startFrameNumber:
                self.lastSelect = 1
            elif obj is self.endFrameNumber:
                self.lastSelect = 2
            else :
                self.lastSelect = 0

        return super(EditingToolWindow, self).eventFilter(obj, event)

    def actionBtnOpenEvent(self) :
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        self.filePath, _ = QFileDialog.getOpenFileName(self, "Choose wav file", "", "Wav Files (*.wav)", options=options)
        if self.filePath:
            fileName = os.path.basename(self.filePath)

            self.isPlaying = False
            self.frameCount = 0
            self.frameIndex = 0

            self.pathRGB_1 = None
            self.pathRGB_2 = None
            self.pathDepthVideo = None
            self.pathDepth = None
            self.pathAudio = None

            self.inputRGB_1 = []
            self.inputRGB_2 = []
            self.inputDepthVideo = []
            self.inputDepth = []
            self.inputAudio = None

            self.isPlaying = False

            if "_" in fileName :
                self.directoryPath = os.path.dirname(self.filePath)
                self.baseName = fileName.split('_')[0]

                print("Directory : {}".format(self.directoryPath))

                self.directoryOutputPath = os.path.join(self.directoryPath, self.baseName + "_edit")
                print("Output directory : {}".format(self.directoryOutputPath))

                self.pathDataTable = os.path.join(self.directoryPath, self.baseName + ".txt")
                print("Output table data : {}".format(self.pathDataTable))

                fileList = os.listdir(self.directoryPath)

                checkCounter = 0
                files = []
                for file in fileList :
                    if self.baseName in file :
                        if not "_edit" in file and not ".txt" in file :
                            files.append(file)
                            checkCounter +=1

                if checkCounter == 5 :
                    for file in files :
                        if "cam_1" in file:
                            self.pathRGB_1 = file
                        if "cam_2" in file:
                            self.pathRGB_2 = file
                        if "depth" in file and not ".npz" in file :
                            self.pathDepthVideo = file
                        if ".npz" in file:
                            self.pathDepth = file
                        if ".wav" in file:
                            self.pathAudio = file
                    self.statusbar.showMessage("Opening : {}".format(self.baseName))
                    self.initData()
                elif checkCounter < 5:
                    QMessageBox.warning(self, "Message", "Some file is missed!")
                else :
                    QMessageBox.warning(self, "Message", "Some file is duplicated!")

                self.tableWidget.clear();
                self.tableWidget.setRowCount(1);

                if os.path.exists(self.pathDataTable) :
                    file = open(self.pathDataTable, "r")
                    lines = file.readlines()
                    self.tableLoading = True

                    for i, line in enumerate(lines) :
                        data = line.strip().split()
                        for j, text in enumerate(data) :
                            if j < self.tableWidget.columnCount() - 1 :
                                self.tableWidget.setItem(i, j, QTableWidgetItem(text))
                            else:
                                combobox = QComboBox()
                                for tone in self.classesTone:
                                    combobox.addItem(tone)
                                combobox.currentIndexChanged.connect(self.changeCombobox)

                                if text.isnumeric() :
                                    combobox.setCurrentIndex(int(text))

                                self.tableWidget.setCellWidget(i, j, combobox)
                        self.tableWidget.insertRow(self.tableWidget.rowCount())

                    file.close()
                    self.tableLoading = False

                combobox = QComboBox()
                for tone in self.classesTone:
                    combobox.addItem(tone)
                combobox.currentIndexChanged.connect(self.changeCombobox)
                self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, self.tableWidget.columnCount() - 1, combobox)

            else :
                QMessageBox.warning(self, "Message", "You've chosen the wrong file!")

    def cutData(self) :
        if self.isPlaying :
            return

        if self.isPlaying or self.startFrameNumber.text() == '' or self.endFrameNumber.text() == "":
            return

        reply = QMessageBox.question(self, 'Message', 'Are you sure you want to cut?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            ''
        else:
            return

        start = int(self.startFrameNumber.text())
        end = int(self.endFrameNumber.text())
        if start < 0 or start > end :
            return

        print("Cutting [{}-{}]".format(start, end))

        if start == 0 and end < self.frameCount - 1:
            print("Cut first section...")

            self.inputRGB_1 = self.inputRGB_1[end:]
            self.inputRGB_2 = self.inputRGB_2[end:]
            self.inputDepthVideo = self.inputDepthVideo[end:]
            self.inputDepth = self.inputDepth[end:]
            self.inputAudio = self.inputAudio[int(self.inputAudioSize * (end / self.frameCount)):]


        elif 0 < start and end < self.frameCount - 1:
            print("Cut middle section...")

            self.inputRGB_1 = self.inputRGB_1[:start] + self.inputRGB_1[end:]
            self.inputRGB_2 = self.inputRGB_2[:start] + self.inputRGB_2[end:]
            self.inputDepthVideo = self.inputDepthVideo[:start] + self.inputDepthVideo[end:]
            self.inputDepth = self.inputDepth[:start] + self.inputDepth[end:]
            self.inputAudio = self.inputAudio[:int(self.inputAudioSize * (start / self.frameCount))] + self.inputAudio[int(self.inputAudioSize * (end / self.frameCount)):]


        elif 0 < start and end == self.frameCount - 1 :
            print("Cut last section...")

            self.inputRGB_1 = self.inputRGB_1[:start]
            self.inputRGB_2 = self.inputRGB_2[:start]
            self.inputDepthVideo = self.inputDepthVideo[:start]
            self.inputDepth = self.inputDepth[:start]
            self.inputAudio = self.inputAudio[:int(self.inputAudioSize * (start / self.frameCount))]

        else :
            print("All range is selected...")
            return

        self.inputAudioSize = len(self.inputAudio)

        self.frameCount = len(self.inputRGB_1)
        self.dataSlider.setRange(0, self.frameCount - 1)
        self.frameIndex = 0
        self.changeFrameStatus()
        self.showFirstData()

        self.startFrameNumber.setText('')
        self.endFrameNumber.setText('')

    def changeCell(self, cell) :
        if not self.tableLoading :
            self.checkEmptyRow()

    def changeCombobox(self, index) :
        if not self.tableLoading :
            self.checkEmptyRow()

    def checkEmptyRow(self):
        emptyRowFound = False
        for i in range(self.tableWidget.rowCount()) :
            emptyColFound = True
            for j in range(self.tableWidget.columnCount()) :
                if self.tableWidget.item(i , j) is not None and self.tableWidget.item(i , j).text() != "":
                    emptyColFound = False
                    break

            if emptyColFound :
                emptyRowFound = True
                break

        if emptyRowFound is False :
            self.tableWidget.setRowCount(self.tableWidget.rowCount() + 1);

            combobox = QComboBox()
            for tone in self.classesTone:
                combobox.addItem(tone)
            combobox.currentIndexChanged.connect(self.changeCombobox)
            self.tableWidget.setCellWidget(self.tableWidget.rowCount() - 1, self.tableWidget.columnCount() - 1, combobox)

    def KeyPressed(self, event):
        if event.key() == QtCore.Qt.Key_Delete:
            reply = QMessageBox.question(self, 'Message', 'Are you sure you delete selected row?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                ''
            else:
                return

            self.tableLoading = True
            if self.tableWidget.rowCount() == 1 :
                for j in range(self.tableWidget.columnCount()) :
                    self.tableWidget.setItem(self.tableWidget.currentRow(), j, QTableWidgetItem(''))
            elif self.tableWidget.rowCount() > 1 :
                self.tableWidget.removeRow(self.tableWidget.currentRow())
            self.tableLoading = False

    def actionExitBtnEvent(self) :
        self.close()

    def closeEvent(self,event):
        exit()

    def resizeEvent(self, event) :
        self.outputLabel.setFixedWidth(self.centralwidget.width() - 350)
        self.tableWidget.setFixedWidth(300)
        self.showFirstData()

    def tableSelectionChanged(self):
        try:
            if len(self.tableWidget.selectedItems()) == 2 and (int(self.tableWidget.item(self.tableWidget.currentRow(), 0).text()) < int(self.tableWidget.item(self.tableWidget.currentRow(), 1).text())):
                self.startFrameNumber.setText(self.tableWidget.item(self.tableWidget.currentRow(), 0).text())
                self.endFrameNumber.setText(self.tableWidget.item(self.tableWidget.currentRow(), 1).text())
        except:
            print("Something wrong!")

if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = EditingToolWindow()
    app.exec_()
