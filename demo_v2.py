import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMdiArea, QMdiSubWindow,
                             QAction, QMenu, QFileDialog, QLabel)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class MDIApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('MDI Image Viewer')
        self.setGeometry(100, 100, 800, 600)

        self.mdiArea = QMdiArea()
        self.setCentralWidget(self.mdiArea)

        self.createMenus()

    def createMenus(self):
        self.fileMenu = self.menuBar().addMenu('File')
        self.openAction = QAction('Open', self)
        self.openAction.triggered.connect(self.openImage)
        self.fileMenu.addAction(self.openAction)

        self.convertAction = QAction('Convert to BMP', self)
        self.convertAction.triggered.connect(self.convertToBMP)
        self.fileMenu.addAction(self.convertAction)

        self.exitAction = QAction('Exit', self)
        self.exitAction.triggered.connect(self.close)
        self.fileMenu.addAction(self.exitAction)

        self.editMenu = self.menuBar().addMenu('Edit')
        self.invertAction = QAction('Invert', self)
        self.invertAction.triggered.connect(self.invertImage)
        self.editMenu.addAction(self.invertAction)

        self.rotateAction = QAction('Rotate', self)
        self.rotateAction.triggered.connect(self.rotateImage)
        self.editMenu.addAction(self.rotateAction)

        self.mirrorAction = QAction('Mirror', self)
        self.mirrorAction.triggered.connect(self.mirrorImage)
        self.editMenu.addAction(self.mirrorAction)

    def openImage(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.bmp *.jpg *.png);;All Files (*)', options=options)
        if file_path:
            image = cv2.imread(file_path)
            self.current_image = image
            
            self.showImage(image, file_path)

    def showImage(self, image, title):
        height, width, channel = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(image, width, height, channel*width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        label = QLabel()
        label.setPixmap(pixmap)

        subWindow = QMdiSubWindow()
        subWindow.setWindowTitle(title)
        subWindow.setWidget(label)
        subWindow.setAttribute(Qt.WA_DeleteOnClose)
        self.mdiArea.addSubWindow(subWindow)
        subWindow.show()

    def convertToBMP(self):
        if hasattr(self, 'current_image'):
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'BMP Files (*.bmp);;All Files (*)', options=options)
            if file_path:
                cv2.imwrite(file_path, self.current_image)

    def invertImage(self):
        if hasattr(self, 'current_image'):
            inverted_image = cv2.bitwise_not(self.current_image)
            self.showImage(inverted_image, 'Inverted')

    def rotateImage(self):
        if hasattr(self, 'current_image'):
            rows, cols = self.current_image.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
            rotated_image = cv2.warpAffine(self.current_image, M, (cols, rows))
            self.showImage(rotated_image, 'Rotated')

    def mirrorImage(self):
        if hasattr(self, 'current_image'):
            mirrored_image = cv2.flip(self.current_image, 1)
            self.showImage(mirrored_image, 'Mirrored')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mdiApp = MDIApp()
    mdiApp.show()
    sys.exit(app.exec_())
