import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMdiArea, QMdiSubWindow, QDialog, QDialogButtonBox,
                             QAction, QMenu, QFileDialog, QLabel, QInputDialog, QFormLayout, QLineEdit)
from PyQt5.QtGui import QImage, QPixmap, QDoubleValidator
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt

from myfft import my_fft2d, my_ifft2d, homomorphic_filter

class FloatDialog(QDialog):
    def __init__(self, title = ''):
        super().__init__()
        
        self.initUI(title)
        
    def initUI(self, title):
        self.setWindowTitle('please input ' + title)
        self.setGeometry(200, 200, 400, 100)
        
        self.layout = QFormLayout()
        
        self.floatInput1 = QLineEdit(self)
        self.floatInput1.setValidator(QDoubleValidator())
        self.layout.addRow('x:', self.floatInput1)
        
        self.floatInput2 = QLineEdit(self)
        self.floatInput1.setValidator(QDoubleValidator())
        self.layout.addRow('y:', self.floatInput2)
        
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        self.layout.addWidget(self.buttonBox)
        
        self.setLayout(self.layout)
        
    def getInputs(self):
        return float(self.floatInput1.text()), float(self.floatInput2.text())
    
    @staticmethod
    def getFInputs(title):
        dialog = FloatDialog(title)
        if dialog.exec() == QDialog.Accepted:
            return float(dialog.floatInput1.text()), float(dialog.floatInput2.text())
        else:
            return None, None

class MDIApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('MDI Image Viewer')
        self.setGeometry(100, 100, 1000, 1000)

        self.mdiArea = QMdiArea()
        self.setCentralWidget(self.mdiArea)

        self.createMenus()

    def createMenus(self):
        # =================FILE MENU======================
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

        # ================EDIT MENU=======================
        self.editMenu = self.menuBar().addMenu('Edit')
        self.invertAction = QAction('Invert', self)
        self.invertAction.triggered.connect(self.invertImage)
        self.editMenu.addAction(self.invertAction)

        self.translateAction = QAction('Translate', self)
        self.translateAction.triggered.connect(self.translateImage)
        self.editMenu.addAction(self.translateAction)

        self.rotateAction = QAction('Rotate', self)
        self.rotateAction.triggered.connect(self.rotateImage)
        self.editMenu.addAction(self.rotateAction)

        self.mirrorAction = QAction('Mirror', self)
        self.mirrorAction.triggered.connect(self.mirrorImage)
        self.editMenu.addAction(self.mirrorAction)

        self.fftAction = QAction('FFT', self)
        self.fftAction.triggered.connect(self.imageFFT)
        self.editMenu.addAction(self.fftAction)

        self.ifftAction = QAction('IFFT', self)
        self.ifftAction.triggered.connect(self.imageIFFT)
        self.editMenu.addAction(self.ifftAction)
        # ===================ENHANCE MENU=========================
        self.enhanceMenu = self.menuBar().addMenu('Enhance')
        self.histEqAction = QAction('Hist Equalize', self)
        self.histEqAction.triggered.connect(self.histEqualize)
        self.enhanceMenu.addAction(self.histEqAction)

        self.homoFilterAction = QAction('Homomorphic Filter', self)
        self.homoFilterAction.triggered.connect(self.homomorphicFilter)
        self.enhanceMenu.addAction(self.homoFilterAction)

        self.expTransformAction = QAction('Exp Transform', self)
        self.expTransformAction.triggered.connect(self.expTransform)
        self.enhanceMenu.addAction(self.expTransformAction)

        self.laplaceSharpenAction = QAction('Laplace Sharpen', self)
        self.laplaceSharpenAction.triggered.connect(self.laplaceSharpen)
        self.enhanceMenu.addAction(self.laplaceSharpenAction)
        # ==================EDGE MENU=============================
        self.edgeMenu = self.menuBar().addMenu('Edge')
        self.robertsAction = QAction('Roberts', self)
        self.robertsAction.triggered.connect(self.robertsEdge)
        self.edgeMenu.addAction(self.robertsAction)

        self.sobelAction = QAction('Sobel', self)
        self.sobelAction.triggered.connect(self.sobelEdge)
        self.edgeMenu.addAction(self.sobelAction)

        self.prewittAction = QAction('Prewitt', self)
        self.prewittAction.triggered.connect(self.prewittEdge)
        self.edgeMenu.addAction(self.prewittAction)

        self.laplaceAction = QAction('Laplace', self)
        self.laplaceAction.triggered.connect(self.laplaceEdge)
        self.edgeMenu.addAction(self.laplaceAction)
        # ==================DESCRIPTOR MENU=========================
        self.descriptorMenu = self.menuBar().addMenu('Descriptor')
        self.fourierDescriptorAction = QAction('Fourier Descriptor', self)
        self.fourierDescriptorAction.triggered.connect(self.fourierDescriptor)
        self.descriptorMenu.addAction(self.fourierDescriptorAction)
        
    def openImage(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.bmp *.jpg *.png);;All Files (*)', options=options)
        if file_path:
            image = cv2.imread(file_path)
            self.current_image = image
            
            self.showImage(image, file_path)

    def showImage(self, image, title):
        image = np.asarray(image, dtype=np.uint8)
        if len(image.shape) == 2:
            height, width = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            height, width, channel = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(image, width, height, 3*width, QImage.Format_RGB888)
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
            # inverted_image = cv2.bitwise_not(self.current_image)
            inverted_image = 255 - self.current_image
            self.showImage(inverted_image, 'Inverted')

    def rotateImage(self):
        if hasattr(self, 'current_image'):
            angle, _ = QInputDialog.getInt(self, 'rotate angle', 'please input rotate angle')
            rows, cols = self.current_image.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_image = cv2.warpAffine(self.current_image, M, (cols, rows))
            self.showImage(rotated_image, 'Rotated')

    def mirrorImage(self):
        if hasattr(self, 'current_image'):
            mirrored_image = cv2.flip(self.current_image, 1)
            self.showImage(mirrored_image, 'Mirrored')

    def translateImage(self):
        if hasattr(self, 'current_image'):
            # text, _ = QInputDialog.getText(self, 'translate pixel', 'please input translate pixel')
            rows, cols = self.current_image.shape[:2]
            tx, ty = FloatDialog.getFInputs('translate distance')
            if tx is None or ty is None:
                return
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            translated_image = cv2.warpAffine(self.current_image, M, (cols, rows))
            self.showImage(translated_image, 'Translated')

    def imageFFT(self):
        if hasattr(self, 'current_image'):
            to_fft = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
   
            # fft_image = np.fft.fftshift(np.fft.fft2(to_fft))
            # self.current_fft_image = fft_image
            # fft_image = np.log(1 + np.abs(fft_image))
            
            fft_image = my_fft2d(to_fft)
            self.current_fft_image = fft_image
            # fft_visualize
            fft_visual = np.fft.fftshift(fft_image)
            fft_visual = np.log(1 + np.abs(fft_visual))
            fft_visual = fft_visual / np.max(fft_visual) * 255  # 归一化

            self.showImage(fft_visual, 'FFT_image')

    def imageIFFT(self):
        if hasattr(self, 'current_fft_image'):
            to_ifft = self.current_fft_image
        elif hasattr(self, 'current_image'):
            to_ifft = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            return
        ifft_image = np.abs(my_ifft2d(to_ifft))
        self.showImage(ifft_image, 'IFFT_image')

    def histEqualize(self):
        if hasattr(self, 'current_image'):
            to_input = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            eq_img = cv2.equalizeHist(to_input)

            self.showImage(eq_img, "HistEqualised")

            eq_img = cv2.equalizeHist(eq_img)

            self.showImage(eq_img, "HistEqualised_twice")

    def homomorphicFilter(self):
        if hasattr(self, 'current_image'):
            hf_image = homomorphic_filter(self.current_image)
            # hf_image = np.uint8(hf_image / np.max(hf_image) * 255)
            # hf_image = np.uint8(hf_image)
            self.showImage(hf_image, "Homomorphic")

    def expTransform(self):
        if hasattr(self, 'current_image'):
            gamma, _ = QInputDialog.getDouble(self, 'gamma', 'please input gamma of exp transform')
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            after = ((gray / 255) ** gamma) * 255

            self.showImage(after, "ExpTransformed")

    def laplaceSharpen(self):
        if hasattr(self, 'current_image'):
            laplaced = cv2.Laplacian(self.current_image, cv2.CV_64F, ksize=3)
            laplaced = np.uint8(np.abs(laplaced))
            sharpened = cv2.addWeighted(self.current_image, 1.5, laplaced, -0.5, 0)

            self.showImage(sharpened, "LaplaceSharpened")

    def robertsEdge(self):
        if hasattr(self, 'current_image'):
            kernelx = np.array([[1, 0], [0, -1]], dtype=int)
            kernely = np.array([[0, 1], [-1, 0]], dtype=int)
            edge_robertsx = cv2.filter2D(self.current_image, cv2.CV_16S, kernelx)
            edge_robertsy = cv2.filter2D(self.current_image, cv2.CV_16S, kernely)
            edge_roberts = cv2.addWeighted(np.abs(edge_robertsx), 0.5, np.abs(edge_robertsy), 0.5, 0)

            self.showImage(edge_roberts, "RobertsEdge")

    def sobelEdge(self):
        if hasattr(self, 'current_image'):
            edge_sobelx = cv2.Sobel(self.current_image, cv2.CV_64F, 1, 0, ksize=3)
            edge_sobely = cv2.Sobel(self.current_image, cv2.CV_64F, 0, 1, ksize=3)
            edge_sobel = cv2.addWeighted(np.abs(edge_sobelx), 0.5, np.abs(edge_sobely), 0.5, 0)

            self.showImage(edge_sobel, "SobelEdge")

    def prewittEdge(self):
        if hasattr(self, 'current_image'):
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
            edge_prewittx = cv2.filter2D(self.current_image, cv2.CV_16S, kernelx)
            edge_prewitty = cv2.filter2D(self.current_image, cv2.CV_16S, kernely)
            edge_prewitt = cv2.addWeighted(np.abs(edge_prewittx), 0.5, np.abs(edge_prewitty), 0.5, 0)

            self.showImage(edge_prewitt, "PrewittEdge")

    def laplaceEdge(self):
        if hasattr(self, 'current_image'):
            edge_laplace = cv2.Laplacian(self.current_image, cv2.CV_64F, ksize=3)
            edge_laplace = np.uint8(np.abs(edge_laplace))

            self.showImage(edge_laplace, "LaplaceEdge")

    def fourierDescriptor(self):
        if hasattr(self, 'current_image'):
            num, _ = QInputDialog.getInt(self, 'num', 'please input num of descriptors')
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            points = np.argwhere(gray == 255)
            # print(len(points))

            points_fft = np.array([complex(p[0], p[1]) for p in points])
            descriptors = np.fft.fft(points_fft)
            # 选取部分
            descriptors_new = np.zeros_like(descriptors)
            descriptors_new[:num] = descriptors[:num] 
            points_ifft = np.fft.ifft(descriptors_new)

            points_new = np.array([[np.real(p), np.imag(p)] for p in points_ifft], dtype=int)
            
            ret = np.zeros_like(gray, dtype=np.uint8)
            H, W = ret.shape
            for p in points_new:
                y = p[0]
                x = p[1]
                if 0 <= x < H and 0 <= y < W:  # Ensure points are within image bounds
                    ret[x, y] = 255

            self.showImage(ret, "FourierDescriptor")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mdiApp = MDIApp()
    mdiApp.show()
    sys.exit(app.exec_())
