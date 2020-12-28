import sys
#from PyQt5 import QtWidgets
from UIFrame import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
import Question_1
import Question_2
import Question_3
import Question_4

class window(QMainWindow):
    def __init__(self):
        super(window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton1_1.clicked.connect(Question_1.loadImage)
        self.ui.pushButton1_2.clicked.connect(Question_1.colorsep)
        self.ui.pushButton1_3.clicked.connect(Question_1.flipping)
        self.ui.pushButton1_4.clicked.connect(Question_1.blending)
        self.ui.pushButton2_1.clicked.connect(Question_2.medianfilter)
        self.ui.pushButton2_2.clicked.connect(Question_2.gaussianBlur)
        self.ui.pushButton2_3.clicked.connect(Question_2.bilateralFilter)
        self.ui.pushButton3_1.clicked.connect(Question_3.gaussianBlur)
        self.ui.pushButton3_2.clicked.connect(Question_3.sobelx)
        self.ui.pushButton3_3.clicked.connect(Question_3.sobely)
        self.ui.pushButton3_4.clicked.connect(Question_3.magnitude)
        self.ui.pushButton4.clicked.connect(lambda:Question_4.tsf(self.ui.input4_1.text(), self.ui.input4_2.text(), self.ui.input4_3.text(), self.ui.input4_4.text()))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = window()
    win.show()
    sys.exit(app.exec_())


