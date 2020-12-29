import sys
#from PyQt5 import QtWidgets
from hw2_ui import Ui_Form
from PyQt5.QtWidgets import QApplication, QMainWindow
import Q4
import Q1
class window(QMainWindow):
    def __init__(self):
        super(window, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.pushButton_4.clicked.connect(question4.selectpoint)
        self.ui.pushButton.clicked.connect(Q1.drawContour)
        self.ui.pushButton_2.clicked.connect(self.q1setText)
        # self.ui.pushButton1_2.clicked.connect(Question_1.colorsep)
        # self.ui.pushButton1_3.clicked.connect(Question_1.flipping)
        # self.ui.pushButton1_4.clicked.connect(Question_1.blending)
        # self.ui.pushButton2_1.clicked.connect(Question_2.medianfilter)
        # self.ui.pushButton2_2.clicked.connect(Question_2.gaussianBlur)
        # self.ui.pushButton2_3.clicked.connect(Question_2.bilateralFilter)
        # self.ui.pushButton3_1.clicked.connect(Question_3.gaussianBlur)
        # self.ui.pushButton3_2.clicked.connect(Question_3.sobelx)
        # self.ui.pushButton3_3.clicked.connect(Question_3.sobely)
        # self.ui.pushButton3_4.clicked.connect(Question_3.magnitude)
        # self.ui.pushButton4.clicked.connect(lambda:Question_4.tsf(self.ui.input4_1.text(), self.ui.input4_2.text(), self.ui.input4_3.text(), self.ui.input4_4.text()))
    def q1setText(self):
        self.ui.label_1.setText(Q1.countcoin1())
        self.ui.label_2.setText(Q1.countcoin2())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    question4 = Q4.disparityimg()
    win = window()
    win.show()
    sys.exit(app.exec_())


