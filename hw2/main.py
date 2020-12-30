import sys
#from PyQt5 import QtWidgets
from hw2_ui import Ui_Form
from PyQt5.QtWidgets import QApplication, QMainWindow
import Q1
import Q2
import Q3
import Q4
import Q5

class window(QMainWindow):
    def __init__(self):
        super(window, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(Q1.drawContour)
        self.ui.pushButton_2.clicked.connect(self.q1setText)
        self.ui.pushButton_4.clicked.connect(question4.selectpoint)
        self.ui.pushButton_5.clicked.connect(Q2.findcorner)
        self.ui.pushButton_6.clicked.connect(Q2.IntrinsicMatrix)
        self.ui.pushButton_7.clicked.connect(Q2.DistortionMatrix)
        self.ui.pushButton_3.clicked.connect(Q3.AugmentedReality)
        self.ui.pushButton_8.clicked.connect(lambda:Q2.ExtrinsicMatrix(self.ui.comboBox.currentText()))
        self.ui.pushButton_9.clicked.connect(Q5.showaccurancy)
        self.ui.pushButton_10.clicked.connect(Q5.showscreenshot)
        self.ui.pushButton_11.clicked.connect(Q5.prediction)
        self.ui.pushButton_12.clicked.connect(Q5.showresize)
    def q1setText(self):
        self.ui.label_1.setText(Q1.countcoin1())
        self.ui.label_2.setText(Q1.countcoin2())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    question4 = Q4.disparityimg()
    win = window()
    win.show()
    sys.exit(app.exec_())


