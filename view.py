import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from util import *


class Visualization_Form(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)

        self.ui = uic.loadUi("./ui/View.ui", self)
        self.initUI()

    def initUI(self):
        self.ui.setWindowTitle('Visualization')
        self.ui.show()

        self.btn.clicked.connect(self.get_file)

    def get_file(self):
        file_name = QFileDialog.getOpenFileName()
        self.label.setText(file_name[0])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = View()
    sys.exit(app.exec())
