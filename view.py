import sys
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
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

        self.input_btn.clicked.connect(self.get_input_image)
        self.model_btn.clicked.connect(self.get_model)

    def get_input_image(self):
        self.img_path = QFileDialog.getOpenFileName()[0]
        self.input.setPixmap(QPixmap(self.img_path).scaledToWidth(self.input.width()))

    def get_model(self):
        self.model_path = QFileDialog.getOpenFileName()[0]
        self.model_check.setText(self.model_path.split("/")[-1])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Visualization_Form()
    sys.exit(app.exec())
