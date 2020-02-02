import sys
from PyQt5 import QtWidgets
from login import Login_Form

app = QtWidgets.QApplication(sys.argv)
w = Login_Form()
sys.exit(app.exec())