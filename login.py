import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from util import *
from view import Visualization_Form


class Sign_Up_Form(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)

        self.ui = uic.loadUi("./ui/Sign_up.ui", self)
        self.initUI()

        self.name = None
        self.id = None
        self.pw = None

    def initUI(self):
        self.ui.setWindowTitle('회원가입')

        self.ui.ok_btn.clicked.connect(self.ok_btn_clicked)
        self.ui.back_btn.clicked.connect(self.back_btn_clicked)

    def ok_btn_clicked(self):
        self.name = self.line_name.text()
        self.id = self.line_id.text()
        self.pw = self.line_pw.text()
        self.close()

    def back_btn_clicked(self):
        self.close()


# TODO : DB
class Login_Form(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)

        self.ui = uic.loadUi("./ui/Login_Form.ui", self)
        self.initUI()

        self.admin_id = "admin"
        self.admin_name = "admin"
        self.admin_pw = "admin"

    def initUI(self):
        self.ui.setWindowTitle('로그인')
        self.ui.show()

        self.ui.sign_up_btn.clicked.connect(self.sign_up_btn_clicked)
        self.ui.login_btn.clicked.connect(self.login_btn_clicked)

    def login_btn_clicked(self):
        input_id = self.line_id.text()
        input_pw = self.line_pw.text()

        if self.admin_id == input_id and self.admin_pw == input_pw:
            Visualization_Form()
            self.close()
        elif isEmpty(input_id):
            QMessageBox.about(self, "알림", "아이디를 입력하세요")
        elif isEmpty(input_pw):
            QMessageBox.about(self, "알림", "비밀번호를 입력하세요")
        else:
            QMessageBox.about(self, "알림", "아이디 혹은 비밀번호가 틀렸습니다.")

    def sign_up_btn_clicked(self):
        sign_up = Sign_Up_Form()
        sign_up.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Login_Form()
    app.exec_()