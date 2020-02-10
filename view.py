import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic
from vis.cam import CAM
import logging
from util import isEmpty


class QTextEditLogger(logging.Handler):
    def __init__(self, plain_text):
        super().__init__()
        self.widget = plain_text
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)


class Visualization_Form(QDialog, QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.label_path = "./labels/imagenet_labels.pkl"
        self.ui = uic.loadUi("./ui/View.ui", self)
        self.initUI()

    def initUI(self):
        self.ui.setWindowTitle('Visualization')
        self.ui.show()

        logTextBox = QTextEditLogger(self.plainTextEdit)
        # You can format what is printed to text box
        logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(logTextBox)
        # You can control the logging level
        logging.getLogger().setLevel(logging.DEBUG)

        self.input_btn.clicked.connect(self.get_input_image)
        self.start_btn.clicked.connect(self.start)

    def get_input_image(self):
        self.img_path = QFileDialog.getOpenFileName()[0]
        self.input_img.setPixmap(QPixmap(self.img_path).scaledToWidth(self.input.width()))
        # TODO
        self.main_label.setText("시작 버튼을 누르세요")
        logging.info("Input Image")
        
    def start(self):
        if isEmpty(self.input_img.text()):
            QMessageBox.information(self, '메세지', "이미지를 업로드 하세요", QMessageBox.Yes)
            return

        mode = self.mode_box.currentText()
        model_name = self.model_box.currentText()
        cls = self.cls_box.value()

        logging.info("\nMode : %s, Model : %s" % (mode, model_name))

        if mode == 'cam':
            cam = CAM(self.img_path,
                      self.label_path,
                      model_name=model_name)

            img, info = cam.get_img(cls)

            logging.info(info)

        h, w, c = img.shape
        qImg = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)

        self.main_label.setPixmap(pixmap.scaledToWidth(self.main_label.width()))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Visualization_Form()
    sys.exit(app.exec())
