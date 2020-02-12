import sys
import logging
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
from vis.cam import CAM
from vis.grad_cam import GradCAM
from util import use_theme


class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = parent
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
        self.isInput = False
        self.vis = None

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
        self.img_path, i = QFileDialog.getOpenFileName(self, 'Open file', '.', "Image files (*.jpg *.gif *.png)")

        if not self.img_path:
            logging.info("\nNot Selected Image")
            self.isInput = False

            self.input_img.setPixmap(QPixmap(None))

        else:
            logging.info("\nInput Image")
            self.main_label.setText("시작 버튼을 누르세요")
            self.isInput = True

            self.input_img.setPixmap(QPixmap(self.img_path).scaledToWidth(self.input.width()))
        
    def start(self):
        if self.isInput is False:
            QMessageBox.information(self, '메세지', "이미지를 업로드 하세요", QMessageBox.Yes)
            return
        self.layers_widget.clear()

        mode = self.mode_box.currentText()
        model_name = self.model_box.currentText()
        cls = self.cls_box.value()

        logging.info("\nMode : %s, Model : %s" % (mode, model_name))

        if mode == 'cam':
            self.vis = CAM(self.img_path,
                           self.label_path,
                           model_name=model_name)
            info = self.vis.save_img(cls)
            img = self.vis.load_img()

            self.drawing(img)

        elif mode == 'grad cam':
            self.main_label.setText("레이어를 선택하세요")

            self.vis = GradCAM(self.img_path,
                               self.label_path,
                               model_name=model_name)
            info = self.vis.save_img(cls)

            # set list view
            self.set_layers(self.vis.layer_names)
        else:
            info = None

        logging.info(info)

    def set_layers(self, layer_names):
        for name in layer_names:
            item = QListWidgetItem(name, self.layers_widget)
            custom_widget = QLabel(name)
            item.setSizeHint(custom_widget.sizeHint())
            self.layers_widget.setItemWidget(item, custom_widget)
            self.layers_widget.addItem(item)

        self.layers_widget.itemSelectionChanged.connect(self.layer_click)

    def layer_click(self):
        num_row = self.layers_widget.currentRow()
        layer_name = self.layers_widget.currentItem().text()

        # set image
        img = self.vis.load_img("%d_%s" % (num_row, layer_name))

        self.drawing(img)

    def drawing(self, img):
        h, w, c = img.shape
        qImg = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)

        self.main_label.setPixmap(pixmap.scaledToWidth(self.main_label.width()))


if __name__ == '__main__':
    app = QApplication(sys.argv)

    use_theme(app, "theme/darkgray.qss")

    w = Visualization_Form()
    sys.exit(app.exec())
