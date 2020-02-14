import sys
import logging
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from vis.cam import CAM, GradCAM
from vis.grad import Vanilla, Smooth, Guided_Backprop
from util import use_theme, make_dir


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
        self.islayer = False

    def initUI(self):
        self.ui.setWindowTitle('Visualization')
        self.ui.show()

        make_dir("./results")

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
            self.main_label.setText("Press the start button")
            self.isInput = True

            self.input_img.setPixmap(QPixmap(self.img_path).scaledToWidth(self.input.width()))
        
    def start(self):
        if self.isInput is False:
            QMessageBox.information(self, 'Message', "Upload the Image", QMessageBox.Yes)
            return

        if self.islayer is True:
            self.layers_widget.itemSelectionChanged.disconnect()
            self.islayer = False

        self.layers_widget.clear()

        mode = self.mode_box.currentText()
        model_name = self.model_box.currentText()
        cls = self.cls_box.value()

        logging.info("\nMode : %s, Model : %s" % (mode, model_name))

        if mode == 'cam':
            cam = CAM(self.img_path,
                      self.label_path,
                      model_name=model_name)
            info = cam.save_img(cls)
            img = cam.load_img()

            self.drawing(img)

        elif mode == 'grad cam':
            self.groupBox.setTitle("Layers")
            self.main_label.setText("Select layer")

            grad_cam = GradCAM(self.img_path,
                               self.label_path,
                               model_name=model_name)
            info = grad_cam.save_img(cls)

            # set list view
            self.set_layers(grad_cam)

        elif mode == 'guided backprop':
            self.groupBox.setTitle("Modes")
            self.main_label.setText("Select mode")

            guided = Guided_Backprop(self.img_path,
                                     self.label_path,
                                     model_name=model_name)
            info = guided.save_img(cls)

            # set list view
            self.set_layers(guided)

        elif mode == 'vanilla grad':
            vanilla = Vanilla(self.img_path,
                              self.label_path,
                              model_name=model_name)
            info = vanilla.save_img(cls)
            img = vanilla.load_img()

            self.drawing(img)

        elif mode == 'smooth grad':
            smooth = Smooth(self.img_path,
                            self.label_path,
                            model_name=model_name)
            info = smooth.save_img(cls)
            img = smooth.load_img()

            self.drawing(img)

        else:
            info = None

        logging.info(info)

    def set_layers(self, vis):
        for name in vis.items:
            item = QListWidgetItem(name, self.layers_widget)
            custom_widget = QLabel(name)
            item.setSizeHint(custom_widget.sizeHint())
            self.layers_widget.setItemWidget(item, custom_widget)
            self.layers_widget.addItem(item)

        self.layers_widget.itemSelectionChanged.connect(self.item_clicked(vis))
        self.islayer = True

    def item_clicked(self, vis):
        def clicked_drawing():
            item_name = self.layers_widget.currentItem().text()
            # set image
            img = vis.load_img(item_name)
            self.drawing(img)

        return clicked_drawing

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
