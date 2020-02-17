import sys
import logging
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic, QtCore
from vis.cam import CAM, GradCAM
from vis.grad import Vanilla, Smooth, Guided_Backprop
from util import use_theme, get_label


class Class_Form(QDialog, QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        label_path = "./labels/imagenet_labels.pkl"
        
        self.labels = get_label(label_path)
        self.ui = uic.loadUi("./ui/Class_View.ui", self)
        self.initUI()

    def initUI(self):
        # 검색 버튼 이벤트 핸들링
        self.ui.setWindowTitle('Class')
        self.ui.show()

        self.search_line.setPlaceholderText("Input Class Name or Number")
        self.search_btn.clicked.connect(self.search)

        self.content_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.content_table.cellDoubleClicked.connect(self.click_table)
        self.set_table()

    def search(self):
        text = self.search_line.text()
        find_items = self.content_table.findItems(text, QtCore.Qt.MatchContains)

        items = [item.row() for item in find_items]

        for idx in range(0, len(self.labels)):
            if idx in items:
                self.content_table.setRowHidden(idx, False)
            else:
                self.content_table.setRowHidden(idx, True)

        self.search_line.setText("")

    def click_table(self):
        cur_item = self.content_table.currentItem()
        print(self.content_table.item(cur_item.row(), 0).text())

    def set_table(self):
        self.content_table.setRowCount(len(self.labels))
        self.content_table.setColumnCount(1)

        for idx, name in self.labels.items():
            currentRowCount = self.content_table.rowCount()
            self.content_table.setItem(idx, 0, QTableWidgetItem(name))


if __name__ == '__main__':
    app = QApplication(sys.argv)

    use_theme(app, "theme/darkgray.qss")

    w = Class_Form()
    sys.exit(app.exec())
