import requests
import numpy as np
import pickle
import torchvision.models as models
from PyQt5.QtCore import QFile, QTextStream

def isEmpty(value):
    return True if value is "" else False


def save_url_img(url, save_path):
    img = requests.get(url)
    open(save_path, 'wb').write(img.content)


def get_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        AssertionError("[ERROR] This model is not supported")

    return model


def get_label(label_path):
    with open(label_path, "rb") as f:
        cls_name = pickle.load(f)

    return cls_name


def scaling(img):
    img = img - np.min(img)
    img = img / np.max(img)

    return img


def use_theme(app, path):
    file = QFile(path)
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())