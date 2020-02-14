import os
import requests
import numpy as np
import pickle
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from PyQt5.QtCore import QFile, QTextStream


def isEmpty(value):
    return True if value is "" else False


def make_dir(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)


def get_tensor_img(path):
    img = Image.open(path)
    cvt_tensor = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    tensor_img = cvt_tensor(img).view(1, 3, 224, 224)

    return tensor_img

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


def get_gray_img(grad):
    gray_img = np.sum(np.abs(grad), axis=0)

    max_value = np.percentile(gray_img, 99)
    min_value = np.min(gray_img)

    gray_img = (np.clip((gray_img - min_value) / (max_value - min_value), 0, 1))
    gray_img = np.expand_dims(gray_img, axis=0)

    return gray_img


def use_theme(app, path):
    file = QFile(path)
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())