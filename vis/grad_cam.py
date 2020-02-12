import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from PIL import Image
from util import *


class GradCAM(object):
    def __init__(self,
                 img_path,
                 label_path,
                 model_name,):
        self.img_path = img_path
        self.label = get_label(label_path)
        self.model = get_model(model_name)
        self.model.eval()

        self.grads = []
        self.features = []
        self.layer_names = []

    def get_feature_hook(self, name):
        def hook(module, input, output):
            self.layer_names.append(name)
            self.features.append(output)
        return hook

    def get_gradient_hook(self, module, grad_in, grad_out):
        self.grads.append(grad_out[0])

    def register(self):
        for module, (name, _) in zip(self.model.modules(), self.model.named_modules()):
            if type(module) == nn.Conv2d or type(module) == nn.BatchNorm2d or type(module) == nn.ReLU:
                module.register_forward_hook(self.get_feature_hook(name))
                module.register_backward_hook(self.get_gradient_hook)

    def save_img(self, cls=-1):
        # get tensor image
        img = Image.open(self.img_path)
        cvt_tensor = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        tensor_img = cvt_tensor(img).view(1, 3, 224, 224)

        # register hook
        self.register()

        # predict
        output = self.model(tensor_img)
        h_x = F.softmax(output, dim=1).data.squeeze()
        pred = h_x.argmax(0).item()

        one_hot_output = torch.zeros(1, h_x.size()[0])
        one_hot_output[0][pred] = 1

        # backprop
        output.backward(gradient=one_hot_output)

        # get grad cam
        sel = cls if cls != -1 else pred

        self.grads = self.grads[::-1]

        for idx, name in enumerate(self.layer_names):
            grad = self.grads[idx][0].mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            feature = self.features[idx][0]

            grad_cam = F.relu((grad * feature).sum(dim=0)).squeeze(0)
            scaled_grad_cam = scaling(grad_cam.detach().cpu().numpy())

            resized_grad_cam = cv2.resize(scaled_grad_cam, (448, 448))
            heatmap = cv2.applyColorMap(np.uint8(255 * resized_grad_cam), cv2.COLORMAP_JET)

            img = cv2.imread(self.img_path)
            img = cv2.resize(img, (448, 448))
            heatimg = heatmap * 0.4 + img * 0.5
            cv2.imwrite('../results/grad_cam/%d_%s.jpg' % (idx,name), heatimg)

        label_info = "\nSELECT CLASS NUMBER  : %d " \
                     "\nSELECT CLASS NAME    : %s " \
                     "\nPREDICT CLASS NUMBER : %d " \
                     "\nPREDICT CLASS NAME   : %s" \
                     % (sel, self.label[sel], pred, self.label[pred])

        return label_info

    @staticmethod
    def load_img(layer):
        grad_cam_img = cv2.imread('./results/grad_cam/%s.jpg' % layer)
        grad_cam_img = cv2.cvtColor(grad_cam_img, cv2.COLOR_BGR2RGB)

        return grad_cam_img


if __name__ == "__main__":
    grad_cam = GradCAM(img_path="../test.png",
                       label_path="../labels/imagenet_labels.pkl",
                       model_name="resnet18")

    info = grad_cam.save_img()
    print(info)