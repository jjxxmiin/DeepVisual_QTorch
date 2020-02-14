import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from util import *


class CAM(object):
    def __init__(self,
                 img_path,
                 label_path,
                 model_name,):
        self.img_path = img_path
        self.save_dir = "./results/cam"
        self.label = get_label(label_path)
        self.model = get_model(model_name)
        self.model.eval()

        self.feature = []
        self.layer_names = []

    def get_feature_hook(self, module, input, output):
        self.feature.append(output.cpu().data.numpy())

    def register(self, layer_name='layer4'):
        self.layer_names.append(layer_name)
        self.model._modules.get(layer_name).register_forward_hook(self.get_feature_hook)

    def save_img(self, cls=-1):
        make_dir(self.save_dir)

        # register hook
        self.register(layer_name='layer4')

        # get softmax input weight
        params = list(self.model.parameters())
        class_weights = np.squeeze(params[-2].cpu().data.numpy())

        # get tensor image
        tensor_img = get_tensor_img(self.img_path)

        # predict
        output = self.model(tensor_img)
        h_x = F.softmax(output, dim=1).data.squeeze()
        pred = h_x.argmax(0).item()

        # get cam
        sel = cls if cls != -1 else pred
        final_conv = self.feature[0][0]

        total_cam = np.zeros(dtype=np.float32, shape=final_conv.shape[1:3])

        for i, w in enumerate(class_weights[sel]):
            total_cam += w * final_conv[i, :, :]

        scaled_cam = scaling(total_cam)
        resized_cam = cv2.resize(scaled_cam, (448, 448))
        heatmap = cv2.applyColorMap(np.uint8(255 * resized_cam), cv2.COLORMAP_JET)

        img = cv2.imread(self.img_path)
        img = cv2.resize(img, (448, 448))
        heatimg = heatmap * 0.4 + img * 0.5
        cv2.imwrite(os.path.join(self.save_dir, 'cam.jpg'), heatimg)

        label_info = "\nSELECT CLASS NUMBER  : %d " \
                     "\nSELECT CLASS NAME    : %s " \
                     "\nPREDICT CLASS NUMBER : %d " \
                     "\nPREDICT CLASS NAME   : %s"\
                     % (sel, self.label[sel], pred, self.label[pred])

        return label_info

    def load_img(self):
        cam_img = cv2.imread(os.path.join(self.save_dir, 'cam.jpg'))
        cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)

        return cam_img


class GradCAM(object):
    def __init__(self,
                 img_path,
                 label_path,
                 model_name,):
        self.img_path = img_path
        self.save_dir = "./results/grad_cam"
        self.label = get_label(label_path)
        self.model = get_model(model_name)
        self.model.eval()

        self.grads = []
        self.features = []
        self.items = []
        self.item_id = 0

    def get_feature_hook(self, name):
        def hook(module, input, output):
            self.items.append('%d_%s' % (self.item_id, name))
            self.features.append(output)
            self.item_id += 1
        return hook

    def get_gradient_hook(self, module, grad_in, grad_out):
        self.grads.append(grad_out[0])

    def register(self):
        for module, (name, _) in zip(self.model.modules(), self.model.named_modules()):
            if type(module) == nn.Conv2d or type(module) == nn.BatchNorm2d or type(module) == nn.ReLU:
                module.register_forward_hook(self.get_feature_hook(name))
                module.register_backward_hook(self.get_gradient_hook)

    def save_img(self, cls=-1):
        make_dir(self.save_dir)

        # get tensor image
        tensor_img = get_tensor_img(self.img_path)

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

        for idx, name in enumerate(self.items):
            grad = self.grads[idx][0].mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            feature = self.features[idx][0]

            grad_cam = F.relu((grad * feature).sum(dim=0)).squeeze(0)
            scaled_grad_cam = scaling(grad_cam.detach().cpu().numpy())

            resized_grad_cam = cv2.resize(scaled_grad_cam, (448, 448))
            heatmap = cv2.applyColorMap(np.uint8(255 * resized_grad_cam), cv2.COLORMAP_JET)

            img = cv2.imread(self.img_path)
            img = cv2.resize(img, (448, 448))
            heatimg = heatmap * 0.4 + img * 0.5
            cv2.imwrite(os.path.join(self.save_dir, '%s.jpg' % (name)), heatimg)

        label_info = "\nSELECT CLASS NUMBER  : %d " \
                     "\nSELECT CLASS NAME    : %s " \
                     "\nPREDICT CLASS NUMBER : %d " \
                     "\nPREDICT CLASS NAME   : %s" \
                     % (sel, self.label[sel], pred, self.label[pred])

        return label_info

    def load_img(self, item):
        grad_cam_img = cv2.imread(os.path.join(self.save_dir, '%s.jpg' % item))
        grad_cam_img = cv2.cvtColor(grad_cam_img, cv2.COLOR_BGR2RGB)

        return grad_cam_img


if __name__ == "__main__":
    grad_cam = GradCAM(img_path="../test.png",
                       label_path="../labels/imagenet_labels.pkl",
                       model_name="resnet18")

    info = grad_cam.save_img()
    print(info)