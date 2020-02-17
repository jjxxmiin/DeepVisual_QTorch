import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


class Vanilla(object):
    def __init__(self,
                 img_path,
                 label_path,
                 model_name,):
        self.img_path = img_path
        self.save_dir = "./results/vanilla"
        self.label = get_label(label_path)
        self.model = get_model(model_name)
        self.model.eval()

        self.grad = None

    def get_grad_hook(self, module, grad_in, grad_out):
        self.grad = grad_in[0]

    def register(self):
        self.first_layer = list(self.model._modules.items())[0][1]
        self.first_layer.register_backward_hook(self.get_grad_hook)

    def save_img(self, cls=-1):
        make_dir(self.save_dir)

        # register hook
        self.register()

        # get tensor image
        tensor_img = get_tensor_img(self.img_path)

        # requires input gradient
        tensor_img = Variable(tensor_img, requires_grad=True)

        # predict
        output = self.model(tensor_img)
        h_x = F.softmax(output, dim=1).data.squeeze()
        pred = h_x.argmax(0).item()

        # get vanilla grad
        sel = cls if cls != -1 else pred

        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
        one_hot_output[0][sel] = 1

        output.backward(gradient=one_hot_output)

        grad = self.grad.cpu().data.numpy()[0]

        grad_img = scaling(grad).transpose(1, 2, 0)
        grad_img = (grad_img * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(self.save_dir, 'vanilla.jpg'), grad_img)

        label_info = "\nSELECT CLASS NUMBER  : %d " \
                     "\nSELECT CLASS NAME    : %s " \
                     "\nPREDICT CLASS NUMBER : %d " \
                     "\nPREDICT CLASS NAME   : %s"\
                     % (sel, self.label[sel], pred, self.label[pred])

        return label_info

    def load_img(self):
        vanilla_img = cv2.imread(os.path.join(self.save_dir, 'vanilla.jpg'))
        vanilla_img = cv2.cvtColor(vanilla_img, cv2.COLOR_BGR2RGB)

        return vanilla_img


class Smooth(object):
    def __init__(self,
                 img_path,
                 label_path,
                 model_name,):
        self.img_path = img_path
        self.save_dir = "./results/smooth"
        self.label = get_label(label_path)
        self.model = get_model(model_name)
        self.model.eval()

        self.grad = None

    def get_grad_hook(self, module, grad_in, grad_out):
        self.grad = grad_in[0]

    def register(self):
        self.first_layer = list(self.model._modules.items())[0][1]
        self.first_layer.register_backward_hook(self.get_grad_hook)

    def save_img(self, prog, cls=-1):
        make_dir(self.save_dir)

        # register hook
        self.register()

        # get tensor image
        tensor_img = get_tensor_img(self.img_path)

        # requires input gradient
        tensor_img = Variable(tensor_img, requires_grad=True)

        # predict
        output = self.model(tensor_img)
        h_x = F.softmax(output, dim=1).data.squeeze()
        pred = h_x.argmax(0).item()

        # get vanilla grad
        sel = cls if cls != -1 else pred

        smooth_grad = np.zeros(tensor_img.size()[1:])

        param_n = 50
        param_sigma_multiplier = 5

        mean = 0
        sigma = param_sigma_multiplier / (torch.max(tensor_img) - torch.min(tensor_img)).item()

        for x in range(param_n):
            # progressBar
            prog.setValue(100 / 50 * (x + 1))
            # Generate noise
            noise = Variable(tensor_img.data.new(tensor_img.size()).normal_(mean, sigma ** 2))
            # Add noise to the image
            noisy_img = tensor_img+noise
            # Calculate gradients
            output = self.model(noisy_img)

            one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
            one_hot_output[0][sel] = 1

            self.grad = None

            output.backward(gradient=one_hot_output)

            grad_img = self.grad[0].cpu().data.numpy()[0]

            # Add gradients to smooth_grad
            smooth_grad += grad_img ** 2

        smooth_grad = smooth_grad / param_n

        smooth_grad_img = scaling(smooth_grad).transpose(1, 2, 0)
        smooth_grad_img = (smooth_grad_img * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(self.save_dir, 'smooth.jpg'), smooth_grad_img)

        label_info = "\nSELECT CLASS NUMBER  : %d " \
                     "\nSELECT CLASS NAME    : %s " \
                     "\nPREDICT CLASS NUMBER : %d " \
                     "\nPREDICT CLASS NAME   : %s"\
                     % (sel, self.label[sel], pred, self.label[pred])

        return label_info

    def load_img(self):
        smooth_img = cv2.imread(os.path.join(self.save_dir, 'smooth.jpg'))
        smooth_img = cv2.cvtColor(smooth_img, cv2.COLOR_BGR2RGB)

        return smooth_img


class Guided_Backprop(object):
    def __init__(self,
                 img_path,
                 label_path,
                 model_name,):
        self.img_path = img_path
        self.save_dir = "./results/guided_backprop"
        self.label = get_label(label_path)
        self.model = get_model(model_name)
        self.model.eval()

        self.grad = None
        self.forward_relu = []
        self.backward_relu = []
        self.items = ['color', 'grad', 'pos', 'neg']

    def get_gradient_hook(self, module, grad_input, grad_output):
        self.grad = grad_input[0]

    def get_forward_relu_hook(self, module, input, output):
        self.forward_relu.append(output)

    def get_backward_relu_hook(self, module, grad_in, grad_out):
        cor = self.forward_relu[-1]
        cor[cor > 0] = 1
        modified_grad_out = cor * torch.clamp(grad_in[0], min=0.0)
        del self.forward_relu[-1]  # Remove last forward output
        return (modified_grad_out,)

    def register(self):
        self.first_layer = list(self.model._modules.items())[0][1]
        self.first_layer.register_backward_hook(self.get_gradient_hook)

        for module, (name, _) in zip(self.model.modules(), self.model.named_modules()):
            if type(module) == nn.ReLU:
                module.register_forward_hook(self.get_forward_relu_hook)
                module.register_backward_hook(self.get_backward_relu_hook)

    def save_img(self, cls=-1):
        make_dir(self.save_dir)

        # register hook
        self.register()

        # get tensor image
        tensor_img = get_tensor_img(self.img_path)

        # requires input gradient
        tensor_img = Variable(tensor_img, requires_grad=True)

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
        grad = self.grad.cpu().data.numpy()[0]

        # get color img
        color_img = scaling(grad.copy()).transpose(1, 2, 0)
        color_img = (color_img * 255.0).astype('uint8')

        # get gray img
        gray_img = get_gray_img(grad.copy())
        gray_img = scaling(np.repeat(gray_img, 3, axis=0)).transpose(1, 2, 0)# channel 1 -> 3
        gray_img = (gray_img * 255.0).astype('uint8')

        # get pos grad
        pos_saliency = (np.maximum(0, grad) / grad.max())
        pos_saliency -= pos_saliency.min()
        pos_saliency /= pos_saliency.max()
        pos_saliency = pos_saliency.transpose(1, 2, 0)
        pos_saliency = (pos_saliency * 255.0).astype('uint8')

        # get neg grad
        neg_saliency = (np.maximum(0, -grad) / -grad.min())
        neg_saliency -= neg_saliency.min()
        neg_saliency /= neg_saliency.max()
        neg_saliency = neg_saliency.transpose(1, 2, 0)
        neg_saliency = (neg_saliency * 255.0).astype('uint8')

        cv2.imwrite(os.path.join(self.save_dir, '%s.jpg' % (self.items[0])), color_img)
        cv2.imwrite(os.path.join(self.save_dir, '%s.jpg' % (self.items[1])), gray_img)
        cv2.imwrite(os.path.join(self.save_dir, '%s.jpg' % (self.items[2])), pos_saliency)
        cv2.imwrite(os.path.join(self.save_dir, '%s.jpg' % (self.items[3])), neg_saliency)

        label_info = "\nSELECT CLASS NUMBER  : %d " \
                     "\nSELECT CLASS NAME    : %s " \
                     "\nPREDICT CLASS NUMBER : %d " \
                     "\nPREDICT CLASS NAME   : %s" \
                     % (sel, self.label[sel], pred, self.label[pred])

        return label_info

    def load_img(self, item):
        guided_grad_img = cv2.imread(os.path.join(self.save_dir, '%s.jpg' % item))
        guided_grad_img = cv2.cvtColor(guided_grad_img, cv2.COLOR_BGR2RGB)

        return guided_grad_img


if __name__ == "__main__":
    gg = Smooth(img_path="../test.png",
                label_path="../labels/imagenet_labels.pkl",
                model_name="resnet18")

    info = gg.save_img()
    print(info)
    a = gg.load_img()
