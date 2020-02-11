import cv2
from torch.nn import functional as F
import torchvision.transforms as transforms
from PIL import Image
from util import *


class CAM(object):
    def __init__(self,
                 img_path,
                 label_path,
                 model_name,):
        self.img_path = img_path
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

    def get_img(self, cls=-1):
        # register hook
        self.register(layer_name='layer4')

        # get softmax input weight
        params = list(self.model.parameters())
        class_weights = np.squeeze(params[-2].cpu().data.numpy())

        # get tensor image
        img = Image.open(self.img_path)
        cvt_tensor = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        tensor_img = cvt_tensor(img).view(1, 3, 224, 224)

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
        cv2.imwrite('./cam.jpg', heatimg)

        cam_img = cv2.imread('./cam.jpg')
        cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)

        label_info = "\nSELECT CLASS NUMBER  : %d " \
                     "\nSELECT CLASS NAME    : %s " \
                     "\nPREDICT CLASS NUMBER : %d " \
                     "\nPREDICT CLASS NAME   : %s"\
                     % (sel, self.label[sel], pred, self.label[pred])

        return cam_img, label_info