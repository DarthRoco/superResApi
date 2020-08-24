import io
import os
import logging
import torch
import numpy as np

from PIL import Image
from torch.autograd import Variable
from torchvision import transforms


logger = logging.getLogger(__name__)


class MNISTDigitClassifier(object):
    """
    MNISTDigitClassifier handler class. This handler takes a greyscale image
    and returns the digit in that image.
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""

        properties = ctx.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "DF2K.pth")
        # Read model definition file
        model_def_path = os.path.join(model_dir, "model.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")

        from model import RRDBNet as Net
        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = Net(in_nc=3, out_nc=3,nf=64, nb=23)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug('Model file {0} loaded successfully'.format(model_pt_path))
        self.initialized = True

    def preprocess(self, data):
        """
         Scales, crops, and normalizes a PIL image for a MNIST model,
         returns an Numpy array
        """
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")


        image = Image.open(io.BytesIO(image))
        img = image.astype(np.float32) / 255.
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # some images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        H, W, C = img.shape
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        return img

    def inference(self, img):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # Convert 2D image to 1D vector
        img = img.unsqueeze(0)
       #img = torch.from_numpy(img)

        self.model.eval()
        inputs = Variable(img).to(self.device)
        outputs = self.model.forward(inputs)


        return outputs

    def get_current_visuals(self,ip, op):
            out_dict = OrderedDict()
            out_dict['LQ'] = ip.detach()[0].float().cpu()
            out_dict['SR'] = op.detach()[0].float().cpu()
            return out_dict

    def tensor2img(self,tensor, out_type=np.uint8, min_max=(0, 1)):
        '''
        Converts a torch Tensor into an image Numpy array
        Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
        Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
        '''
        tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
        tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
        n_dim = tensor.dim()
        if n_dim == 4:
            n_img = len(tensor)
            img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        elif n_dim == 3:
            img_np = tensor.numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        elif n_dim == 2:
            img_np = tensor.numpy()
        else:
            raise TypeError(
                'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
            # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
        return img_np.astype(out_type)


_service = MNISTDigitClassifier()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data1 = _service.preprocess(data)
    data1 = _service.inference(data1)
    data1 = _service.tensor2img(data1)

    return data1