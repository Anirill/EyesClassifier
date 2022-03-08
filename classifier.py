import torch
import cv2
from torchvision import transforms

MODEL_PATH = "./enet_b3_0021.pth"
USE_CUDA = False


class OpenEyesClassificator:
    def __init__(self, model_path=MODEL_PATH, use_cuda=USE_CUDA):
        """
        :param model_path: path to model
        :param use_cuda: set True if needs to use CUDA
        """
        self.use_cuda = use_cuda
        self.location = 'cuda' if self.use_cuda else 'cpu'
        self.model = torch.load(model_path, map_location=torch.device(self.location))

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.sigmoid = torch.nn.Sigmoid()

    def predict(self, inpIm):
        """
        :param inpIm: full path to image
        :return: is_open_score: float is 1 for open eye, 0 for closed
        """
        self.model.eval()
        image = cv2.imread(inpIm)
        image = self.transforms(image)
        image = image.unsqueeze(0)

        if self.use_cuda:
            self.model.cuda()
            image = image.cuda()
        with torch.no_grad():
            pred = self.model(image)
            pred = self.sigmoid(pred)
            is_open_score = torch.softmax(pred, dim=1).detach().cpu().squeeze().numpy()[1]
        return is_open_score
