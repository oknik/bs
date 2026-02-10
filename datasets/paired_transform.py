import random
from torchvision import transforms

class PairedTransform:
    def __init__(self):
        self.resize = transforms.Resize((256,256))

        self.crop = transforms.RandomResizedCrop(224, scale=(0.8,1.0))
        self.hflip = transforms.RandomHorizontalFlip(p=0.5)

        self.color = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        self.gray = transforms.RandomGrayscale(p=0.2)

        self.rotate = transforms.RandomRotation(15)
        self.affine = transforms.RandomAffine(15)

        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )

    def _sync(self, transform, imgC, imgG):
        seed = random.randint(0, 99999)
        random.seed(seed)
        imgC = transform(imgC)
        random.seed(seed)
        imgG = transform(imgG)
        return imgC, imgG

    def __call__(self, imgC, imgG):
        imgC = self.resize(imgC)
        imgG = self.resize(imgG)

        imgC, imgG = self._sync(self.crop, imgC, imgG)
        imgC, imgG = self._sync(self.hflip, imgC, imgG)
        imgC, imgG = self._sync(self.color, imgC, imgG)
        imgC, imgG = self._sync(self.gray, imgC, imgG)
        imgC, imgG = self._sync(self.rotate, imgC, imgG)
        imgC, imgG = self._sync(self.affine, imgC, imgG)

        imgC = self.norm(self.to_tensor(imgC))
        imgG = self.norm(self.to_tensor(imgG))

        return imgC, imgG
