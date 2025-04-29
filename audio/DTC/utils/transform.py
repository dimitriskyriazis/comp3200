from torchvision import transforms
import random
from PIL import Image

class RandomTranslateWithReflect:
    """ Translate image randomly
    """
    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, img):
        if self.max_translation == 0:
            return img
        x_translation, y_translation = random.randint(-self.max_translation, self.max_translation), random.randint(-self.max_translation, self.max_translation)
        x_pad, y_pad = abs(x_translation), abs(y_translation)

        # reflection padding
        img = transforms.functional.pad(img, (x_pad, y_pad), padding_mode='reflect')
        img = transforms.functional.affine(img, angle=0, translate=(x_translation, y_translation), scale=1.0, shear=0)

        return img

class TransformTwice:
    """ Take two random crops of one image as the query and key. """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)
