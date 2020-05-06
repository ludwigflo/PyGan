from ml_utils.data_utils.data_loader import ImgLoader
from ml_utils.utils import read_parameter_file
import matplotlib.pyplot as plt
import unittest


class TestImgLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        parameter_file = 'dummy_params.yaml'
        cls.params = read_parameter_file(parameter_file)

    def test_constructor(self):
        img_loader = ImgLoader(self.params, img_size=(3, 128, 128), shuffle_data=True, return_labels=False)

        if self.params['data']['base_path'] == 'images/base_path2/':
            self.assertEqual(len(img_loader.split[0]), 2)
            self.assertEqual(len(img_loader.split[1]), 2)
            if len(img_loader.split) == 3:
                self.assertEqual(len(img_loader.split[2]), 2)

    def test_train_generator(self):

        # test the ImgLoader for returning data without labels
        img_loader = ImgLoader(self.params, img_size=(3, 128, 128), shuffle_data=True, return_labels=False)
        train_loader = img_loader.train_generator(2)

        for i, img in enumerate(train_loader):
            self.assertEqual(img.shape, (2, 3, 128, 128))
            break

        # test the ImgLoader for returning data with additional labels
        img_loader = ImgLoader(self.params, img_size=(3, 128, 128), shuffle_data=False, return_labels=True)
        train_loader = img_loader.train_generator(7)
        for i, (img, label) in enumerate(train_loader):
            self.assertEqual(img.shape, (7, 3, 128, 128))
            break

    def test_val_generator(self):
        img_loader = ImgLoader(self.params, img_size=(3, 128, 128), shuffle_data=True, return_labels=False)
        val_loader = img_loader.val_generator(1)
        for i, (img, done) in enumerate(val_loader):
            self.assertEqual(img.shape, (1, 3, 128, 128))
            print(img.shape, done)
            if done:
                break

    def test_test_generator(self):
        img_loader = ImgLoader(self.params, img_size=(3, 128, 128), shuffle_data=True, return_labels=False)
        test_loader = img_loader.test_generator(1)
        for i, (img, done) in enumerate(test_loader):
            self.assertEqual(img.shape, (1, 3, 128, 128))
            if done:
                break

if __name__ == '__main__':
    unittest.main()