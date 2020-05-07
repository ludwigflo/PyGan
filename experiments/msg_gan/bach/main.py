from pygan.ml_utils.src.ml_utils.data_utils.data_loader import ImgLoader
from pygan.utils import read_parameter_file
from pygan.approaches import msg_gan
from data_loader import data_loader
from torch.nn import BCELoss
import torch
import os


# hyper parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parameter_file = 'parameters.yaml'
batch_size = 64
noise_size = 30

# create data loaders for train and test data
print('Load data and create data loaders...')
parameters = read_parameter_file(parameter_file)
data_generator = ImgLoader(parameters, img_size=(3, 64, 64))
print('Done!\n')

# define ethe model parameters and initialize the model
# print('Initialize the model')
# channels=(256, 128, 64, 32, 1)
# kernel_sizes=(4, 4, 4, 6)
# latent_dim=noise_size
# paddings=(1, 1, 2, 2)
# strides=(2, 2, 2, 2)
# spec_norm_dis=True
# spec_norm_gen=True
# coord_conv=False
# org_size=(2, 2)
#
# model = msg_gan.GanModule(channels=channels, kernel_sizes=kernel_sizes, paddings=paddings, strides=strides,
#                           spec_norm_dis=spec_norm_dis, spec_norm_gen=spec_norm_gen, coord_conv=coord_conv,
#                           org_size=org_size, latent_dim=latent_dim)
# print('Done!\n')
#
# # initialize the trainer
# trainer = msg_gan.Trainer(model, train_loader, BCELoss, test_loader, parameter_file, device)
# trainer.train()
