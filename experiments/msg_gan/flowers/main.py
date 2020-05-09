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
batch_size = 256

# initialize the model
print('Initialize Model...')
channels = (512, 256, 128, 64, 3)
kernel_sizes = (4, 4, 4, 6)
strides = (2, 2, 2, 2)
paddings=(1, 1, 1, 2)
spec_norm_gen = True
spec_norm_dis = True
org_size = (8, 8)
latent_dim = 100
model = msg_gan.GanModule(latent_dim = latent_dim, channels = channels, kernel_sizes = kernel_sizes, strides = strides,
                          paddings = paddings, org_size = org_size, spec_norm_gen = spec_norm_gen,
                          spec_norm_dis = spec_norm_dis)
print('Done!\n')

# create data loaders for train and test data
print('Load data and create data loaders...')
parameters = read_parameter_file(parameter_file)
data_generator = ImgLoader(parameters, img_size=(3, 128, 128))
train_loader = data_loader(data_generator, batch_size=1, noise_size=latent_dim, size_tuple=(16, 32, 64), train_data=True)
validation_loader = data_loader(data_generator, batch_size=1, noise_size=latent_dim,
                                size_tuple=(16, 32, 64), train_data=False)
print('Done!\n')

# initialize the trainer
trainer = msg_gan.Trainer(model, train_loader, BCELoss, validation_loader, parameter_file, device)
trainer.train()
