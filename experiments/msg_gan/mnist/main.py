from pygan.approaches import msg_gan
from data_loader import data_loader
from torch.nn import BCELoss
import torch

# hyper parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parameter_file = 'parameters.yaml'
batch_size = 256
noise_size = 30

# create data loaders for train and test data
print('Load data and create data loaders...')
data_path = '/media/data2/data/MNIST/processed/'
train_data_name = 'training.pt'
test_data_name = 'test.pt'
train_loader = data_loader(data_path, train_data_name, batch_size, rand = True, noise_size=noise_size)
test_loader = data_loader(data_path, test_data_name, batch_size=1, rand=False, noise_size=noise_size, fix_noise=True)
print('Done!\n')

# define ethe model parameters and initialize the model
print('Initialize the model')
channels=(256, 128, 64, 32, 1)
kernel_sizes=(4, 4, 4, 6)
latent_dim=noise_size
paddings=(1, 1, 2, 2)
strides=(2, 2, 2, 2)
spec_norm_dis=True
spec_norm_gen=True
coord_conv=False
org_size=(2, 2)
model = msg_gan.GanModule(channels=channels, kernel_sizes=kernel_sizes, paddings=paddings, strides=strides,
                          spec_norm_dis=spec_norm_dis, spec_norm_gen=spec_norm_gen, coord_conv=coord_conv,
                          org_size=org_size, latent_dim=latent_dim)
print('Done!\n')
#
# # initialize the trainer
# trainer = msg_gan.Trainer(model, train_loader, BCELoss, test_loader, parameter_file, device)
# trainer.train()
