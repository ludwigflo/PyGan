from ...layers import Flatten, Reshape, Conv2d, ConvTranspose2d
from ...interfaces import GanGenerator, Gan
import torch.nn as nn
import torch


class MsgGanGenerator(GanGenerator, nn.Module):
    """
    Class for MSG-GAN generators.
    """
    def __init__(self, latent_dim: int = 30, channels: tuple = (512, 256, 128, 64, 3), paddings: tuple = (1, 1, 1, 2),
                 strides: tuple = (1, 1, 1, 1), kernel_sizes: tuple = (4, 4, 4, 6), org_size: tuple = (4, 4),
                 spec_norm: bool = True, coord_conv: bool = False):
        """
        A GAN generator model, which is based on the implementation of MSG-GAN with the DC-Gan architecture as
        backbone. The original paper of MSG-GAN can be found at https://arxiv.org/abs/1903.06048. The DC-Gan paper
        can be found at https://arxiv.org/abs/1511.06434.

        Parameters:
            latent_dim:   Dimension of the latent space, from which the samples are generated.
            channels:     Number of channels of the intermediate feature maps as well as the number of channels of the
                          output image (last tuple entry).
            org_size:     Spatial dimensions of the smallest convolutional feature map, to which the latent vector
                          should be mapped. This base size is then further upsampled using multiple transposed
                          convolutions to finally reach the output image size.
            paddings:     Number of pixels, which are used to pad the inputs of the transposed convolution layers.
            strides:      Strides of the transposed convolution layers.
            spec_norm:    Indicates, whether to use spectral normalization (https://arxiv.org/abs/1802.05957).
            coord_conv:   Indicates, whether to use CoordConv layers (https://arxiv.org/abs/1807.03247).
            kernel_sizes: Sizes of the kernels of the transposed convolution layers.
        """
        super(MsgGanGenerator, self).__init__()

        # dimensionality of the latent space, from which new samples are created
        latent_dim = latent_dim

        # convolutional layers, which are used to generate images from intermediate outputs
        self.conv1 = nn.Conv2d(channels[1], channels[4], kernel_size=1)
        self.conv2 = nn.Conv2d(channels[2], channels[4], kernel_size=1)
        self.conv3 = nn.Conv2d(channels[3], channels[4], kernel_size=1)

        # first block, which processes the latent vector and generates a 2D feature map with channels[1] channels.
        self.first_size = nn.Sequential(
            Flatten(),
            nn.Linear(latent_dim, org_size[0] * org_size[1] * channels[0]),
            nn.LeakyReLU(negative_slope=0.2),

            # reshape the vector into 2D form
            Reshape((channels[0], org_size[0], org_size[1])),

            # upsample the 2D feature map
            ConvTranspose2d(channels[0], channels[1], kernel_sizes[0], stride=strides[0], padding=paddings[0],
                            spec_norm=spec_norm, coord_conv=coord_conv),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # second block, which further upsamples the output of the first block
        self.second_size = nn.Sequential(
            ConvTranspose2d(channels[1], channels[2], kernel_sizes[1], stride=strides[1], padding=paddings[1],
                            spec_norm=spec_norm, coord_conv=coord_conv),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # third block, which further upsamples the output of the second block
        self.third_size = nn.Sequential(
            ConvTranspose2d(channels[2], channels[3], kernel_sizes[2], stride=strides[2], padding=paddings[2],
                            spec_norm=spec_norm, coord_conv=coord_conv),
            nn.BatchNorm2d(channels[3]),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # fourth block, which computes the final image
        self.fourth_size = nn.Sequential(
            ConvTranspose2d(channels[3], channels[4], kernel_sizes[3], stride=strides[3], padding=paddings[3],
                            spec_norm=spec_norm, coord_conv=coord_conv),
            nn.Tanh()
        )

    def forward(self, noise: torch.Tensor) -> list:
        """
         Forward pass of a noise tensor through the generator model in order to produces synthesize data.

         Parameters
         ----------
         noise: Noise tensor which is used by the GAN generator to synthesize data.

         Returns
         -------
         synth_data: Synthetic data samples in different resolutions.
         """

        # forward the data through the generator model and store intermediate feature maps
        x1 = self.first_size(noise)
        x2 = self.second_size(x1)
        x3 = self.third_size(x2)

        # compute the final image
        img4 = self.fourth_size(x3)

        # compute intermediate images, based on the computed intermediate feature maps
        img1 = self.conv1(x1)
        img2 = self.conv2(x2)
        img3 = self.conv3(x3)

        # return a list of all computed images
        img_list = [img1, img2, img3, img4]
        return img_list

    def generate_data_dict(self, noise: torch.Tensor, *args, **kwargs) -> dict:
        """
        Generates a dictionary of images, synthesized from the noise vector.

        Parameters
        ----------
        noise: Noise tensor which is used by the GAN generator to synthesize data.

        Returns
        -------
        image_dict: Dictionary containing the synthesized images.
        """

        # initialize the image dictionary, synthesize the images and  store their resolutions
        image_dict = dict()
        output_list = self.forward(noise)
        batch_size = output_list[0].size()[0]
        channels = output_list[0].size()[1]
        size_list_x = [x.size()[2] for x in output_list]
        size_list_y = [x.size()[3] for x in output_list]

        # store the images in the image dictionary and return the dictionary
        for i in range(batch_size):
            for j, img in enumerate(output_list):
                if channels == 1:
                    img = img[i, 0, ...]
                else:
                    img = channel_first_to_last(img)[i, ...]
                    img -= img.min()
                    img = img / img.max()
                image_dict['img_'+str(size_list_x[j])+'x'+str(size_list_y[j])+'_'+str(i)+'.png'] = img
        return image_dict


class MsgGanDiscriminator(nn.Module):

    def __init__(self, spec_norm: bool = True, coord_conv: bool = False, channels: tuple = (512, 256, 128, 64, 3),
                 org_size: tuple = (4, 4), paddings: tuple = (1, 1, 1, 2), strides: tuple = (2, 2, 2, 2),
                 kernel_sizes: tuple = (4, 4, 4, 6)):
        """
        A GAN discriminator model, which is based on the implementation of MSG-GAN with DC-Gan architecture as
        backbone. The original paper of MSG-GAN can be found at https://arxiv.org/abs/1903.06048. The DC-Gan paper
        can be found at https://arxiv.org/abs/1511.06434.

        Parameters:
            channels:     Stores the number of channels of the intermediate feature maps as well as the number of
                          channels of the output image.
            org_size:     Spatial dimensions of the smallest convolutional feature map, which is mapped into a latent
                          vector, which is then used in order to classify the input into real or fake.
            paddings:     Number of pixels, which are used to pad the inputs of the convolution layers.
            strides:      Strides of the convolution layers.
            spec_norm:    Indicates, whether to use spectral normalization (https://arxiv.org/abs/1802.05957).
            coord_conv:   Indicates, whether to use CoordConv layers (https://arxiv.org/abs/1807.03247).
            kernel_sizes: Sizes of the kernels of the convolution layers.
        """

        super(MsgGanDiscriminator, self).__init__()
        # first block, which further downsamples the output of the second block and computes predictions
        self.first_size = nn.Sequential(
            Conv2d(channels[1] + channels[-1], channels[0], kernel_sizes[0], stride=strides[0], padding=paddings[0],
                   spec_norm=spec_norm, coord_conv=coord_conv),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(negative_slope=0.2),

            Flatten(),
            nn.Linear(in_features=channels[0] * org_size[0] * org_size[1], out_features=1),
            nn.Sigmoid()
        )

        # second block, which further downsamples the output of the third block
        self.second_size = nn.Sequential(
            Conv2d(channels[2] + channels[-1], channels[1], kernel_sizes[1], stride=strides[1], padding=paddings[1],
                   spec_norm=spec_norm, coord_conv=coord_conv),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # third block, which further downsamples the output of the fourth block
        self.third_size = nn.Sequential(
            Conv2d(channels[3] + channels[-1], channels[2], kernel_sizes[2], stride=strides[2], padding=paddings[2],
                   spec_norm=spec_norm, coord_conv=coord_conv),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # fourth block, which processes an input image computes a convolutional feature map from it
        self.fourth_size = nn.Sequential(
            Conv2d(channels[4], channels[3], kernel_sizes[3], stride=strides[3], padding=paddings[3],
                   spec_norm=spec_norm, coord_conv=coord_conv),
            nn.BatchNorm2d(channels[3]),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, image_list: list) -> torch.Tensor:
        """
        Forwards the elements of an image list and decides, whether the data is real or synthesized by a GAN generator.

        Parameters
        ----------
        image_list: List of torch.Tensors, which is the base for predicting whether the data synthetic or not.

        Returns
        -------
        out: Predictions, whether the data is synthetic or not
        """

        [img1, img2, img3, img4] = image_list
        x3 = self.fourth_size(img4)
        x3 = torch.cat((x3, img3), 1)
        x2 = self.third_size(x3)
        x2 = torch.cat((x2, img2), 1)
        x1 = self.second_size(x2)
        x1 = torch.cat((x1, img1), 1)
        out = self.first_size(x1)
        return out


class MsgGan(Gan):
    def __init__(self, latent_dim=50, channels=(512, 256, 128, 64, 3), org_size=(4, 4),
                 spec_norm_gen=True, spec_norm_dis=True, coord_conv=False,
                 paddings=(1, 1, 1, 2), strides=(2, 2, 2, 2), kernel_sizes=(4, 4, 4, 6)):

        # call the constructor of the super class
        super(MsgGan, self).__init__()

        self._discriminator = None
        self._generator = None

        # create the discriminator of the MSG-GAN
        self.discriminator = MsgGanDiscriminator(coord_conv=coord_conv, channels=channels, org_size=org_size,
                                                 paddings=paddings, strides=strides, kernel_sizes=kernel_sizes,
                                                 spec_norm=spec_norm_gen)

        # create the generator of the MSG-GAN
        self.generator = MsgGanGenerator(latent_dim=latent_dim, channels=channels, paddings=paddings, strides=strides,
                                         kernel_sizes=kernel_sizes, spec_norm=spec_norm_dis, coord_conv=coord_conv,
                                         org_size=org_size)

    @property
    def generator(self) -> MsgGanGenerator:
        """
        Gan Generator.
        """
        return self._generator

    @generator.setter
    def generator(self, generator: MsgGanGenerator) -> None:
        """
        Gan Generator setter.
        """
        self._generator = generator

    @property
    def discriminator(self) -> MsgGanDiscriminator:
        """
        Gan Discriminator.
        """
        return self._discriminator

    @discriminator.setter
    def discriminator(self, discriminator: MsgGanDiscriminator) -> None:
        """
        Gan Generator setter.
        """
        self._discriminator = discriminator

    def generate_data(self, noise: torch.Tensor, *args, **kwargs)->dict:
        """
        Synthesizes new data samples based on an input noise and returns them in a consistent way with dictionaries.

        Parameters
        ----------
        noise: torch.Tensor, noise tensor which is used by the GAN generator to synthesize data.

        Returns
        -------
        data_dict: dict, containing synthetic data samples (keys are some kind of their names)

        """

        data_dict = self.generator.generate_data_dict(noise)
        return data_dict

    def discriminate_data(self, data: list, *args, **kwargs)->torch.Tensor:
        """
        Forwards data through the discriminator and decides, whether the data is real or synthetic.
        Parameters
        ----------
        data: list, list containing data (torch.Tensor, usually images) in various resolutions, which should be
              classified into real and fake data.
        Returns
        -------
        predictions: torch.Tensor, prediction whether the provided data is real or synthetic.

        """
        predictions = self.discriminator(data)
        return predictions

    def forward(self, noise: torch.Tensor, *args, **kwargs)->list:
        """
        Synthesizes new data samples based on an input noise. The data is provided in all resolutions.
        Parameters
        ----------
        noise: torch.Tensor, noise tensor which is used by the GAN generator to synthesize data.

        Returns
        -------
        data_list: list, list of synthesized data in various resolutions.

        """
        data_list = self.generator(noise)
        return data_list