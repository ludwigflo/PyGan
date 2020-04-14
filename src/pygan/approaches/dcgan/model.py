from ...model_utils.layers import GConvTranspose2d, GConv2d, Flatten, Reshape
from ...interfaces import Gan, GanGenerator
import torch.nn as nn
import torch


class DCGanGenerator(GanGenerator, nn.Module):

    """
    A GAN generator model, which is based on the implementation of DC-Gan architecture. The DC-Gan paper can be found
    at https://arxiv.org/abs/1511.06434.
    """

    def __init__(self, latent_dim=30, channels=(512, 256, 128, 64, 3), org_size=(4, 4),
                 paddings=(1, 1, 1, 1), strides=(2, 2, 2, 2), kernel_sizes=(4, 4, 4, 6),
                 spec_norm=True, coord_conv=False):

        super(DCGanGenerator, self).__init__()

        # the base generator model, which is used to synthesize data
        self.model = nn.Sequential(
            # flatten the input data in order to get the correct tensor shape for the linear module
            Flatten(),
            nn.Linear(latent_dim, org_size[0] * org_size[1] * channels[0]),
            nn.LeakyReLU(negative_slope=0.2),

            # reshape the data to 2D feature map, which can be further upsampled the final image shape
            Reshape((channels[0], org_size[0], org_size[1])),

            # upsample the feature map and apply batch norm and the non-linearity
            GConvTranspose2d(channels[0], channels[1], kernel_sizes[0], stride=strides[0], padding=paddings[0],
                             spec_norm=spec_norm, coord_conv=coord_conv),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(negative_slope=0.2),

            # upsample the feature map and apply batch norm and the non-linearity
            GConvTranspose2d(channels[1], channels[2], kernel_sizes[1], stride=strides[1], padding=paddings[1],
                             spec_norm=spec_norm, coord_conv=coord_conv),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(negative_slope=0.2),

            # upsample the feature map and apply batch norm and the non-linearity
            GConvTranspose2d(channels[2], channels[3], kernel_sizes[2], stride=strides[2], padding=paddings[2],
                             spec_norm=spec_norm, coord_conv=coord_conv),
            nn.BatchNorm2d(channels[3]),
            nn.LeakyReLU(negative_slope=0.2),

            # upsample the feature map and compute the final output image
            GConvTranspose2d(channels[3], channels[4], kernel_sizes[3], stride=strides[3], padding=paddings[3],
                             spec_norm=spec_norm, coord_conv=coord_conv),
            nn.Tanh()
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
         Forward pass of a noise tensor through the generator model in order to produces synthesize data.

         Parameters
         ----------
         noise: Noise tensor which is used by the GAN generator to synthesize data.

         Returns
         -------
         synth_data: Synthetic data samples.
         """

        synth_data = self.model(noise)
        return synth_data

    def generate_data_dict(self, noise: torch.Tensor, *args, **kwargs)->dict:
        """
        Generates a dictionary of images, synthesized from the noise vector.

        Parameters
        ----------
        noise: Noise tensor which is used by the GAN generator to synthesize data.

        Returns
        -------
        image_dict: Dictionary containing the synthesized images.
        """

        data_dict = dict()
        output_tensor = self.forward(noise)
        batch_size = output_tensor.size()[0]
        channels = output_tensor.size()[1]

        # convert the images from channels first to channels last convention
        if channels == 1:
            output_tensor = output_tensor[:, 0, ...]
        else:
            output_tensor = channel_first_to_last(output_tensor)

        # store the images in the image dictionary and return the dictionary
        for i in range(batch_size):
            img = output_tensor[i, ...]
            data_dict['img_'+str(i)+'.png'] = img
        return data_dict


class DCGanDiscriminator(nn.Module):

    """
    A GAN discriminator model, which is based on the implementation of DC-Gan architecture. The DC-Gan paper can be
    found at https://arxiv.org/abs/1511.06434.
    """

    def __init__(self, spec_norm=True, coord_conv=False, channels=(512, 256, 128, 64, 3),
                 org_size=(4, 4), paddings=(1, 1, 1, 1), strides=(2, 2, 2, 2), kernel_sizes=(4, 4, 4, 6)):
        """
        Constructor of the DCGanDiscriminator class

        Parameters
        ----------
        channels: Tuple, which stores the number of channels of the intermediate feature maps as well as the number of
                  channels of the output image.
        org_size: Spatial dimensions of the smallest convolutional feature map, to which the latent vector should be
                  mapped (tuple).
        paddings: Number of pixels, which are used to pad the input of the transposed convolution layers.
        strides:  Strides of the transposed convolution layers.
        spec_norm: Specifies whether spectral normalization (https://arxiv.org/abs/1802.05957) should be used.
        coord_conv: Specifies, whether CoordConv layers (https://arxiv.org/abs/1807.03247) should be used.
        kernel_sizes: Sizes of the kernels of the transposed convolution layers.
        """

        super(DCGanDiscriminator, self).__init__()

        self.model = nn.Sequential(

            # downsample the input image and apply batch norm and a non-linearity
            GConv2d(channels[4], channels[3], kernel_sizes[3], stride=strides[3],
                    padding=paddings[3], spec_norm=spec_norm, coord_conv=coord_conv),
            nn.BatchNorm2d(channels[3]),
            nn.LeakyReLU(negative_slope=0.2),

            # downsample the feature map and apply batch norm and a non-linearity
            GConv2d(channels[3], channels[2], kernel_sizes[2], stride=strides[2],
                    padding=paddings[2], spec_norm=spec_norm, coord_conv=coord_conv),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(negative_slope=0.2),

            # downsample the feature map and apply batch norm and a non-linearity
            GConv2d(channels[2], channels[1], kernel_sizes[1], stride=strides[1], padding=paddings[1],
                    spec_norm=spec_norm, coord_conv=coord_conv),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(negative_slope=0.2),

            # downsample the feature map and apply batch norm and a non-linearity
            GConv2d(channels[1], channels[0], kernel_sizes[0], stride=strides[0], padding=paddings[0],
                    spec_norm=spec_norm, coord_conv=coord_conv),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(negative_slope=0.2),

            # flatten the input to get the correct dimensions for the Linear layer
            Flatten(),

            # compute the predictions
            nn.Linear(in_features=channels[0] * org_size[0] * org_size[1], out_features=1),
            nn.Sigmoid()
        )

    def forward(self, data_tensor: torch.Tensor)->torch.Tensor:
        """
        Forwards the elements of an image tensor and decides, whether the data is real or synthesized.

        Parameters
        ----------
        data_tensor: Tensor, which should be classified into real and synthetic.

        Returns
        -------
        out: Predictions of the discriminator, whether the data is synthetic or not
        """

        out = self.model(data_tensor)
        return out


class DCGan(Gan):
    def __init__(self, latent_dim=30, channels=(512, 256, 128, 64, 3), org_size=(4, 4),
                 paddings=(1, 1, 1, 1), strides=(2, 2, 2, 2), kernel_sizes=(4, 4, 4, 6),
                 spec_norm_gen=True, spec_norm_dis=True, coord_conv=False):

        # call the constructor of the super class
        super(DCGan, self).__init__()

        # create the generator of the DC-GAN
        self.g = DCGanGenerator(latent_dim=latent_dim, channels=channels, org_size=org_size, kernel_sizes=kernel_sizes,
                                strides=strides,  paddings=paddings, spec_norm=spec_norm_gen, coord_conv=coord_conv)

        # create the discriminator of the DC-GAN
        self.d = DCGanDiscriminator(spec_norm=spec_norm_dis, coord_conv=coord_conv, channels=channels,
                                    org_size=org_size, paddings=paddings, strides=strides, kernel_sizes=kernel_sizes)

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

        data_dict = self.g.generate_data_dict(noise)
        return data_dict

    def discriminate_data(self, data: torch.Tensor, *args, **kwargs)->torch.Tensor:
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

        predictions = self.d(data)
        return predictions

    def forward(self, noise: torch.Tensor, *args, **kwargs)->torch.Tensor:
        """
        Synthesizes new data samples based on an input noise. The data is provided in all resolutions.

        Parameters
        ----------
        noise: torch.Tensor, noise tensor which is used by the GAN generator to synthesize data.

        Returns
        -------
        data: torch.Tensor containing synthesized data.
        """

        data = self.g(noise)
        return data
