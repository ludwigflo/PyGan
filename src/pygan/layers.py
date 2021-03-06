from torch.nn.utils import spectral_norm
import torch.nn as nn
import torch


class Flatten(nn.Module):
    """
    Flattens an input value into size (batch_size, -1)
    """

    @staticmethod
    def forward(x: torch.Tensor):
        """
        Flattens an input value into size (batch_size, -1)

        Parameters
        ----------
            x: Input tensor, which should be reshaped.

        Returns
        -------
            x: Reshaped tensor.
        """

        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    """
    Reshapes an input value into batch_size, shape.
    """
    def __init__(self, shape: tuple=(1, 28, 28)):
        """
        Initializes the class by storing parameter, which defines the target shape.

        Parameters
        ----------
            shape: target shape (without batch_size).
        """

        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor):
        """
        Reshapes an input value to the desired target shape.

        Parameters
        ----------
            x: Input tensor, which should be reshaped.

        Returns
        -------
            x: Reshaped tensor.
        """

        return x.view(x.size(0), self.shape[0], self.shape[1], self.shape[2])


class Conv2d(nn.Module):
    """
    Wrapper for the torch.nn.Conv2d class. It augments the class by properties like spectral normalization or coord-conv
    approaches.
    """

    def __init__(self, in_channels, out_channels, size, stride, padding,
                 spec_norm=False, coord_conv=False):
        """
        Initializes the Conv2d layer.

        Parameters
        ----------
            in_channels: Number of input channels for the convolutional layer.
            out_channels: Number of output channels for the convolutional layer.
            size: Kernel size of the convolutional layer.
            stride: Stride, which is used by the convolutional layer for sliding over the input feature map.
            padding: Padding, which is applied to the input feature map.
            spec_norm: Specifies whether to use spectral normalization (often used in GAN layers) or not.
            coord_conv: Specifies, whether to apply the coord_conv approach.
        """

        super(Conv2d, self).__init__()
        self.coord_conv = coord_conv

        # generate the torch conv2d layer with corresponding properties
        if coord_conv:
            in_channels += 2
        if not spec_norm:
            self.conv = nn.Conv2d(in_channels, out_channels, size, stride=stride, padding=padding)
        else:
            self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, size, stride=stride, padding=padding))

    def forward(self, x):
        """
        Applies the conv2d operation with all defined properties.

        Parameters
        ----------
            x: Input feature map.

        Returns
        -------
            out: Output, which was generated by convolving the input with the convolutional layer.
        """

        # prepare the input for the conv layer, if the coord convolution should be applied
        if self.coord_conv:
            batch, _, height, width = x.size()
            coord_info = torch.zeros(batch, 2, height, width).float()

            for i in range(height):
                coord_info[:, 0, i, :] = i
            for i in range(width):
                coord_info[:, 1, :, i] = i
            coord_info[:, 0, ...] /= height
            coord_info[:, 1, ...] /= width
            if x.is_cuda:
                coord_info = coord_info.cuda()
            x = torch.cat((x, coord_info), 1)

        # perform the convolution
        out = self.conv(x)
        return out


class ConvTranspose2d(nn.Module):
    """
    Wrapper for the torch.nn.ConvTranspose2d class. It augments the class by properties like spectral normalization or
    coord-conv approaches.
    """

    def __init__(self, in_channels, out_channels, size, stride, padding,
                 spec_norm=False, coord_conv=False):
        """
        Initializes the ConvTranspose2d layer.

        Parameters
        ----------
            in_channels: Number of input channels for the convolutional layer.
            out_channels: Number of output channels for the convolutional layer.
            size: Kernel size of the convolutional layer.
            stride: Stride, which is used by the convolutional layer for sliding over the input feature map.
            padding: Padding, which is applied to the input feature map.
            spec_norm: Specifies whether to use spectral normalization (often used in GAN layers) or not.
            coord_conv: Specifies, whether to apply the coord_conv approach.
        """

        super(ConvTranspose2d, self).__init__()
        self.coord_conv = coord_conv

        if coord_conv:
            in_channels += 2
        if not spec_norm:
            self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, size, stride=stride, padding=padding)
        else:
            self.conv_trans = spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, size,
                                                               stride=stride, padding=padding))

    def forward(self, x):
        """
        Applies the ConvTranspose2d operation with all defined properties.

        Parameters
        ----------
            x: Input feature map.

        Returns
        -------
            out: Output, which was generated by convolving the input with the convolutional layer.
        """
        if self.coord_conv:
            batch, _, height, width = x.size()
            coord_info = torch.zeros(batch, 2, height, width).float()

            for i in range(height):
                coord_info[:, 0, i, :] = i
            for i in range(width):
                coord_info[:, 1, :, i] = i
            coord_info[:, 0, ...] /= height
            coord_info[:, 1, ...] /= width
            if x.is_cuda:
                coord_info = coord_info.cuda()
            x = torch.cat((x, coord_info), 1)
        out = self.conv_trans(x)
        return out
