import torch.nn as nn
from Conv_Layer import ConvLayer
from Primary_Caps import PrimaryCaps
from Digit_Caps import DigitCaps
from Decoder import Decoder

class CapsuleNetwork(nn.Module):

    def __init__(self):
        '''Constructs a complete Capsule Network.'''
        super(CapsuleNetwork, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.decoder = Decoder()

    def forward(self, images):
        '''Defines the feedforward behavior.
           param images: the original MNIST image input data
           return: output of DigitCaps layer, reconstructed images, class scores
           '''
        primary_caps_output = self.primary_capsules(self.conv_layer(images))
        caps_output = self.digit_capsules(primary_caps_output).squeeze().transpose(0, 1)
        reconstructions, y = self.decoder(caps_output)
        return caps_output, reconstructions, y