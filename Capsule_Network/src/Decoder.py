import torch
import torch.nn as nn
import torch.nn.functional as F

# check if CUDA is available
TRAIN_ON_GPU = torch.cuda.is_available()

num_classes = 40

class Decoder(nn.Module):
    def __init__(self, input_vector_length=16, input_capsules=num_classes, hidden_dim=512):
        '''Constructs an series of linear layers + activations.
           param input_vector_length: dimension of input capsule vector, default value = 16
           param input_capsules: number of capsules in previous layer, default value = 10
           param hidden_dim: dimensions of hidden layers, default value = 512
           '''
        super(Decoder, self).__init__()

        # calculate input_dim
        input_dim = input_vector_length * input_capsules

        # define linear layers + activations
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # first hidden layer
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),  # second, twice as deep
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, 75 * 75),  # can be reshaped into 28*28 image
            nn.Sigmoid()  # sigmoid activation to get output pixel values in a range from 0-1
        )

    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input; vectors from the previous DigitCaps layer
           return: two things, reconstructed images and the class scores, y
           '''
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        # find the capsule with the maximum vector length
        # here, vector length indicates the probability of a class' existence
        _, max_length_indices = classes.max(dim=1)

        # create a sparse class matrix
        sparse_matrix = torch.eye(40)  # 40 is the number of classes
        if TRAIN_ON_GPU:
            sparse_matrix = sparse_matrix.cuda()
        # get the class scores from the "correct" capsule
        y = sparse_matrix.index_select(dim=0, index=max_length_indices.data)

        # create reconstructed pixels
        x = x * y[:, :, None]
        # flatten image into a vector shape (batch_size, vector_dim)
        flattened_x = x.contiguous().view(x.size(0), -1)
        # create reconstructed image vectors
        reconstructions = self.linear_layers(flattened_x)

        # return reconstructions and the class scores, y
        return reconstructions, y