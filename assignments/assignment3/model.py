import numpy as np

from layers import (
    FullyConnectedLayer,
    ReLULayer,
    ConvolutionalLayer,
    MaxPoolingLayer,
    Flattener,
    softmax_with_cross_entropy,
    l2_regularization,
)


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.layers = [
            ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            Flattener(),
            FullyConnectedLayer(4 * conv2_channels, n_output_classes),
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        params = self.params()
        for key in params.keys():
            params[key].grad = np.zeros_like(params[key].value)

        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment

        layer_result = X
        for i in range(len(self.layers)):
            layer_result = self.layers[i].forward(layer_result)

        loss, grad = softmax_with_cross_entropy(layer_result, y)

        dX = grad
        for i in reversed(range(len(self.layers))):
            dX = self.layers[i].backward(dX)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        layer_result = X
        for i in range(len(self.layers)):
            layer_result = self.layers[i].forward(layer_result)

        return np.argmax(layer_result, axis=1)

    def params(self):
        result = {}

        result["C1W"] = self.layers[0].W
        result["C1B"] = self.layers[0].B
        result["C2W"] = self.layers[3].W
        result["C2B"] = self.layers[3].B
        result["FCW"] = self.layers[7].W
        result["FCB"] = self.layers[7].B

        return result
