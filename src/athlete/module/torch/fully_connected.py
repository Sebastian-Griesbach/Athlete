import torch
import torch.nn as nn
from typing import List, Callable, Optional, Union, Sequence


class NonLinearFullyConnectedNet(nn.Module):
    """A class to generate a fully connected neural network with options for activation functions and layer initialization."""

    LAYER_PREFIX = "fully_connected"
    ACTIVATION_PREFIX = "activation"

    def __init__(
        self,
        layer_dims: List[int],
        activation: nn.Module = nn.ReLU(),
        final_activation: Optional[nn.Module] = None,
        initial_activation: Optional[nn.Module] = None,
        layer_initialization_functions: Optional[
            Union[
                Callable[[nn.Module], None],
                Sequence[Optional[Callable[[nn.Module], None]]],
            ]
        ] = None,
    ):
        """Initializes a fully connected neural network.

        Args:
            layer_dims (List[int]): List of integers representing the dimensions of each layer. This includes the input and output layers.
            activation (nn.Module, optional): Activation function to be used between layers. Defaults to nn.ReLU().
            final_activation (Optional[nn.Module], optional): Activation function to be used after the final layer. Defaults to None.
            initial_activation (Optional[nn.Module], optional): Activation function to be used before the first layer. Defaults to None.
            layer_initialization_functions (Optional[ Union[ Callable[[nn.Module], None], Sequence[Optional[Callable[[nn.Module], None]]], ] ], optional):
                A function or a sequence of functions to initialize the layers. If a single function is provided, it will be applied to all layers.
                If a sequence is provided, it should have the same length as the number of layers. Defaults to None

        Raises:
            ValueError: If the length of layer_initialization_functions does not match the number of layers.
        """
        super(NonLinearFullyConnectedNet, self).__init__()
        self.activation = activation
        self.layers = torch.nn.Sequential()

        # Number of Linear layers in the network
        num_linear_layers = len(layer_dims) - 1

        # Normalize the initialization functions to a list
        init_functions: List[Optional[Callable[[nn.Module], None]]] = []

        if layer_initialization_functions is not None:
            if callable(layer_initialization_functions):
                # Single function - repeat for all layers
                init_functions = [layer_initialization_functions] * num_linear_layers
            else:
                # Sequence of functions - check length
                if len(layer_initialization_functions) != num_linear_layers:
                    raise ValueError(
                        f"Expected {num_linear_layers} initialization functions, got {len(layer_initialization_functions)}"
                    )
                init_functions = list(layer_initialization_functions)
        else:
            # No initialization functions
            init_functions = [None] * num_linear_layers

        # Add initial activation if provided
        if initial_activation is not None:
            self.layers.add_module("initial_activation", initial_activation)

        # Create the network layers
        for i in range(num_linear_layers):
            # Create linear layer
            layer_name = f"{self.LAYER_PREFIX}_{i+1}"
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])

            # Apply initialization if provided
            if init_functions[i] is not None:
                init_functions[i](linear_layer)

            # Add the layer
            self.layers.add_module(layer_name, linear_layer)

            # Add activation except after the final layer
            if i < num_linear_layers - 1:
                self.layers.add_module(
                    f"{self.ACTIVATION_PREFIX}_{i+1}", self.activation
                )

        # Add final activation if provided
        if final_activation is not None:
            self.layers.add_module("final_activation", final_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        return self.layers(x)
