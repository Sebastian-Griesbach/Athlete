from typing import Tuple, Callable, Optional, Dict
from dataclasses import field

import jax
import jax.numpy as jnp
import flax.linen as nn


class FlaxNonLinearFullyConnectedNet(nn.Module):
    """A flexible fully connected neural network in Flax.

    This module creates a multi-layer perceptron with configurable layer dimensions,
    activations, and optional pre-activation normalization.

    Attributes:
        layer_dims: Tuple of layer dimensions (input_dim, hidden1, hidden2, ..., output_dim).
        activation: Activation function applied after each hidden layer. Default: nn.relu.
        final_activation: Optional activation applied to the output layer. Default: None.
        initial_activation: Optional activation applied to the input. Default: None.
        weight_init: Kernel initializer for Dense layers. Default: lecun_uniform.
        bias_init: Bias initializer for Dense layers. Default: uniform with scale=1/sqrt(fan_in).
        pre_activation_module: Optional normalization module (e.g., nn.LayerNorm, RMSNorm)
            applied after Dense and before activation in hidden layers. Default: None.
        pre_activation_module_kwargs: Dictionary of keyword arguments for pre_activation_module. Default: None.

    Example:
        >>> # With LayerNorm
        >>> net = FlaxNonLinearFullyConnectedNet(
        ...     layer_dims=(128, 256, 256, 64),
        ...     activation=nn.relu,
        ...     pre_activation_module=nn.LayerNorm,
        ...     pre_activation_module_kwargs={"epsilon": 1e-6}
        ... )
        >>> # With RMSNorm
        >>> net = FlaxNonLinearFullyConnectedNet(
        ...     layer_dims=(128, 256, 256, 64),
        ...     activation=nn.relu,
        ...     pre_activation_module=RMSNorm,
        ...     pre_activation_module_kwargs={"epsilon": 1e-6, "use_scale": True}
        ... )
        >>> params = net.init(key, jnp.ones((1, 128)))
        >>> output = net.apply(params, x)

    Layer ordering for hidden layers:
        Dense → pre_activation_module (if provided) → Activation

    Output layer:
        Dense → final_activation (if provided)
    """

    # Configuration attributes
    layer_dims: Tuple[int, ...]
    activation: Callable = nn.relu
    final_activation: Optional[Callable] = None
    initial_activation: Optional[Callable] = None
    weight_init: Callable = None
    bias_init: Callable = None
    pre_activation_module: Optional[Callable] = None
    pre_activation_module_kwargs: Dict = field(default_factory=dict)

    def setup(self):
        """Set up the layers of the network."""
        # Number of Linear layers in the network
        num_linear_layers = len(self.layer_dims) - 1

        # Create Dense layers - list will be converted into a tuple by Flax
        self.layers = [
            nn.Dense(
                features=self.layer_dims[i + 1],
                kernel_init=(
                    nn.initializers.lecun_uniform()
                    if self.weight_init is None
                    else self.weight_init
                ),
                bias_init=(
                    nn.initializers.uniform(scale=1 / jnp.sqrt(self.layer_dims[i]))
                    if self.bias_init is None
                    else self.bias_init
                ),
                name=f"dense_{i+1}",
            )
            for i in range(num_linear_layers)
        ]

        # Create pre-activation normalization layers for hidden layers only (not output layer)
        if self.pre_activation_module is not None:

            self.pre_activation_layers = [
                self.pre_activation_module(
                    name=f"pre_activation_{i+1}", **self.pre_activation_module_kwargs
                )
                for i in range(num_linear_layers - 1)
            ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network.

        Args:
            x: Input tensor to the network of shape (batch_size, input_dim)

        Returns:
            Output tensor after passing through the network
        """
        # Apply initial activation if provided
        if self.initial_activation is not None:
            x = self.initial_activation(x)

        # Apply layers with optional pre-activation normalization
        num_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            # Apply Dense layer
            x = layer(x)

            # For hidden layers: apply pre-activation module (if provided) then activation
            if i < num_layers - 1:
                if self.pre_activation_module is not None:
                    x = self.pre_activation_layers[i](x)
                x = self.activation(x)

        # Apply final activation if provided
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x
