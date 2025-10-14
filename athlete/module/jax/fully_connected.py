from typing import Tuple, Callable, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn


def make_bias_init(fan_in):
    def bias_init(key, shape, dtype=jnp.float32):
        limit = 1.0 / jnp.sqrt(fan_in)
        return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)

    return bias_init


class FlaxNonLinearFullyConnectedNet(nn.Module):
    """A flexible fully connected neural network in Flax."""

    # Configuration attributes
    layer_dims: Tuple[int, ...]
    activation: Callable = nn.relu
    final_activation: Optional[Callable] = None
    initial_activation: Optional[Callable] = None
    # Same default initialization as in Pytorch
    weight_init: Callable = nn.initializers.lecun_uniform()
    bias_init: Callable = None

    def setup(self):
        """Set up the layers of the network."""
        # Number of Linear layers in the network
        num_linear_layers = len(self.layer_dims) - 1

        # List will be converted into a tuple by Flax
        self.layers = [
            nn.Dense(
                features=self.layer_dims[i + 1],
                kernel_init=self.weight_init,
                bias_init=(
                    make_bias_init(fan_in=self.layer_dims[i])
                    if self.bias_init is None
                    else self.bias_init
                ),
                name=f"dense_{i+1}",
            )
            for i in range(num_linear_layers)
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network.

        Args:
            x: Input tensor to the network
            training: Whether we're in training mode (for dropout, etc.)

        Returns:
            Output tensor after passing through the network
        """
        # Apply initial activation if provided
        if self.initial_activation is not None:
            x = self.initial_activation(x)

        # Apply layers and activations
        num_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Add activation except after the final layer
            if i < num_layers - 1:
                x = self.activation(x)

        # Apply final activation if provided
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x
