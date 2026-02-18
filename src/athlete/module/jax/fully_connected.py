from typing import Tuple, Callable, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn


class FlaxNonLinearFullyConnectedNet(nn.Module):
    """A flexible fully connected neural network in Flax.

    This module creates a multi-layer perceptron with configurable layer dimensions,
    activations, and optional layer normalization.

    Attributes:
        layer_dims: Tuple of layer dimensions (input_dim, hidden1, hidden2, ..., output_dim).
        activation: Activation function applied after each hidden layer. Default: nn.relu.
        final_activation: Optional activation applied to the output layer. Default: None.
        initial_activation: Optional activation applied to the input. Default: None.
        weight_init: Kernel initializer for Dense layers. Default: lecun_uniform.
        bias_init: Bias initializer for Dense layers. Default: uniform with scale=1/sqrt(fan_in).
        use_layer_norm: Whether to apply LayerNorm after each hidden layer. Default: False.
        layer_norm_epsilon: Epsilon for LayerNorm numerical stability. Default: 1e-6.

    Example:
        >>> net = FlaxNonLinearFullyConnectedNet(
        ...     layer_dims=(128, 256, 256, 64),
        ...     activation=nn.relu,
        ...     use_layer_norm=True
        ... )
        >>> params = net.init(key, jnp.ones((1, 128)))
        >>> output = net.apply(params, x)

    Layer ordering for hidden layers:
        Dense → LayerNorm (if enabled) → Activation

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
    use_layer_norm: bool = False
    layer_norm_epsilon: float = 1e-6

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

        # Create LayerNorm layers for hidden layers only (not output layer)
        if self.use_layer_norm:
            self.layer_norms = [
                nn.LayerNorm(epsilon=self.layer_norm_epsilon, name=f"layer_norm_{i+1}")
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

        # Apply layers with optional layer normalization
        num_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            # Apply Dense layer
            x = layer(x)

            # For hidden layers: apply LayerNorm (if enabled) then activation
            if i < num_layers - 1:
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
                x = self.activation(x)

        # Apply final activation if provided
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x
