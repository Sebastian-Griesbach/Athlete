from typing import Tuple, Callable
import math

import jax.numpy as jnp
import flax.linen as nn

from athlete.module.jax.fully_connected import FlaxNonLinearFullyConnectedNet


class FlaxFCDiscreteQValueFunction(nn.Module):
    """A discrete Q-value function implemented in Flax.
    Takes in observations and returns Q-values for discrete actions.
    """

    # Configuration attributes
    observation_shape: Tuple[int, ...]
    num_actions: int
    hidden_dims: Tuple[int, ...]
    activation: Callable = nn.relu
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    def setup(self):
        """Set up the Q-value network structure."""
        # Calculate sizes from spaces
        self.observation_size = math.prod(self.observation_shape)

        # Create the Q-network using the generic FC network
        self.evaluation_net = FlaxNonLinearFullyConnectedNet(
            layer_dims=(
                self.observation_size,
                *self.hidden_dims,
                self.num_actions,
            ),
            activation=self.activation,
            weight_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass to compute Q-values for observations.

        Args:
            x: Observation batch of shape (batch_size, *observation_shape)
            training: Whether we're in training mode

        Returns:
            Q-values of shape (batch_size, num_actions)
        """
        # Reshape observations to flat vectors if needed
        x = x.reshape(-1, self.observation_size)

        # Pass through the network
        return self.evaluation_net(x)
