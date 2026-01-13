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
    activation: Callable = None
    kernel_init: Callable = None
    bias_init: Callable = None

    def setup(self):
        """Set up the Q-value network structure."""
        # Calculate sizes from spaces
        self.observation_size = math.prod(self.observation_shape)

        evaluation_net_arguments = {
            "layer_dims": (
                self.observation_size,
                *self.hidden_dims,
                self.num_actions,
            ),
        }
        # Only include these if they are not None, otherwise defaults will be used
        if self.activation is not None:
            evaluation_net_arguments["activation"] = self.activation
        if self.kernel_init is not None:
            evaluation_net_arguments["weight_init"] = self.kernel_init
        if self.bias_init is not None:
            evaluation_net_arguments["bias_init"] = self.bias_init

        # Create the Q-network using the generic FC network
        self.evaluation_net = FlaxNonLinearFullyConnectedNet(**evaluation_net_arguments)

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
