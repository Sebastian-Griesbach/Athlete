from operator import itemgetter
from functools import reduce, partial
from typing import Union, List, Callable, Any, Tuple, Dict

import numpy as np
import torch
from gymnasium.spaces import Space
import jax
import jax.numpy as jnp

from athlete import constants

# Specific dtype mapping for numpy to torch conversion
# This assumes that our torch implementation uses float32 for most operations
# Mapping most ints to int64 because torch can only perform indexing with int64
# Mapping int8 to int8 because we assume this is only used if you want to save memory
# By default bools are to integers, this mapping maps booleans to booleans
TORCH_DTYPE_MAP = {
    "float16": torch.float32,
    "float32": torch.float32,
    "float64": torch.float32,
    "int8": torch.int64,
    "int16": torch.int64,
    "int32": torch.int64,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

JAX_DTYPE_MAP = {
    "float16": jnp.float32,
    "float32": jnp.float32,
    "float64": jnp.float32,
    "int8": jnp.int32,
    "int16": jnp.int32,
    "int32": jnp.int32,
    "int64": jnp.int32,
    "uint8": jnp.uint8,
    "bool": jnp.bool_,
}


def numpy_to_torch_tensor(np_array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Transforms a numpy array to a torch tensor following a specific dtype mapping.

    Args:
        np_array (np.ndarray): Numpy array to be transformed.
        device (str, optional): Device to which the tensor should be moved. Defaults to "cpu".

    Returns:
        torch.Tensor: Transformed torch tensor.
    """
    dtype = TORCH_DTYPE_MAP.get(np_array.dtype.name, torch.float32)
    return (
        torch.from_numpy(np_array)
        .to(device=device, dtype=dtype, non_blocking=True)
        .requires_grad_(False)
    )


def torch_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Safely transforms a torch tensor to a numpy array. Uses default dtype mapping from numpy.

    Args:
        tensor (torch.Tensor): Torch tensor to be transformed.

    Returns:
        np.ndarray: Transformed numpy array.
    """
    return tensor.detach().cpu().numpy()


def numpy_to_jax_array(np_array: np.ndarray) -> jnp.ndarray:
    """Transforms a numpy array to a jax array following a specific dtype mapping.

    Args:
        np_array (np.ndarray): Numpy array to be transformed.

    Returns:
        jnp.ndarray: Transformed jax array.
    """
    dtype = JAX_DTYPE_MAP.get(np_array.dtype.name, jnp.float32)
    return jnp.array(np_array, dtype=dtype)


def jax_array_to_numpy(array: jnp.ndarray) -> np.ndarray:
    """Safely transforms a jax array to a numpy array. Uses default dtype mapping from numpy.

    Args:
        array (jnp.ndarray): Jax array to be transformed.

    Returns:
        np.ndarray: Transformed numpy array.
    """
    return np.array(jax.device_get(array))


def gymnasium_value_to_batched_numpy_array(value: Union[int, np.ndarray]) -> np.ndarray:
    """Takes a value that might be a numpy array or an int and and returns a numpy array with a batch dimension.
    Also copies the data to ensure that the original data is not modified by following operations.

    Args:
        value (Union[int, np.ndarray]): Value to be transformed. Can be a numpy array or an int.

    Returns:
        np.ndarray: Transformed numpy array with a batch dimension.
    """
    if isinstance(value, np.ndarray):
        value = value.copy()
    else:
        value = np.array([value])

    value = np.expand_dims(value, axis=0)

    return value


def single_safe_itemgetter(keys: List[str]) -> Callable[[Any], Tuple[Any]]:
    """
    An Itemgetter that always returns a tuple even if only one key is provided.

    Args:
        keys (list): List of keys.

    Returns:
        callable: A Itemgetter function that always returns a tuple.
    """
    if len(keys) > 1:
        return itemgetter(*keys)
    else:
        return lambda _dict: (itemgetter(*keys)(_dict),)


def chain_functions(
    function_list: List[Callable], input_value: Any
) -> Callable[[Any], Any]:
    """Returns a function that takes an input value and applies a list of functions to it in order.

    Args:
        function_list (_type_): List of functions to be applied to the input value.
        input_value (_type_): Input value to be passed through the functions in succession.

    Returns:
        Callable[[Any], Any]: A function that takes an input value and applies the list of functions to it in order.
    """
    return (
        reduce(
            lambda intermediate_value, function: function(intermediate_value),
            function_list,
            input_value,
        )
        if function_list
        else input_value
    )


def extract_data_from_batch(
    data_batch: Dict[str, np.ndarray], keys: List[str], device: str
) -> Dict[str, torch.Tensor]:
    """Extracts data from a batch and converts it to torch tensors.

    Args:
        data_batch (Dict[str, np.ndarray]): Batch of data to be extracted.
        keys (List[str]): List of keys to extract from the batch.
        device (str): Device to which the tensors should be moved.

    Returns:
        Dict[str, torch.Tensor]: Extracted data as torch tensors.
    """
    return dict(
        zip(
            keys,
            map(
                lambda data: numpy_to_torch_tensor(data, device=device),
                single_safe_itemgetter(keys)(data_batch),
            ),
        )
    )


def jax_extract_data_from_batch(
    data_batch: Dict[str, np.ndarray], keys: List[str]
) -> Dict[str, np.ndarray]:
    """Extracts data from a batch.

    Args:
        data_batch (Dict[str, np.ndarray]): Batch of data to be extracted.
        keys (List[str]): List of keys to extract from the batch.

    Returns:
        Dict[str, torch.Tensor]: Extracted data.
    """
    # Seems to be faster if we just do the conversion implicitly, might cause trouble with specific dtypes
    # return dict(
    #     zip(
    #         keys,
    #         map(
    #             lambda data: numpy_to_jax_array(data),
    #             single_safe_itemgetter(keys)(data_batch),
    #         ),
    #     )
    # )
    return dict(zip(keys, single_safe_itemgetter(keys)(data_batch)))


def create_transition_data_info(
    observation_space: Space, action_space: Space
) -> Dict[str, Dict[str, Any]]:
    """Creates a dictionary with data information of a transition used for the replay buffer according to the observation and action space.

    Args:
        observation_space (Space): Observation space of the data.
        action_space (Space): Action space of the data.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with data information of a transition.
    """

    observation_info = {
        "shape": observation_space.shape,
        "dtype": str(observation_space.dtype),
    }
    if observation_info["shape"] == ():
        observation_info["shape"] = (1,)

    action_info = {"shape": action_space.shape, "dtype": str(action_space.dtype)}
    if action_info["shape"] == ():
        action_info["shape"] = (1,)

    return {
        constants.DATA_REWARDS: {"shape": (1,), "dtype": np.float32},
        constants.DATA_OBSERVATIONS: observation_info,
        constants.DATA_NEXT_OBSERVATIONS: observation_info,
        constants.DATA_ACTIONS: action_info,
        constants.DATA_TERMINATEDS: {"shape": (1,), "dtype": np.bool_},
    }


def create_jnp_transition_data_info(
    observation_space: Space, action_space: Space
) -> Dict[str, Dict[str, Any]]:
    """Creates a dictionary with data information of a transition used for the replay buffer according to the observation and action space.

    Args:
        observation_space (Space): Observation space of the data.
        action_space (Space): Action space of the data.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with data information of a transition.
    """

    observation_info = {
        "shape": observation_space.shape,
        "dtype": str(jnp.dtype(observation_space.dtype)),
    }
    if observation_info["shape"] == ():
        observation_info["shape"] = (1,)

    action_info = {
        "shape": action_space.shape,
        "dtype": str(jnp.dtype(action_space.dtype)),
    }
    if action_info["shape"] == ():
        action_info["shape"] = (1,)

    return {
        constants.DATA_REWARDS: {"shape": (1,), "dtype": jnp.float32},
        constants.DATA_OBSERVATIONS: observation_info,
        constants.DATA_NEXT_OBSERVATIONS: observation_info,
        constants.DATA_ACTIONS: action_info,
        constants.DATA_TERMINATEDS: {"shape": (1,), "dtype": jnp.bool_},
    }


@jax.jit
def jax_mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((predictions - targets) ** 2)


@jax.jit
def jax_huber_loss(
    predictions: jnp.ndarray, targets: jnp.ndarray, delta: float = 1.0
) -> jnp.ndarray:
    error = predictions - targets
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return jnp.mean(0.5 * quadratic**2 + delta * linear)


@jax.jit
def berhu_loss(  # Reversed Huber
    predictions: jnp.ndarray, targets: jnp.ndarray, delta: float = 1.0
) -> jnp.ndarray:
    error = predictions - targets
    abs_error = jnp.abs(error)

    # Linear for small errors, quadratic for large errors (C1 at |error| = delta)
    linear_part = abs_error
    quadratic_part = (abs_error**2 + delta**2) / (2.0 * delta)

    return jnp.mean(jnp.where(abs_error <= delta, linear_part, quadratic_part))


@partial(jax.jit, static_argnames=("exponent",))
def generalized_berhu_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    delta: float = 1.0,
    exponent: float = 4.0,
) -> jnp.ndarray:
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if exponent <= 1.0:
        raise ValueError("exponent must be > 1 for a smooth polynomial tail")

    error = predictions - targets
    abs_error = jnp.abs(error)

    linear_part = abs_error
    poly_part = (abs_error**exponent) / (
        exponent * (delta ** (exponent - 1.0))
    ) + delta * (1.0 - 1.0 / exponent)

    return jnp.mean(jnp.where(abs_error <= delta, linear_part, poly_part))


@jax.jit
def l1_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(predictions - targets))
