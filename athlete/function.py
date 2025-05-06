from operator import itemgetter
from functools import reduce
from typing import Union, List, Callable, Any, Tuple, Dict

import numpy as np
import torch
from gymnasium.spaces import Space

from athlete import constants

# Specific dtype mapping for numpy to torch conversion
# This assumes that our torch implementation uses float32 for most operations
# Mapping most ints to int64 because torch can only perform indexing with int64
# Mapping int8 to int8 because we assume this is only used if you want to save memory
# By default bools are to integers, this mapping maps booleans to booleans
DTYPE_MAP = {
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


def numpy_to_tensor(np_array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Transforms a numpy array to a torch tensor following a specific dtype mapping.

    Args:
        np_array (np.ndarray): Numpy array to be transformed.
        device (str, optional): Device to which the tensor should be moved. Defaults to "cpu".

    Returns:
        torch.Tensor: Transformed torch tensor.
    """
    dtype = DTYPE_MAP.get(np_array.dtype.name, torch.float32)
    return (
        torch.from_numpy(np_array).to(device=device, dtype=dtype).requires_grad_(False)
    )


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Safely transforms a torch tensor to a numpy array. Uses default dtype mapping from numpy.

    Args:
        tensor (torch.Tensor): Torch tensor to be transformed.

    Returns:
        np.ndarray: Transformed numpy array.
    """
    return tensor.detach().cpu().numpy()


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
                lambda data: numpy_to_tensor(data, device=device),
                single_safe_itemgetter(keys)(data_batch),
            ),
        )
    )


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
