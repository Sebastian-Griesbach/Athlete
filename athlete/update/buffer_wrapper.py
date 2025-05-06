from typing import Dict, Any, List, Optional
import numpy as np
from operator import itemgetter

from athlete.update.buffer import Buffer
from athlete.saving.file_handler import FileHandler
from athlete.function import chain_functions
from athlete.saving.saveable_component import SaveContext


class BufferWrapper(Buffer):
    """Base class for buffer wrappers. Useful if data should be manipulated in some way when being and or sampled from the buffer.
    Simply mimics the behavior of the wrapped buffer if no function is overwritten."""

    def __init__(self, replay_buffer: Buffer) -> None:
        """Initializes the BufferWrapper with a replay buffer.

        Args:
            replay_buffer (Buffer): The replay buffer to be wrapped.
        """
        self.replay_buffer = replay_buffer

    def add(
        self, transition_dictionary: Dict[str, np.ndarray], episode_ended: np.ndarray
    ) -> None:
        self.replay_buffer.add(
            data_dictionary=transition_dictionary, episode_ended=episode_ended
        )

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        return self.replay_buffer.sample(batch_size=batch_size)

    def encode_sample(self, sample_ids: List[int]) -> Dict[str, np.ndarray]:
        return self.replay_buffer.encode_sample(sample_ids)

    def save_checkpoint(self, context: SaveContext):
        self.replay_buffer.save_checkpoint(
            context=context,
        )

    def load_checkpoint(self, file_handler: FileHandler):
        self.replay_buffer.load_checkpoint(
            file_handler=file_handler,
        )

    @property
    def size(self):
        return self.replay_buffer.size

    @property
    def max_size(self):
        return self.replay_buffer.max_size

    @property
    def data_info(self) -> Dict[str, Dict[str, Any]]:
        return self.replay_buffer.data_info

    @property
    def pointer_position(self):
        return self.replay_buffer.pointer_position


class InputOutputWrapper(BufferWrapper):
    """Baseclass for specific wrapper type that does some modification during adding and sampling.
    This is useful for things like normalization, where you want to update statistics on adding,
    and then normalize data upon sampling.
    """

    def __init__(
        self,
        replay_buffer: Buffer,
        data_info: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(replay_buffer=replay_buffer)

        if data_info is None:
            self._data_info = replay_buffer.data_info
        else:
            self._data_info = data_info

    def add(
        self, transition_dictionary: Dict[str, np.ndarray], episode_ended: np.ndarray
    ) -> None:
        transformed = self.in_transform(data_dictionary=transition_dictionary)
        self.replay_buffer.add(data_dictionary=transformed, episode_ended=episode_ended)
        self.post_add_routine()

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        raw_sample = self.replay_buffer.sample(batch_size=batch_size)
        transformed = self.out_transform(data_dictionary=raw_sample)
        return transformed

    def in_transform(self, data_dictionary: Dict[str, Any]) -> Dict[str, Any]:
        """Overwrite this function to transform the data before adding it to the buffer.

        Args:
            data_dictionary (Dict[str, Any]): The data to be transformed.

        Returns:
            Dict[str, Any]: The transformed data.
        """
        return data_dictionary

    def post_add_routine(self):
        """This function is called after adding data to the buffer."""
        pass

    def out_transform(self, data_dictionary: Dict[str, Any]) -> Dict[str, Any]:
        """Overwrite this function to transform the data after sampling it from the buffer.

        Args:
            data_dictionary (Dict[str, Any]): The sampled data to be transformed.

        Returns:
            Dict[str, Any]: Transformed sampled data.
        """
        return data_dictionary


class PostBufferPreprocessingWrapper(InputOutputWrapper):
    """A wrapper that applies preprocessing functions to the data after sampling from the buffer.
    This is useful if the data is stored in a different format than it is used for training.
    E.g. using int8 values to save memory of pixel inputs and then converting them to float32 for training.
    """

    def __init__(
        self,
        replay_buffer: Buffer,
        post_replay_buffer_data_preprocessing: Dict[str, List[callable]],
    ) -> None:
        """Initializes the PostBufferPreprocessingWrapper with a replay buffer and a dictionary of preprocessing functions.
        The keys of the dictionary are the names of the fields in the replay buffer, and the values are lists of functions to be applied to the corresponding field.

        Args:
            replay_buffer (Buffer): The replay buffer to be wrapped.
            post_replay_buffer_data_preprocessing (Dict[str, List[callable]]): Dictionary of preprocessing functions to be applied to the data after sampling from the buffer.
                The keys of the dictionary are the names of the fields in the replay buffer, and the values are lists of functions to be applied to the corresponding field.
        """
        super().__init__(replay_buffer=replay_buffer)
        self.post_replay_buffer_preprocessing = post_replay_buffer_data_preprocessing
        self.field_getter = itemgetter(
            *list(post_replay_buffer_data_preprocessing.keys())
        )

    def out_transform(self, data_dictionary: Dict[str, Any]) -> Dict[str, Any]:
        """Applies the preprocessing functions to the data after sampling from the buffer.

        Args:
            data_dictionary (Dict[str, Any]): The sampled data to be transformed.

        Returns:
            Dict[str, Any]: Transformed sampled data.
        """

        preprocessed_values = map(
            chain_functions,
            self.post_replay_buffer_preprocessing.values(),
            self.field_getter(data_dictionary),
        )

        preprocessed_dictionary = dict(
            zip(self.post_replay_buffer_preprocessing.keys(), preprocessed_values)
        )

        data_dictionary.update(preprocessed_dictionary)

        return data_dictionary
