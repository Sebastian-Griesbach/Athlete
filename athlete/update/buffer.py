from typing import Dict, List, Any, Optional
import numpy as np
from abc import ABC, abstractmethod
import os
import sys
from collections import deque
from copy import deepcopy
import math
import gc

import cpprb

from athlete import constants
from athlete.saving.saveable_component import SaveContext
from athlete.global_objects import RNGHandler


class Buffer(ABC):
    """Base class for replay buffers commonly used in off-policy reinforcement learning algorithms.

    This abstract class defines the interface that all buffer implementations should follow,
    providing standard methods for adding transitions and sampling batches of experiences.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def add(
        self,
        data_dictionary: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, any]] = None,
    ) -> None:
        """Add data to the replay buffer.

        Args:
            data_dictionary (Dict[str, np.ndarray]): Data to be added to the replay buffer.
            metadata (Optional[Dict[str, any]], optional): Metadata that somehow affect the buffer,
            e.g. end episode signal used for memory compression for observation and next observation. Defaults to None.
        """
        ...

    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample data from the replay buffer.

        Args:
            batch_size (int): Number of entries to return.
        """
        ...

    @abstractmethod
    def encode_sample(self, sample_ids: List[int]) -> Dict[str, np.ndarray]:
        """Get specific data from the replay buffer by id.

        Args:
            sample_ids (List[int]): List of indices to get from the replay buffer.

        Returns:
            Dict[str, np.ndarray]: Retrieved data.
        """
        ...

    @property
    def size(self) -> int:
        """Current size of the replay buffer."""
        ...

    @property
    def max_size(self) -> int:
        """Maximum capacity of the replay buffer."""
        ...

    @property
    def data_info(self) -> Dict[str, Dict[str, Any]]:
        """Information describing the content of the replay buffer, (field names, shapes, dtypes)."""
        ...

    @property
    def pointer_position(self) -> int:
        """Current position of the pointer in the replay buffer (position where the next data will be added)."""
        ...

    def save_checkpoint(self, context: SaveContext):
        pass

    def load_checkpoint(self, context: SaveContext):
        pass


class EpisodicCPPReplayBuffer(Buffer):
    """Replay buffer using the cpprb library. This replay buffer can handle large amounts of data and use various data compression methods.
    It provides saving functionalities that allow for saving the replay buffer in chunks if it is too large to be saved in one go.
    """

    # based on https://ymd_h.gitlab.io/cpprb/
    # 4 GB in bytes
    REPLAY_SPLIT_SIZE = 4 * 1024 * 1024 * 1024

    FILE_REPLAY_BUFFER_HANDLING = "replay_buffer_handling_stats"
    FILE_REPLAY_BUFFER = "cpp_replay_buffer"
    FILE_EMPTY_SIGNAL = "empty"

    # Almost all of the complexity in this class stems from handling saves such that:
    # 1. The insertion point ends up in the correct place
    # 2. The episode borders are correctly saved and loaded which is relevant for compressing data in memory
    # 3. The replay buffer is saved in chunks if it is to large to be saved in one go

    def __init__(
        self,
        capacity: int,
        replay_buffer_info: Dict[str, Dict[str, Any]],
        additional_arguments: Dict[str, Any] = {},
    ) -> None:
        """Initialize the replay buffer.

        Args:
            capacity (int): Maximum capacity of the replay buffer.
            replay_buffer_info (Dict): Information describing the content of the replay buffer, (field names, shapes, dtypes). See cpprb documentation for more details. https://ymd_h.gitlab.io/cpprb/
            additional_arguments (Dict[str, Any], optional): Additional arguments passed to the cpprb initialization. Can be sued for specific memory compressions. Defaults to {}.
        """
        super().__init__()
        initialization_replay_buffer_info = (
            self._create_initialization_replay_buffer_info(
                replay_buffer_info=replay_buffer_info,
                additional_arguments=additional_arguments,
            )
        )
        self.replay_buffer = cpprb.ReplayBuffer(
            capacity, initialization_replay_buffer_info, **additional_arguments
        )
        self._max_size = capacity
        self.last_added_end_position = 0
        self.replay_buffer_info = replay_buffer_info
        # Useful to determine batch_size
        self.some_data_key = list(self.replay_buffer_info.keys())[0]

        self.buffer_episode_ended = np.full(
            self._max_size, dtype=np.bool_, fill_value=False
        )

        self.random_numbers_generator = RNGHandler.get_random_number_generator()

    def add(
        self,
        data_dictionary: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        episode_ended = metadata.get(constants.METADATA_EPISODE_ENDED, None)
        if episode_ended is None:
            raise ValueError(
                "Episode ended metadata is required to add data to episodic replay buffer."
            )

        self._add_episodic_data(
            data_dictionary=data_dictionary, episode_ended=episode_ended
        )

        added_end_position = self.replay_buffer.get_next_index()

        # keep track of episode borders, needed for loading
        if added_end_position > self.last_added_end_position:
            self.buffer_episode_ended[
                self.last_added_end_position : added_end_position
            ] = episode_ended
        else:
            split = self._max_size - self.last_added_end_position
            self.buffer_episode_ended[self.last_added_end_position :] = episode_ended[
                :split
            ]
            self.buffer_episode_ended[:added_end_position] = episode_ended[split:]

        self.last_added_end_position = added_end_position

    def sample(self, batch_size: int):

        stored_size = self.size
        if stored_size < batch_size:
            # with replacement
            sample_ids = self.random_numbers_generator.choice(
                stored_size, size=batch_size, replace=True
            )
        else:
            # without replacement
            sample_ids = self.random_numbers_generator.choice(
                stored_size, size=batch_size, replace=False
            )

        return self.encode_sample(sample_ids)

    def encode_sample(self, sample_ids: List[int]) -> Dict[str, np.ndarray]:
        return self.replay_buffer._encode_sample(sample_ids)

    def save_checkpoint(self, context: SaveContext):
        if self.size == 0:
            # replay buffer is empty
            context.file_handler.save_to_file(
                to_save=self.FILE_EMPTY_SIGNAL,
                save_path=os.path.join(
                    context.save_path, context.prefix + self.FILE_REPLAY_BUFFER_HANDLING
                ),
            )
            return
        # Save replay buffer info

        replay_memory = self._calculate_memory_size()
        number_of_splits = math.ceil(replay_memory / self.REPLAY_SPLIT_SIZE)

        replay_buffer_size = self.replay_buffer.get_stored_size()
        split_size = math.ceil(replay_buffer_size / number_of_splits)
        rb_full = self._max_size <= self.replay_buffer.get_stored_size()

        handling_stats = (
            self.last_added_end_position,
            rb_full,
            self.buffer_episode_ended,
            number_of_splits,
            split_size,
        )
        context.file_handler.save_to_file(
            to_save=handling_stats,
            save_path=os.path.join(
                context.save_path,
                context.prefix + self.FILE_REPLAY_BUFFER_HANDLING,
            ),
        )

        # If replay buffer is to large to be saved in one go.
        if number_of_splits > 1:
            self._save_split_replay_buffer(
                file_handler=context.file_handler,
                save_path=context.save_path,
                prefix=context.prefix,
                number_of_splits=number_of_splits,
                split_size=split_size,
            )
        else:
            context.file_handler.save_to_file(
                to_save=self.replay_buffer.get_all_transitions(),
                save_path=os.path.join(
                    context.save_path, context.prefix + self.FILE_REPLAY_BUFFER
                ),
                enable_cache=False,
            )

    def load_checkpoint(self, context: SaveContext):

        loaded_handling_data = context.file_handler.load_from_file(
            load_path=os.path.join(
                context.save_path, context.prefix + self.FILE_REPLAY_BUFFER_HANDLING
            )
        )
        if loaded_handling_data == self.FILE_EMPTY_SIGNAL:
            # replay buffer is empty
            self.replay_buffer.clear()
            self.last_added_end_position = 0
            return
        # Load replay buffer info

        (
            self.last_added_end_position,
            rb_full,
            self.buffer_episode_ended,
            number_of_splits,
            split_size,
        ) = loaded_handling_data

        self.replay_buffer.clear()
        # add dummy data to set pointer to correct position
        if rb_full and not self.last_added_end_position == 0:
            self._add_dummy_data()

        # If replay buffer is to large to be loaded in one chunk
        if number_of_splits > 1:
            self._load_split_replay_buffer(
                file_handler=context.file_handler,
                load_path=context.save_path,
                prefix=context.prefix,
                number_of_splits=number_of_splits,
                split_size=split_size,
                rb_full=rb_full,
                environment_saved=context.metadata.get(
                    constants.SAVE_ARGUMENT_SAVE_ENVIRONMENT_STATE, False
                ),
            )
        else:
            self._load_cpprb(
                file_handler=context.file_handler,
                load_path=context.save_path,
                prefix=context.prefix,
                rb_full=rb_full,
                replay_buffer_info=self.replay_buffer_info,
                environment_saved=context.metadata.get(
                    constants.SAVE_ARGUMENT_SAVE_ENVIRONMENT_STATE, False
                ),
            )

    @property
    def size(self):
        return self.replay_buffer.get_stored_size()

    @property
    def max_size(self):
        return self._max_size

    @property
    def data_info(self):
        return self.replay_buffer_info

    @property
    def pointer_position(self):
        return self.last_added_end_position

    def _save_split_replay_buffer(
        self, file_handler, save_path, prefix, number_of_splits, split_size
    ):
        replay_buffer_size = self.replay_buffer.get_stored_size()
        for split in range(number_of_splits):
            split_start = split_size * split
            split_end = min(split_size * (split + 1), replay_buffer_size)

            split_ids = list(range(split_start, split_end))
            split_transition = self.replay_buffer._encode_sample(split_ids)
            file_path = os.path.join(
                save_path,
                prefix + self.FILE_REPLAY_BUFFER + "_split_" + str(split),
            )
            file_handler.save_to_file(
                to_save=split_transition, save_path=file_path, enable_cache=False
            )

            # Make sure the split is deleted from memory before loading the next one
            del split_transition
            gc.collect()

    def _add_non_episodic_data(self, data_dictionary: Dict[str, np.ndarray]) -> None:
        self.replay_buffer.add(**data_dictionary)

    def _add_episodic_data(
        self, data_dictionary: Dict[str, np.ndarray], episode_ended: np.ndarray
    ) -> None:
        batch_size = len(data_dictionary[self.some_data_key])

        episode_start_ids = np.where(episode_ended)[0] + 1
        transition_data_array_list = list(data_dictionary.values())

        # Create dictionary per episode
        episodic_data_list = map(
            lambda transition_data_array: np.split(
                transition_data_array, episode_start_ids, axis=0
            ),
            transition_data_array_list,
        )
        episodic_data_dictionaries = list(
            map(
                lambda episodic_data: dict(zip(data_dictionary.keys(), episodic_data)),
                zip(*episodic_data_list),
            )
        )

        # Add data of ending episodes
        for i in range(len(episode_start_ids)):
            self.replay_buffer.add(**episodic_data_dictionaries[i])
            self.replay_buffer.on_episode_end()

        # Add remaining data without ending episode
        if episode_start_ids.size < 1 or episode_start_ids[-1] < batch_size:
            self.replay_buffer.add(**episodic_data_dictionaries[-1])

    def _add_dummy_data(self):
        dummy_transitions = {}
        for key in self.replay_buffer_info.keys():
            dummy_transitions[key] = np.zeros(
                (self.last_added_end_position, *self.replay_buffer_info[key]["shape"])
            )

        self.replay_buffer.add(**dummy_transitions)
        self.replay_buffer.on_episode_end()

        del dummy_transitions
        gc.collect()

    def _load_split_replay_buffer(
        self,
        file_handler,
        load_path,
        prefix,
        number_of_splits,
        split_size,
        rb_full,
        environment_saved=False,
    ):
        # We dont need to shift the data if the replay buffer is not full,
        # or when it is full but the pointer is at the beginning of the replay buffer
        shift_needed = rb_full and self.last_added_end_position != 0

        if shift_needed:
            shift_split = self.last_added_end_position // split_size
            # shift split ids according to pointer
            splits = deque(range(number_of_splits))
            splits.rotate(-shift_split)
            splits = list(splits)
            shifted_buffer_episode_ended = np.concatenate(
                [
                    self.buffer_episode_ended[self.last_added_end_position :],
                    self.buffer_episode_ended[: self.last_added_end_position],
                ]
            )
            added_so_far = 0

            # load first Split in which the data needs to be shifted according to pointer
            split_transitions = file_handler.load_from_file(
                load_path=os.path.join(
                    load_path,
                    prefix + self.FILE_REPLAY_BUFFER + "_split_" + str(splits[0]),
                ),
                enable_cache=False,
            )
            shift_id = self.last_added_end_position - split_size * splits[0]
            data = {
                key: split_transitions[key][shift_id:]
                for key in self.replay_buffer_info.keys()
            }
            # The data that is right in front of the pointer needs to be added last
            to_add_last_data = {
                key: split_transitions[key][:shift_id]
                for key in self.replay_buffer_info.keys()
            }
            first_split_shift_length = len(data[self.some_data_key])
            # If pointer is right at the beginning of the split, no data needs to be shifted
            if first_split_shift_length > 0:
                self._add_episodic_data(
                    data_dictionary=data,
                    episode_ended=shifted_buffer_episode_ended[
                        :first_split_shift_length
                    ],
                )
                added_so_far += first_split_shift_length

            del data
            gc.collect()

            remaining_splits = splits[1:]
        else:
            # If no shift is needed, we can load the data as is
            added_so_far = 0
            remaining_splits = range(number_of_splits)
            shifted_buffer_episode_ended = self.buffer_episode_ended

        # Tell the replay buffer that due to the loading the episode was truncated, if the environment is not saved
        if not environment_saved:
            shifted_buffer_episode_ended[-1] = True

        # load all other splits
        for split in remaining_splits:
            data = {}
            split_transitions = file_handler.load_from_file(
                load_path=os.path.join(
                    load_path,
                    prefix + self.FILE_REPLAY_BUFFER + "_split_" + str(split),
                ),
                enable_cache=False,
            )
            split_size = len(split_transitions[self.some_data_key])
            self._add_episodic_data(
                data_dictionary=split_transitions,
                episode_ended=shifted_buffer_episode_ended[
                    added_so_far : added_so_far + split_size
                ],
            )
            added_so_far += split_size

            del data
            gc.collect()

        # add last bit of first split
        if shift_needed:
            if len(to_add_last_data[self.some_data_key]) > 0:
                self._add_episodic_data(
                    data_dictionary=to_add_last_data,
                    episode_ended=shifted_buffer_episode_ended[added_so_far:],
                )

            del to_add_last_data
            gc.collect()

    def _load_cpprb(
        self,
        file_handler,
        load_path,
        prefix,
        rb_full,
        replay_buffer_info,
        environment_saved=False,
    ):
        all_transitions = file_handler.load_from_file(
            load_path=os.path.join(load_path, prefix + self.FILE_REPLAY_BUFFER)
        )
        data_dict = {}
        if rb_full:
            for key in replay_buffer_info.keys():
                # shift data according to pointer
                data_dict[key] = np.vstack(
                    [
                        all_transitions[key][self.last_added_end_position :],
                        all_transitions[key][: self.last_added_end_position],
                    ]
                )
                shifted_episode_ended_bools = np.concatenate(
                    [
                        self.buffer_episode_ended[self.last_added_end_position :],
                        self.buffer_episode_ended[: self.last_added_end_position],
                    ]
                )
            # Tell the replay buffer that due to the loading the last episode was truncated if the environment is not saved
            if not environment_saved:
                shifted_episode_ended_bools[-1] = True
        else:
            data_dict = all_transitions
            shifted_episode_ended_bools = self.buffer_episode_ended
            # Tell the replay buffer that due to the loading the last episode was truncated if the environment is not saved
            num_entries = len(data_dict[self.some_data_key])
            if not environment_saved:
                shifted_episode_ended_bools[num_entries - 1] = True

        self._add_episodic_data(
            data_dictionary=data_dict, episode_ended=shifted_episode_ended_bools
        )

        del data_dict
        gc.collect()

    def _calculate_memory_size(self) -> int:

        bytes_per_transition = 0
        transition = self.replay_buffer.sample(1)
        for key in self.replay_buffer_info.keys():

            bytes_per_transition += sys.getsizeof(deepcopy(transition[key][0]))

        return bytes_per_transition * self.replay_buffer.get_stored_size()

    def _create_initialization_replay_buffer_info(
        self, replay_buffer_info: Dict, additional_arguments: Dict[str, Any]
    ) -> Dict:
        reduced_replay_buffer_info = deepcopy(replay_buffer_info)
        # remove next_of from replay_buffer_info
        if "next_of" in additional_arguments.keys():
            if isinstance(additional_arguments["next_of"], list):
                for key in additional_arguments["next_of"]:
                    if key not in replay_buffer_info.keys():
                        raise KeyError("next_of key {key} is not in replay_buffer_info")
                    del reduced_replay_buffer_info[f"next_{key}"]
            else:
                if additional_arguments["next_of"] not in replay_buffer_info.keys():
                    raise KeyError(
                        f"next_of key {additional_arguments['next_of']} is not in replay_buffer_info"
                    )
                del reduced_replay_buffer_info[
                    f"next_{additional_arguments['next_of']}"
                ]

        return reduced_replay_buffer_info
