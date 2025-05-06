import os
from typing import Dict, Any, Generator

from athlete.global_objects import RNGHandler
from athlete.saving.saveable_component import SaveContext


class OnPolicyBuffer:
    """A simple buffer for storing arbitrary data needed by on-policy algorithms.

    This buffer stores experience batches for on-policy algorithms like PPO and provides
    functionality to generate shuffled mini-batches during policy updates. Unlike
    replay buffers used in off-policy algorithms, this buffer is cleared and refilled
    during each update cycle.
    """

    SAVE_FILE_NAME = "on_policy_buffer"
    FILE_EMPTY_SIGNAL = "empty"

    def __init__(
        self,
        mini_batch_size: int,
        save_file_name: str = SAVE_FILE_NAME,
    ) -> None:
        """Initializes the OnPolicyBuffer with a mini batch size and a save file name.

        Args:
            mini_batch_size (int): Size of the mini batches to be generated. If this is not a fraction of the total size, the last batch will be smaller.
            save_file_name (str, optional): Name of the file to save the buffer to. Defaults to "on_policy_buffer".
        """

        self.data = {}
        self.size = 0
        self.all_keys = []
        self.mini_batch_size = mini_batch_size
        self.random_number_generator = RNGHandler.get_random_number_generator()

        self.save_file_name = save_file_name

    def set_data(self, data_dict: Dict[str, Any]) -> None:
        """Set the data of the buffer. Old data will be overwritten.

        Args:
            data_dict (Dict[str, Any]): Dictionary containing the data to be stored in the buffer. The keys of the dictionary will be used to access the data.
        """
        self.data = data_dict
        self.all_keys = list(data_dict.keys())
        self.size = data_dict[self.all_keys[0]].shape[0]

    def generate_shuffled_batched_epoch(self) -> Generator[Dict[str, Any], None, None]:
        """Generates shuffled mini-batches of data from the buffer. One generator call will generate one epoch of data. The data is shuffled before being split into mini-batches.

        Yields:
            Generator[Dict[str, Any], None, None]: Generator yielding mini-batches of data. Each mini-batch is a dictionary containing the data for each key in the buffer.
        """

        # shuffle the data
        shuffled_ids = self.random_number_generator.permutation(self.size)

        for start in range(0, self.size, self.mini_batch_size):
            end = min(start + self.mini_batch_size, self.size)
            batch_ids = shuffled_ids[start:end]

            batch_data = {}
            for key in self.all_keys:
                batch_data[key] = self.data[key][batch_ids]

            yield batch_data

    def save_checkpoint(self, context: SaveContext):
        save_path = os.path.join(
            context.save_path, context.prefix + self.save_file_name
        )

        if not self.data:
            # Data is empty, save an file only containing an indicator
            to_save = self.FILE_EMPTY_SIGNAL
        else:
            to_save = self.data

        context.file_handler.save_to_file(to_save=to_save, save_path=save_path)

    def load_checkpoint(self, context: SaveContext):
        load_path = os.path.join(
            context.save_path, context.prefix + self.save_file_name
        )

        data = context.file_handler.load_from_file(load_path)
        if data == self.FILE_EMPTY_SIGNAL:
            # Data is empty, set an empty dictionary
            self.data = {}
            self.all_keys = []
            self.size = 0
        else:
            self.set_data(data)
