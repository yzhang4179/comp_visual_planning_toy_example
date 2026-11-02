import numpy as np
import torch

from cvp.utils import MinMaxNormalizer


def create_indices(
    episode_ends: np.ndarray, sequence_length: int, pad_before: int = 0, pad_after: int = 0
) -> np.ndarray:
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0  # episode start index
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]  # episode end index
        episode_length = end_idx - start_idx  # episode length

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


class SequenceSampler:
    def __init__(
        self,
        dataset: dict,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        zero_padding: bool = False,
    ):
        super().__init__()
        assert sequence_length >= 1
        self.root = dataset

        if keys is None:
            keys = list(self.root["data"].keys())

        episode_ends = self.root["meta"]["episode_ends"][:]

        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
        )
        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.zero_padding = zero_padding

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.root["data"][key]
            sample = input_arr[buffer_start_idx:buffer_end_idx]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(shape=(self.sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
                if not self.zero_padding:
                    if sample_start_idx > 0:
                        data[:sample_start_idx] = sample[0]
                    if sample_end_idx < self.sequence_length:
                        data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result


class ToyDataset(torch.utils.data.Dataset):
    '''
    Sliding windows of `horizon` points over the flower episodes in `dataset_path`, an array of shape (N, L, 2).
    The points are min-max normalized to [-1, 1]; the normalizer is kept in `self.normalizers["states"]`.
    '''
    def __init__(
        self,
        dataset_path: str,
        horizon: int = 240,
        pad_before: int = 0,
        pad_after: int = 0,
        ):

        # --- Prepare dataset Items --- #
        dataset = {"data": {"states": list()},
                   "meta": {"episode_ends": list()}}

        datas = np.load(dataset_path)  # (N, 600, 2)
        for data in datas:  # (600, 2)
            dataset["data"]["states"].append(data.astype(np.float32))
            dataset["meta"]["episode_ends"].append(data.shape[0])

        for k, v in dataset["data"].items():
            dataset["data"][k] = np.concatenate(v, axis=0)

        dataset["meta"]["episode_ends"] = np.cumsum(dataset["meta"]["episode_ends"])

        # --- Normalize States --- #
        normalizers = {}
        for k, v in dataset["data"].items():
            normalizers[k] = MinMaxNormalizer(v)
            dataset["data"][k] = normalizers[k].normalize(v)
        self.normalizers = normalizers

        # --- Create SequenceSampler --- #
        self.sampler = SequenceSampler(
                    dataset=dataset,
                    sequence_length=horizon,
                    pad_before=pad_before,
                    pad_after=pad_after,
                    zero_padding=False,
                    )

    def __str__(self):
        return f"Dataset: {len(self.sampler)} samples"

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        sample['states'] = torch.from_numpy(sample['states']).to(dtype=torch.float32)
        return sample
