from pathlib import Path
from typing import Text, Union

import torch
from torchcodec.decoders import AudioDecoder


FilePath = Union[Text, Path]


class AudioLoader:
    def __init__(self, sample_rate: int, mono: bool = True):
        self.sample_rate = sample_rate
        self.mono = mono

    def load(self, filepath: FilePath) -> torch.Tensor:
        """Load an audio file into a torch.Tensor.

        Parameters
        ----------
        filepath : FilePath
            Path to an audio file

        Returns
        -------
        waveform : torch.Tensor, shape (channels, samples)
        """
        num_channels = 1 if self.mono else None
        decoder = AudioDecoder(
            filepath, sample_rate=self.sample_rate, num_channels=num_channels
        )
        waveform = decoder.get_all_samples().data
        return waveform

    @staticmethod
    def get_duration(filepath: FilePath) -> float:
        """Get audio file duration in seconds.

        Parameters
        ----------
        filepath : FilePath
            Path to an audio file.

        Returns
        -------
        duration : float
            Duration in seconds.
        """
        decoder = AudioDecoder(filepath)
        duration = decoder.metadata.duration_seconds
        if duration is None:
            raise ValueError(f"Could not determine duration for {filepath}")
        return duration
