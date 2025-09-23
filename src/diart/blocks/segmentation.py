from typing import Optional, Union, Text
import logging

import torch
from einops import rearrange

from ..features import TemporalFeatures, TemporalFeatureFormatter
from ..models import SegmentationModel

logger = logging.getLogger(__name__)


class SpeakerSegmentation:
    def __init__(self, model: SegmentationModel, device: Optional[torch.device] = None, use_compile: bool = False):
        self.model = model
        self.model.eval()
        self.device = device
        if self.device is None:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        logger.info(f"SpeakerSegmentation model placed on device: {self.device}")
        if use_compile:
            self.model = torch.compile(self.model)
            logger.info("SpeakerSegmentation model compiled with torch.compile")
        self.formatter = TemporalFeatureFormatter()

    @staticmethod
    def from_pretrained(
        model,
        use_hf_token: Union[Text, bool, None] = True,
        device: Optional[torch.device] = None,
        use_compile: bool = False,
    ) -> "SpeakerSegmentation":
        seg_model = SegmentationModel.from_pretrained(model, use_hf_token)
        return SpeakerSegmentation(seg_model, device, use_compile)

    def __call__(self, waveform: TemporalFeatures) -> TemporalFeatures:
        """
        Calculate the speaker segmentation of input audio.

        Parameters
        ----------
        waveform: TemporalFeatures, shape (samples, channels) or (batch, samples, channels)

        Returns
        -------
        speaker_segmentation: TemporalFeatures, shape (batch, frames, speakers)
            The batch dimension is omitted if waveform is a `SlidingWindowFeature`.
        """
        with torch.no_grad():
            wave = rearrange(
                self.formatter.cast(waveform),
                "batch sample channel -> batch channel sample",
            )
            output = self.model(wave.to(self.device)).cpu()
        return self.formatter.restore_type(output)
