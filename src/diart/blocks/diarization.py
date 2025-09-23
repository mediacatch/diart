from __future__ import annotations

from typing import Sequence
import logging
import time

import numpy as np
import torch
from pyannote.core import Annotation, SlidingWindowFeature, SlidingWindow, Segment
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.diarization import DiarizationErrorRate
from typing_extensions import Literal

from . import base
from .aggregation import DelayedAggregation
from .clustering import OnlineSpeakerClustering
from .embedding import OverlapAwareSpeakerEmbedding
from .segmentation import SpeakerSegmentation
from .utils import Binarize
from .. import models as m
from ..utils import SystemMonitor

# Configure logger for diarization pipeline
logger = logging.getLogger(__name__)


class SpeakerDiarizationConfig(base.PipelineConfig):
    def __init__(
        self,
        segmentation: m.SegmentationModel | None = None,
        embedding: m.EmbeddingModel | None = None,
        duration: float = 5,
        step: float = 0.5,
        latency: float | Literal["max", "min"] | None = None,
        tau_active: float = 0.6,
        rho_update: float = 0.3,
        delta_new: float = 1,
        gamma: float = 3,
        beta: float = 10,
        max_speakers: int = 20,
        normalize_embedding_weights: bool = False,
        log_level: int = logging.INFO,
        compile: bool = False,
        device: torch.device | None = None,
        sample_rate: int = 16000,
        **kwargs,
    ):
        # Default segmentation model is pyannote/segmentation
        self.segmentation = segmentation or m.SegmentationModel.from_pyannote(
            "pyannote/segmentation"
        )

        # Default embedding model is pyannote/embedding
        self.embedding = embedding or m.EmbeddingModel.from_pyannote(
            "pyannote/embedding"
        )

        self._duration = duration
        self._sample_rate = sample_rate

        # Latency defaults to the step duration
        self._step = step
        self._latency = latency
        if self._latency is None or self._latency == "min":
            self._latency = self._step
        elif self._latency == "max":
            self._latency = self._duration

        self.tau_active = tau_active
        self.rho_update = rho_update
        self.delta_new = delta_new
        self.gamma = gamma
        self.beta = beta
        self.max_speakers = max_speakers
        self.normalize_embedding_weights = normalize_embedding_weights
        self.compile = compile
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.setLevel(log_level)

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def step(self) -> float:
        return self._step

    @property
    def latency(self) -> float:
        return self._latency

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


class SpeakerDiarization(base.Pipeline):
    def __init__(self, config: SpeakerDiarizationConfig | None = None):
        self._config = SpeakerDiarizationConfig() if config is None else config

        msg = f"Latency should be in the range [{self._config.step}, {self._config.duration}]"
        assert self._config.step <= self._config.latency <= self._config.duration, msg

        self.segmentation = SpeakerSegmentation(
            self._config.segmentation, self._config.device, self._config.compile
        )
        self.embedding = OverlapAwareSpeakerEmbedding(
            self._config.embedding,
            self._config.gamma,
            self._config.beta,
            norm=1,
            normalize_weights=self._config.normalize_embedding_weights,
            device=self._config.device,
            use_compile=self._config.compile,
        )
        self.pred_aggregation = DelayedAggregation(
            self._config.step,
            self._config.latency,
            strategy="hamming",
            cropping_mode="loose",
        )
        self.audio_aggregation = DelayedAggregation(
            self._config.step,
            self._config.latency,
            strategy="first",
            cropping_mode="center",
        )
        self.binarize = Binarize(self._config.tau_active)

        # System monitoring setup
        self.system_monitor = SystemMonitor(logger)

        # Internal state, handle with care
        self.timestamp_shift = 0
        self.clustering = None
        self.chunk_buffer, self.pred_buffer = [], []

        # Track statistics
        self._call_count = 0
        self._total_processing_time = 0
        self._segmentation_times = []
        self._embedding_times = []
        self._clustering_times = []
        self._last_output_time = None

        self.reset()

    @staticmethod
    def get_config_class() -> type:
        return SpeakerDiarizationConfig

    @staticmethod
    def suggest_metric() -> BaseMetric:
        return DiarizationErrorRate(collar=0, skip_overlap=False)

    @staticmethod
    def hyper_parameters() -> Sequence[base.HyperParameter]:
        return [base.TauActive, base.RhoUpdate, base.DeltaNew]

    @property
    def config(self) -> SpeakerDiarizationConfig:
        return self._config

    def set_timestamp_shift(self, shift: float):
        self.timestamp_shift = shift

    def reset(self):
        self.set_timestamp_shift(0)
        self.clustering = OnlineSpeakerClustering(
            self.config.tau_active,
            self.config.rho_update,
            self.config.delta_new,
            "cosine",
            self.config.max_speakers,
        )
        self.chunk_buffer, self.pred_buffer = [], []
        self._call_count = 0
        self._total_processing_time = 0
        self._segmentation_times = []
        self._embedding_times = []
        self._clustering_times = []
        self._last_output_time = None

    def __call__(
        self, waveforms: Sequence[SlidingWindowFeature]
    ) -> Sequence[tuple[Annotation, SlidingWindowFeature]]:
        """Diarize the next audio chunks of an audio stream.

        Parameters
        ----------
        waveforms: Sequence[SlidingWindowFeature]
            A sequence of consecutive audio chunks from an audio stream.

        Returns
        -------
        Sequence[tuple[Annotation, SlidingWindowFeature]]
            Speaker diarization of each chunk alongside their corresponding audio.
        """
        start_time = time.time()
        self._call_count += 1

        batch_size = len(waveforms)
        msg = "Pipeline expected at least 1 input"
        assert batch_size >= 1, msg

        # Create batch from chunk sequence, shape (batch, samples, channels)
        batch = torch.stack([torch.from_numpy(w.data) for w in waveforms])

        expected_num_samples = int(
            np.rint(self.config.duration * self.config.sample_rate)
        )
        msg = f"Expected {expected_num_samples} samples per chunk, but got {batch.shape[1]}"
        assert batch.shape[1] == expected_num_samples, msg

        # Extract segmentation and embeddings
        self.system_monitor.log_system_info("PRE-SEGMENTATION", logging.DEBUG)

        seg_start = time.time()
        segmentations = self.segmentation(batch)  # shape (batch, frames, speakers)
        seg_time = time.time() - seg_start
        self._segmentation_times.append(seg_time)
        logger.debug(f"[SpeakerDiarization] Segmentation took {seg_time:.5f}s, shape: {segmentations.shape}")

        # embeddings has shape (batch, speakers, emb_dim)
        self.system_monitor.log_system_info("PRE-EMBEDDING", logging.DEBUG)

        emb_start = time.time()
        embeddings = self.embedding(batch, segmentations)
        emb_time = time.time() - emb_start
        self._embedding_times.append(emb_time)
        logger.debug(f"[SpeakerDiarization] Embedding extraction took {emb_time:.5f}s, shape: {embeddings.shape}")

        seg_resolution = waveforms[0].extent.duration / segmentations.shape[1]

        outputs = []
        for idx, (wav, seg, emb) in enumerate(zip(waveforms, segmentations, embeddings)):
            # Add timestamps to segmentation
            sw = SlidingWindow(
                start=wav.extent.start,
                duration=seg_resolution,
                step=seg_resolution,
            )
            seg = SlidingWindowFeature(seg.cpu().numpy(), sw)

            # Update clustering state and permute segmentation
            clust_start = time.time()
            permuted_seg = self.clustering(seg, emb)
            clust_time = time.time() - clust_start
            self._clustering_times.append(clust_time)
            logger.debug(f"[SpeakerDiarization] Clustering for chunk {idx} took {clust_time:.3f}s")

            # Update sliding buffer
            self.chunk_buffer.append(wav)
            self.pred_buffer.append(permuted_seg)

            # Aggregate buffer outputs for this time step
            agg_waveform = self.audio_aggregation(self.chunk_buffer)
            agg_prediction = self.pred_aggregation(self.pred_buffer)
            agg_prediction = self.binarize(agg_prediction)

            # Shift prediction timestamps if required
            if self.timestamp_shift != 0:
                shifted_agg_prediction = Annotation(agg_prediction.uri)
                for segment, track, speaker in agg_prediction.itertracks(
                    yield_label=True
                ):
                    new_segment = Segment(
                        segment.start + self.timestamp_shift,
                        segment.end + self.timestamp_shift,
                    )
                    shifted_agg_prediction[new_segment, track] = speaker
                agg_prediction = shifted_agg_prediction

            outputs.append((agg_prediction, agg_waveform))

            # Make place for new chunks in buffer if required
            if len(self.chunk_buffer) == self.pred_aggregation.num_overlapping_windows:
                self.chunk_buffer = self.chunk_buffer[1:]
                self.pred_buffer = self.pred_buffer[1:]

        # Track processing time and output interval
        processing_time = time.time() - start_time
        self._total_processing_time += processing_time

        current_time = time.time()
        if self._last_output_time:
            output_interval = current_time - self._last_output_time
            if output_interval > self.config.step * 2:
                logger.warning(f"[SpeakerDiarization] Large output interval: {output_interval:.3f}s (expected ~{self.config.step}s)")
        self._last_output_time = current_time

        return outputs
