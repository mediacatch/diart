import logging
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue, SimpleQueue
from typing import Any, AnyStr, Dict, Optional, Text, Tuple, Union

import numpy as np
import sounddevice as sd
import torch
from einops import rearrange
from rx.subject import Subject
from torchaudio.io import StreamReader
from websocket_server import WebsocketServer

from . import utils
from .audio import AudioLoader, FilePath

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class AudioSource(ABC):
    """Represents a source of audio that can start streaming via the `stream` property.

    Parameters
    ----------
    uri: Text
        Unique identifier of the audio source.
    sample_rate: int
        Sample rate of the audio source.
    """

    def __init__(self, uri: Text, sample_rate: int):
        self.uri = uri
        self.sample_rate = sample_rate
        self.stream = Subject()

    @property
    def duration(self) -> Optional[float]:
        """The duration of the stream if known. Defaults to None (unknown duration)."""
        return None

    @abstractmethod
    def read(self):
        """Start reading the source and yielding samples through the stream."""
        pass

    @abstractmethod
    def close(self):
        """Stop reading the source and close all open streams."""
        pass

    @abstractmethod
    def restart(self) -> bool:
        """Restart the audio source if applicable."""
        pass


class FileAudioSource(AudioSource):
    """Represents an audio source tied to a file.

    Parameters
    ----------
    file: FilePath
        Path to the file to stream.
    sample_rate: int
        Sample rate of the chunks emitted.
    padding: (float, float)
        Left and right padding to add to the file (in seconds).
        Defaults to (0, 0).
    block_duration: int
        Duration of each emitted chunk in seconds.
        Defaults to 0.5 seconds.
    """

    def __init__(
        self,
        file: FilePath,
        sample_rate: int,
        padding: Tuple[float, float] = (0, 0),
        block_duration: float = 0.5,
    ):
        super().__init__(Path(file).stem, sample_rate)
        self.loader = AudioLoader(self.sample_rate, mono=True)
        self._duration = self.loader.get_duration(file)
        self.file = file
        self.resolution = 1 / self.sample_rate
        self.block_size = int(np.rint(block_duration * self.sample_rate))
        self.padding_start, self.padding_end = padding
        self.is_closed = False

    @property
    def duration(self) -> Optional[float]:
        # The duration of a file is known
        return self.padding_start + self._duration + self.padding_end

    def read(self):
        """Send each chunk of samples through the stream"""
        waveform = self.loader.load(self.file)

        # Add zero padding at the beginning if required
        if self.padding_start > 0:
            num_pad_samples = int(np.rint(self.padding_start * self.sample_rate))
            zero_padding = torch.zeros(waveform.shape[0], num_pad_samples)
            waveform = torch.cat([zero_padding, waveform], dim=1)

        # Add zero padding at the end if required
        if self.padding_end > 0:
            num_pad_samples = int(np.rint(self.padding_end * self.sample_rate))
            zero_padding = torch.zeros(waveform.shape[0], num_pad_samples)
            waveform = torch.cat([waveform, zero_padding], dim=1)

        # Split into blocks
        _, num_samples = waveform.shape
        chunks = rearrange(
            waveform.unfold(1, self.block_size, self.block_size),
            "channel chunk sample -> chunk channel sample",
        ).numpy()

        # Add last incomplete chunk with padding
        if num_samples % self.block_size != 0:
            last_chunk = (
                waveform[:, chunks.shape[0] * self.block_size :].unsqueeze(0).numpy()
            )
            diff_samples = self.block_size - last_chunk.shape[-1]
            last_chunk = np.concatenate(
                [last_chunk, np.zeros((1, 1, diff_samples))], axis=-1
            )
            chunks = np.vstack([chunks, last_chunk])

        # Stream blocks
        for i, waveform in enumerate(chunks):
            try:
                if self.is_closed:
                    break
                self.stream.on_next(waveform)
            except BaseException as e:
                self.stream.on_error(e)
                break
        self.stream.on_completed()
        self.close()

    def close(self):
        self.is_closed = True


class MicrophoneAudioSource(AudioSource):
    """Audio source tied to a local microphone.

    Parameters
    ----------
    block_duration: int
        Duration of each emitted chunk in seconds.
        Defaults to 0.5 seconds.
    device: int | str | (int, str) | None
        Device identifier compatible for the sounddevice stream.
        If None, use the default device.
        Defaults to None.
    """

    def __init__(
        self,
        block_duration: float = 0.5,
        device: Optional[Union[int, Text, Tuple[int, Text]]] = None,
    ):
        # Use the lowest supported sample rate
        sample_rates = [16000, 32000, 44100, 48000]
        best_sample_rate = None
        for sr in sample_rates:
            try:
                sd.check_input_settings(device=device, samplerate=sr)
            except Exception:
                pass
            else:
                best_sample_rate = sr
                break
        super().__init__(f"input_device:{device}", best_sample_rate)

        # Determine block size in samples and create input stream
        self.block_size = int(np.rint(block_duration * self.sample_rate))
        self._mic_stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            latency=0,
            blocksize=self.block_size,
            callback=self._read_callback,
            device=device,
        )
        self._queue = SimpleQueue()

    def _read_callback(self, samples, *args):
        self._queue.put_nowait(samples[:, [0]].T)

    def read(self):
        self._mic_stream.start()
        while self._mic_stream:
            try:
                while self._queue.empty():
                    if self._mic_stream.closed:
                        break
                self.stream.on_next(self._queue.get_nowait())
            except BaseException as e:
                self.stream.on_error(e)
                break
        self.stream.on_completed()
        self.close()

    def close(self):
        self._mic_stream.stop()
        self._mic_stream.close()

class FFmpegAudioSource(AudioSource):
    """Audio source tied to a local microphone using FFmpeg.

    Parameters
    ----------
    block_duration: float
        Duration of each emitted chunk in seconds.
        Defaults to 0.5 seconds.
    device: str | None
        Device identifier for FFmpeg.
        Format varies by OS:
        - Linux: "hw:0" or "default" for ALSA, device path for pulse
        If None, use the default device.
        Defaults to None.
    sample_rate: int
        Sample rate in Hz.
        Defaults to 16000.
    buffer_size: int
        Size of the internal buffer for accumulating partial reads (in number of blocks).
        Defaults to 10.
    """

    def __init__(
        self,
        block_duration: float = 0.5,
        device: Optional[str] = None,
        sample_rate: int = 16000,
        buffer_size: int = 10,
    ):
        super().__init__(f'ffmpeg_input:{device}', sample_rate)

        self.block_duration = block_duration
        self.device = device if device else 'default'
        self.block_size = int(np.rint(block_duration * self.sample_rate))
        self.block_size_bytes = self.block_size * 4  # 32-bit float audio

        self._queue = Queue(maxsize=buffer_size)
        self._ffmpeg_process = None
        self._read_thread = None
        self._stop_flag = threading.Event()

        self._audio_buffer = BytesIO()
        self._buffer_lock = threading.Lock()

        # Track restart statistics
        self._last_restart_time = 0
        self._startup_time = time.time()

        # Restart/pause control
        self._restart_in_progress = threading.Event()

        self._ffmpeg_cmd = self._build_ffmpeg_command()

    def _build_ffmpeg_command(self):
        """Build the FFmpeg command for audio capture."""

        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel',
            'warning',
            '-f',
            'alsa',
            '-thread_queue_size',
            '4096',
            '-probesize',
            '32',
            '-analyzeduration',
            '0',
            '-i',
            self.device,
            '-af',
            'aresample=async=1:min_comp=0.001:min_hard_comp=0.100:first_pts=0',
            '-acodec',
            'pcm_f32le',
            '-ar',
            str(self.sample_rate),
            '-ac',
            '1',
            '-f',
            'f32le',
            '-fflags',
            'nobuffer+flush_packets',
            '-flags',
            'low_delay',
            '-avioflags',
            'direct',
            '-',  # Output to stdout
        ]
        return cmd

    def _restart_ffmpeg(self):
        """Restart the FFmpeg process when audio stream issues are detected."""
        current_time = time.time()
        logger.info('[FFmpegAudioSource] Attempting to restart FFmpeg process')

        # Prevent restarts within 30 seconds of startup
        if current_time - self._startup_time < 30.0:
            logger.debug('[FFmpegAudioSource] Skipping restart - within 30 seconds of startup')
            return False

        # Prevent too frequent restarts (at least 5 seconds apart)
        if current_time - self._last_restart_time < 5.0:
            logger.debug('[FFmpegAudioSource] Skipping restart - too recent')
            return False

        # Signal that restart is in progress to pause the read loop
        self._restart_in_progress.set()

        # Stop current process
        if self._ffmpeg_process:
            try:
                self._ffmpeg_process.terminate()
                self._ffmpeg_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    self._ffmpeg_process.kill()
                    self._ffmpeg_process.wait()
                except Exception as e:
                    logger.error(f'[FFmpegAudioSource] Error killing FFmpeg process: {e}')
            except Exception as e:
                logger.error(f'[FFmpegAudioSource] Error terminating FFmpeg process: {e}')

        # Clear buffer
        with self._buffer_lock:
            self._audio_buffer = BytesIO()

        # Clear queue
        cleared_items = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                cleared_items += 1
            except Exception:
                break

        # Start new process
        try:
            self._ffmpeg_process = subprocess.Popen(
                self._ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            self._last_restart_time = current_time
            logger.info('[FFmpegAudioSource] FFmpeg process restarted successfully')
            # Clear the restart flag to resume the read loop
            self._restart_in_progress.clear()
            return True
        except Exception as e:
            logger.error(f'[FFmpegAudioSource] Failed to restart FFmpeg: {e}')
            # Clear the restart flag even on failure
            logger.debug('[FFmpegAudioSource] Clearing restart in progress flag (after failure)')
            self._restart_in_progress.clear()
            return False

    def _read_ffmpeg_output(self):
        """Read audio data from FFmpeg stdout in a separate thread with buffering."""
        logger.debug('[FFmpegAudioSource] Starting FFmpeg output reader thread')

        consecutive_errors = 0
        max_consecutive_errors = 10
        last_read_time = time.time()
        last_chunk_time = time.time()
        timeout_seconds = 5.0
        read_chunk_size = 4096  # Read in smaller chunks from ffmpeg

        while not self._stop_flag.is_set() and self._ffmpeg_process:
            try:
                if self._restart_in_progress.is_set():
                    restart_wait_start = time.time()
                    last_log_time = 0
                    while self._restart_in_progress.is_set() and not self._stop_flag.is_set():
                        current_time = time.time()
                        if current_time - last_log_time >= 5.0:
                            wait_duration = current_time - restart_wait_start
                            logger.info(f'[FFmpegAudioSource] Reader thread waiting for restart completion ({wait_duration:.1f}s)')
                            last_log_time = current_time
                        time.sleep(0.01)
                    continue
                if self._ffmpeg_process.poll() is not None:
                    # Process has terminated
                    returncode = self._ffmpeg_process.returncode
                    logger.error(
                        f'[FFmpegAudioSource] FFmpeg process terminated with code {returncode}'
                    )

                    # Read any error output
                    if self._ffmpeg_process.stderr:
                        stderr_output = self._ffmpeg_process.stderr.read()
                        if stderr_output:
                            logger.error(
                                f"[FFmpegAudioSource] FFmpeg stderr: {stderr_output.decode('utf-8', errors='ignore')}"
                            )
                    break

                # Read available data from ffmpeg (non-blocking style with timeout)
                try:
                    audio_bytes = self._ffmpeg_process.stdout.read(read_chunk_size)
                    if audio_bytes:
                        logger.debug(f'[FFmpegAudioSource] Read {len(audio_bytes)} bytes from FFmpeg')
                except Exception as e:
                    logger.error(
                        f'[FFmpegAudioSource] Error reading from ffmpeg stdout: {e}'
                    )
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        break
                    continue

                if not audio_bytes:
                    # No data available
                    current_time = time.time()
                    no_data_duration = current_time - last_read_time
                    if no_data_duration > 1.0:  # Log every second when no data
                        logger.debug(f'[FFmpegAudioSource] No data for {no_data_duration:.1f}s')

                    if no_data_duration > timeout_seconds:
                        logger.warning(
                            f'[FFmpegAudioSource] Read timeout: No data for {timeout_seconds}s - attempting restart'
                        )
                        # Try to restart FFmpeg on timeout
                        if self._restart_ffmpeg():
                            # Reset timing after successful restart
                            last_read_time = time.time()
                            last_chunk_time = time.time()
                            # Reset error counters and continue with new process
                            consecutive_errors = 0
                            continue
                        else:
                            logger.error('[FFmpegAudioSource] Restart failed after timeout, stopping reader thread')
                            break
                    time.sleep(0.001)
                    continue

                # Reset error counter and update read time
                consecutive_errors = 0
                last_read_time = time.time()

                # Add to buffer
                restart_needed = False
                lock_acquire_start = time.time()
                with self._buffer_lock:
                    lock_acquire_time = time.time() - lock_acquire_start
                    if lock_acquire_time > self.block_duration:
                        logger.warning(f'[FFmpegAudioSource] Buffer lock took {lock_acquire_time:.3f}s to acquire (expected <{self.block_duration:.3f}s)')
                    self._audio_buffer.write(audio_bytes)
                    buffer_size = self._audio_buffer.tell()
                    logger.debug(f'[FFmpegAudioSource] Buffer size after write: {buffer_size} bytes (need {self.block_size_bytes} for complete block)')

                    # Check if we have enough data for one or more complete blocks
                    blocks_processed = 0
                    while buffer_size >= self.block_size_bytes:
                        blocks_processed += 1
                        # Read a complete block from buffer
                        self._audio_buffer.seek(0)
                        block_bytes = self._audio_buffer.read(self.block_size_bytes)

                        # Convert to numpy array (already in float32 format from ffmpeg)
                        audio_data = np.frombuffer(block_bytes, dtype=np.float32)
                        # Reshape to match expected format (1, block_size)
                        audio_data = audio_data.reshape(1, -1)

                        # Monitor chunk intervals
                        current_time = time.time()
                        chunk_interval = current_time - last_chunk_time
                        expected_interval = self.block_duration
                        if chunk_interval > expected_interval * 3:  # More than 3x step size
                            logger.warning(f'[FFmpegAudioSource] Large chunk interval: {chunk_interval:.3f}s (expected ~{expected_interval:.3f}s) - attempting restart')
                            # Need to release buffer lock before restart to avoid deadlock
                            # Save remaining buffer data first
                            remaining = self._audio_buffer.read()
                            self._audio_buffer = BytesIO()
                            self._audio_buffer.write(remaining)
                            # Exit the with block to release lock, then restart
                            restart_needed = True
                            break
                        last_chunk_time = current_time

                        # Try to put in queue
                        try:
                            self._queue.put(audio_data, block=False)
                            logger.debug(f'[FFmpegAudioSource] Added block {blocks_processed} to queue (chunk interval: {chunk_interval:.3f}s)')
                        except Exception as e:
                            logger.debug(f'[FFmpegAudioSource] Failed to add block to queue: {e}')

                        # Remove processed data from buffer
                        remaining = self._audio_buffer.read()
                        self._audio_buffer = BytesIO()
                        self._audio_buffer.write(remaining)
                        buffer_size = len(remaining)

                    if blocks_processed > 0:
                        logger.debug(f'[FFmpegAudioSource] Processed {blocks_processed} blocks, {buffer_size} bytes remaining in buffer')

                # Handle restart outside the buffer lock to avoid deadlock
                if restart_needed:
                    if self._restart_ffmpeg():
                        # Reset timing after successful restart
                        last_chunk_time = time.time()
                        last_read_time = time.time()
                        # Reset error counters and continue with new process
                        consecutive_errors = 0
                        continue
                    else:
                        # Restart failed, break out of loop
                        logger.error('[FFmpegAudioSource] Restart failed, stopping reader thread')
                        break

            except Exception as e:
                logger.error(f'[FFmpegAudioSource] Error in reader thread: {e}')
                consecutive_errors += 1

                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        f'[FFmpegAudioSource] Too many consecutive errors ({consecutive_errors}), stopping reader'
                    )
                    break

                time.sleep(0.01)

    def read(self):
        """Read audio chunks from the microphone via FFmpeg."""
        try:
            # Ensure restart flag is cleared at start
            self._restart_in_progress.clear()

            # Start FFmpeg process with unbuffered output
            self._ffmpeg_process = subprocess.Popen(
                self._ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered
            )

            # Start reader thread
            self._read_thread = threading.Thread(target=self._read_ffmpeg_output, daemon=True)
            self._read_thread.start()

            empty_queue_count = 0
            max_empty_queue_wait = 1000
            consecutive_errors = 0
            max_consecutive_errors = 3

            # Give ffmpeg a moment to start up
            time.sleep(0.1)

            while not self._stop_flag.is_set():
                try:
                    # Wait if restart is in progress
                    if self._restart_in_progress.is_set():
                        restart_wait_start = time.time()
                        last_log_time = 0
                        while self._restart_in_progress.is_set() and not self._stop_flag.is_set():
                            current_time = time.time()
                            if current_time - last_log_time >= 5.0:
                                wait_duration = current_time - restart_wait_start
                                logger.info(f'[FFmpegAudioSource] Waiting for restart completion ({wait_duration:.1f}s)')
                                last_log_time = current_time
                            time.sleep(0.01)
                        continue

                    # Check if reader thread is still alive
                    if not self._read_thread.is_alive():
                        logger.error(
                            '[FFmpegAudioSource] Reader thread died unexpectedly'
                        )
                        break

                    # Try to get chunk from queue with timeout
                    try:
                        chunk = self._queue.get(timeout=0.05)
                        empty_queue_count = 0
                        logger.debug(f'[FFmpegAudioSource] Got chunk from queue, queue size: {self._queue.qsize()}')
                    except Empty:
                        empty_queue_count += 1

                        if empty_queue_count > max_empty_queue_wait:
                            logger.warning(
                                '[FFmpegAudioSource] No data for extended period, checking ffmpeg status'
                            )
                            if self._ffmpeg_process.poll() is not None:
                                logger.error(
                                    '[FFmpegAudioSource] FFmpeg process terminated'
                                )
                                break
                            empty_queue_count = 0  # Reset but continue
                        continue

                    # Reset error counter on successful read
                    consecutive_errors = 0

                    # Send chunk to stream
                    try:
                        self.stream.on_next(chunk)
                    except Exception as e:
                        logger.error(
                            f'[FFmpegAudioSource] Failed to send chunk to stream: {e}'
                        )
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            logger.error(
                                '[FFmpegAudioSource] Too many consecutive stream errors, stopping'
                            )
                            break

                except KeyboardInterrupt:
                    logger.info('[FFmpegAudioSource] Keyboard interrupt received')
                    break
                except Exception as e:
                    logger.error(
                        f'[FFmpegAudioSource] Unexpected error in read loop: {e}',
                        exc_info=True,
                    )
                    self.stream.on_error(e)
                    break

        except Exception as e:
            logger.error(f'[FFmpegAudioSource] Failed to start FFmpeg: {e}')
            self.stream.on_error(e)
        finally:
            self.stream.on_completed()
            self.close()

    def close(self):
        """Close the FFmpeg process and clean up resources."""
        self._stop_flag.set()

        # Check for remaining buffered data
        with self._buffer_lock:
            remaining_bytes = self._audio_buffer.tell()
            if remaining_bytes > 0:
                logger.warning(
                    f'[FFmpegAudioSource] Discarding {remaining_bytes} bytes of buffered audio'
                )

        # Terminate FFmpeg process
        if self._ffmpeg_process:
            try:
                self._ffmpeg_process.terminate()
                # Wait briefly for graceful termination
                try:
                    self._ffmpeg_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    # Force kill if necessary
                    self._ffmpeg_process.kill()
                    self._ffmpeg_process.wait()
            except Exception as e:
                logger.error(f'[FFmpegAudioSource] Error terminating FFmpeg: {e}')

        # Wait for reader thread
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=3.0)
            if self._read_thread.is_alive():
                logger.warning(
                    '[FFmpegAudioSource] Reader thread did not terminate cleanly'
                )

    def restart(self) -> bool:
        """Restart the FFmpeg audio source."""
        return self._restart_ffmpeg()


class WebSocketAudioSource(AudioSource):
    """Represents a source of audio coming from the network using the WebSocket protocol.

    Parameters
    ----------
    sample_rate: int
        Sample rate of the chunks emitted.
    host: Text
        The host to run the websocket server.
        Defaults to 127.0.0.1.
    port: int
        The port to run the websocket server.
        Defaults to 7007.
    key: Text | Path | None
        Path to a key if using SSL.
        Defaults to no key.
    certificate: Text | Path | None
        Path to a certificate if using SSL.
        Defaults to no certificate.
    """

    def __init__(
        self,
        sample_rate: int,
        host: Text = "127.0.0.1",
        port: int = 7007,
        key: Optional[Union[Text, Path]] = None,
        certificate: Optional[Union[Text, Path]] = None,
    ):
        # FIXME sample_rate is not being used, this can be confusing and lead to incompatibilities.
        #  I would prefer the client to send a JSON with data and sample rate, then resample if needed
        super().__init__(f"{host}:{port}", sample_rate)
        self.client: Optional[Dict[Text, Any]] = None
        self.server = WebsocketServer(host, port, key=key, cert=certificate)
        self.server.set_fn_message_received(self._on_message_received)

    def _on_message_received(
        self,
        client: Dict[Text, Any],
        server: WebsocketServer,
        message: AnyStr,
    ):
        # Only one client at a time is allowed
        if self.client is None or self.client["id"] != client["id"]:
            self.client = client
        # Send decoded audio to pipeline
        self.stream.on_next(utils.decode_audio(message))

    def read(self):
        """Starts running the websocket server and listening for audio chunks"""
        self.server.run_forever()

    def close(self):
        """Close the websocket server"""
        if self.server is not None:
            self.stream.on_completed()
            self.server.shutdown_gracefully()

    def send(self, message: AnyStr):
        """Send a message through the current websocket.

        Parameters
        ----------
        message: AnyStr
            Bytes or string to send.
        """
        if len(message) > 0:
            self.server.send_message(self.client, message)


class TorchStreamAudioSource(AudioSource):
    def __init__(
        self,
        uri: Text,
        sample_rate: int,
        streamer: StreamReader,
        stream_index: Optional[int] = None,
        block_duration: float = 0.5,
    ):
        super().__init__(uri, sample_rate)
        self.block_size = int(np.rint(block_duration * self.sample_rate))
        self._streamer = streamer
        self._streamer.add_basic_audio_stream(
            frames_per_chunk=self.block_size,
            stream_index=stream_index,
            format="fltp",
            sample_rate=self.sample_rate,
        )
        self.is_closed = False

    def read(self):
        for item in self._streamer.stream():
            try:
                if self.is_closed:
                    break
                # shape (samples, channels) to (1, samples)
                chunk = np.mean(item[0].numpy(), axis=1, keepdims=True).T
                self.stream.on_next(chunk)
            except BaseException as e:
                self.stream.on_error(e)
                break
        self.stream.on_completed()
        self.close()

    def close(self):
        self.is_closed = True


class AppleDeviceAudioSource(TorchStreamAudioSource):
    def __init__(
        self,
        sample_rate: int,
        device: str = "0:0",
        stream_index: int = 0,
        block_duration: float = 0.5,
    ):
        uri = f"apple_input_device:{device}"
        streamer = StreamReader(device, format="avfoundation")
        super().__init__(uri, sample_rate, streamer, stream_index, block_duration)
