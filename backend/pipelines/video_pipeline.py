# backend/pipelines/video_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Frame-by-frame video processing using the perception pipeline.
#   Extracts frames from a video at a configurable FPS, runs the full
#   WorkspaceVision pipeline on each frame, and returns per-frame results.
#
# HOW IT WORKS:
#   1. OpenCV reads the video file
#   2. Frames are sampled at VIDEO_FPS_PROCESS (default: 1 frame/sec)
#   3. Each sampled frame runs through PerceptionPipeline.analyze()
#   4. Results are collected into a VideoAnalysisResult
#   5. A scene summary is generated across all frames
#
# DESIGN:
#   - Memory safe: frames processed one at a time, not loaded all at once
#   - Progress callback: optional function called after each frame
#   - Resumable: frame_limit parameter caps processing for large videos
# ─────────────────────────────────────────────────────────────────────────────


import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings
from pipelines.perception_pipeline import PerceptionPipeline, AnalysisResult


logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    """
    Analysis result for one video frame.

    Attributes:
        frame_index    : Index of this frame in the sampled sequence (0, 1, 2...)
        timestamp_sec  : Timestamp in the original video in seconds
        analysis       : Full AnalysisResult from the perception pipeline
        processing_time: Time taken to process this frame in seconds
    """
    frame_index     : int
    timestamp_sec   : float
    analysis        : AnalysisResult
    processing_time : float


    def to_dict(self) -> dict:
        return {
            "frame_index"     : self.frame_index,
            "timestamp_sec"   : round(self.timestamp_sec, 2),
            "processing_time" : round(self.processing_time, 3),
            "object_count"    : self.analysis.get_object_count(),
            "objects"         : self.analysis.get_labels(),
            "detections"      : [d.to_dict() if hasattr(d, 'to_dict') else d for d in self.analysis.detections],
            "annotated_image" : self.analysis.annotated_image_b64,
            "scene_caption"   : self.analysis.scene_caption,
            "timings"         : self.analysis.timings,
            "error"           : self.analysis.error,
        }


@dataclass
class VideoAnalysisResult:
    """
    Complete result of processing one video file.

    Attributes:
        frame_results      : List of FrameResult, one per sampled frame
        total_frames       : Total frames processed
        video_duration_sec : Duration of the video in seconds
        fps_processed      : Frames per second that were sampled
        total_time         : Total wall time for the full video analysis
        all_labels         : Set of all unique object labels seen across frames
        summary            : Text summary of objects seen across the video
    """
    frame_results       : List[FrameResult]  = field(default_factory=list)
    total_frames        : int                = 0
    video_duration_sec  : float              = 0.0
    fps_processed       : float              = 1.0
    total_time          : float              = 0.0
    all_labels          : List[str]          = field(default_factory=list)
    summary             : str                = ""


    def to_dict(self) -> dict:
        return {
            "total_frames"       : self.total_frames,
            "video_duration_sec" : round(self.video_duration_sec, 2),
            "fps_processed"      : self.fps_processed,
            "total_time"         : round(self.total_time, 2),
            "all_labels"         : self.all_labels,
            "summary"            : self.summary,
            "frames"             : [f.to_dict() for f in self.frame_results],
        }


# ── Video pipeline ────────────────────────────────────────────────────────────

class VideoPipeline:
    """
    Processes video files frame-by-frame using WorkspaceVision.

    Usage:
        pipeline = VideoPipeline()
        result   = pipeline.analyze_video("workspace_recording.mp4")

        print(result.summary)
        for frame in result.frame_results:
            print(f"t={frame.timestamp_sec}s: {frame.analysis.get_labels()}")
    """

    def __init__(self):
        """
        Initialises using the singleton PerceptionPipeline.
        Does NOT reload models.
        """
        self.settings   = get_settings()
        self.pipeline   = PerceptionPipeline()
        self.fps_process= self.settings.VIDEO_FPS_PROCESS  # from config.py

        logger.info(
            f"[VideoPipeline] Initialised. "
            f"Processing at {self.fps_process} FPS."
        )


    # ──────────────────────────────────────────────────────────────────────────
    def analyze_video(
        self,
        video_path         : str,
        frame_limit        : Optional[int]      = None,
        confidence_threshold: float             = 0.3,
        skip_segmentation  : bool               = True,
        labels             : Optional[List[str]] = None,
        progress_callback  : Optional[Callable] = None,
    ) -> VideoAnalysisResult:
        """
        Processes a video file and returns per-frame analysis results.

        Args:
            video_path            : Path to video file (.mp4, .avi, .mov)
            frame_limit           : Max frames to process (None = all frames)
                                    Use this to cap processing on long videos
            confidence_threshold  : Min detection confidence (default 0.3)
            skip_segmentation     : Skip SAM2 for speed (default True for video)
                                    Video processing prioritises speed over masks
            progress_callback     : Optional function(frame_idx, total_frames)
                                    Called after each frame — use for progress bars

        Returns:
            VideoAnalysisResult with per-frame results and overall summary
        """

        video_path = str(video_path)
        result     = VideoAnalysisResult(fps_processed=self.fps_process)
        t_start    = time.time()

        # ── Open video with OpenCV ─────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[VideoPipeline] Cannot open video: {video_path}")
            result.summary = f"Error: Could not open video file {video_path}"
            return result

        # ── Get video metadata ─────────────────────────────────────────────
        video_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec   = total_frames / video_fps

        # How many original frames to skip between each processed frame
        # e.g. video is 30fps, we process at 1fps → skip every 30 frames
        frame_interval = max(1, int(video_fps / self.fps_process))

        logger.info(
            f"[VideoPipeline] Video: {duration_sec:.1f}s | "
            f"{video_fps:.0f}fps | {total_frames} frames | "
            f"Processing every {frame_interval} frames."
        )

        result.video_duration_sec = duration_sec
        all_labels  = set()
        frame_idx   = 0   # index of sampled frames (not original frame number)
        orig_frame  = 0   # index in original video

        try:
            while True:
                # Read next frame
                ret, frame_bgr = cap.read()
                if not ret:
                    break   # end of video

                # Skip frames we don't want to process
                if orig_frame % frame_interval != 0:
                    orig_frame += 1
                    continue

                # Check frame limit
                if frame_limit and frame_idx >= frame_limit:
                    logger.info(f"[VideoPipeline] Frame limit {frame_limit} reached.")
                    break

                # ── Process this frame ─────────────────────────────────────
                t_frame = time.time()
                timestamp_sec = orig_frame / video_fps

                # OpenCV loads frames as BGR — convert to RGB for PIL
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Run perception pipeline on this frame
                analysis = self.pipeline.analyze(
                    source               = pil_image,
                    labels               = labels,
                    confidence_threshold = confidence_threshold,
                    skip_segmentation    = skip_segmentation,
                )

                frame_time = time.time() - t_frame

                # ── Store result ───────────────────────────────────────────
                frame_result = FrameResult(
                    frame_index     = frame_idx,
                    timestamp_sec   = timestamp_sec,
                    analysis        = analysis,
                    processing_time = frame_time,
                )
                result.frame_results.append(frame_result)

                # Collect all unique labels seen in this frame
                for label in analysis.get_labels():
                    all_labels.add(label)

                logger.info(
                    f"[VideoPipeline] Frame {frame_idx} "
                    f"(t={timestamp_sec:.1f}s): "
                    f"{analysis.get_object_count()} objects | "
                    f"{frame_time:.2f}s"
                )

                # ── Progress callback ──────────────────────────────────────
                if progress_callback:
                    estimated_total = (
                        frame_limit or int(duration_sec * self.fps_process)
                    )
                    progress_callback(frame_idx, estimated_total)

                frame_idx  += 1
                orig_frame += 1

        finally:
            # Always release the video capture — prevents file handle leak
            cap.release()

        # ── Build final result ─────────────────────────────────────────────
        result.total_frames = frame_idx
        result.all_labels   = sorted(list(all_labels))
        result.total_time   = time.time() - t_start
        result.summary      = self._generate_summary(result)

        logger.info(
            f"[VideoPipeline] Complete. "
            f"{frame_idx} frames in {result.total_time:.1f}s. "
            f"Objects seen: {result.all_labels}"
        )

        return result


    # ──────────────────────────────────────────────────────────────────────────
    def analyze_video_bytes(
        self,
        video_bytes        : bytes,
        filename           : str   = "upload.mp4",
        frame_limit        : Optional[int] = 30,
        confidence_threshold: float        = 0.3,
        skip_segmentation  : bool          = True,
        labels             : Optional[List[str]] = None,
    ) -> VideoAnalysisResult:
        """
        Processes video from raw bytes — used by the FastAPI upload endpoint.
        Saves to a temp file, processes it, then deletes the temp file.

        Args:
            video_bytes          : Raw video file bytes from HTTP upload
            filename             : Original filename (used for temp file extension)
            frame_limit          : Cap frames to avoid timeout on long videos
            confidence_threshold : Min detection confidence
            skip_segmentation    : Skip SAM2 (recommended for video)

        Returns:
            VideoAnalysisResult
        """
        import tempfile
        import os

        # Save bytes to temp file — OpenCV needs a file path, not bytes
        suffix   = Path(filename).suffix or ".mp4"
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False
            ) as tmp:
                tmp.write(video_bytes)
                tmp_path = tmp.name

            return self.analyze_video(
                video_path           = tmp_path,
                frame_limit          = frame_limit,
                confidence_threshold = confidence_threshold,
                skip_segmentation    = skip_segmentation,
                labels               = labels,
            )

        finally:
            # Always clean up temp file
            if tmp_path and Path(tmp_path).exists():
                os.unlink(tmp_path)


    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _generate_summary(result: VideoAnalysisResult) -> str:
        """
        Generates a text summary of the video analysis.
        Describes what objects were seen and for how long.

        Args:
            result: VideoAnalysisResult with frame_results populated

        Returns:
            Summary string
        """
        if not result.frame_results:
            return "No frames were processed."

        # Count how many frames each label appeared in
        label_counts: dict = {}
        for frame in result.frame_results:
            for label in frame.analysis.get_labels():
                label_counts[label] = label_counts.get(label, 0) + 1

        if not label_counts:
            return f"Processed {result.total_frames} frames. No objects detected."

        # Sort by frequency — most common objects first
        sorted_labels = sorted(
            label_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Build summary sentence
        label_parts = [
            f"{label} ({count}/{result.total_frames} frames)"
            for label, count in sorted_labels[:5]   # top 5 objects
        ]

        return (
            f"Analysed {result.total_frames} frames over "
            f"{result.video_duration_sec:.1f}s. "
            f"Most common objects: {', '.join(label_parts)}."
        )
