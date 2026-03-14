# backend/mlops/tracker.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   MLflow wrapper for logging WorkspaceVision inference metrics.
#   Every pipeline run logs: latency, object count, confidence scores,
#   model versions, and device info as a tracked MLflow run.
#
# WHY MLFLOW?
#   Without tracking, you have no idea if model performance degrades over time.
#   MLflow stores every inference run so you can:
#     - Compare latency across model versions
#     - Track detection confidence distributions
#     - Spot regressions when you swap models
#     - Export run history for reporting
#
# HOW IT WORKS:
#   1. Call tracker.start_run() before pipeline
#   2. Call tracker.log_inference() after pipeline completes
#   3. Call tracker.end_run() when done
#   OR use the context manager: with tracker.run_context():
#
# MLFLOW UI:
#   mlflow ui --backend-store-uri ./mlops/tracking
#   Open: http://localhost:5000
# ─────────────────────────────────────────────────────────────────────────────


import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
import sys

import mlflow
import mlflow.pytorch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings


logger = logging.getLogger(__name__)


class InferenceTracker:
    """
    MLflow-based inference tracker for WorkspaceVision.

    Logs per-request metrics to MLflow so you can monitor
    model performance over time via the MLflow UI.

    Usage — context manager (recommended):
        tracker = InferenceTracker()
        with tracker.run_context(source="api"):
            result = pipeline.analyze(image)
            tracker.log_inference(result)

    Usage — manual:
        tracker.start_run()
        result = pipeline.analyze(image)
        tracker.log_inference(result)
        tracker.end_run()
    """

    def __init__(self):
        self.settings    = get_settings()
        self._run_active = False

        # ── Set up MLflow tracking URI ─────────────────────────────────────
        # This tells MLflow WHERE to store run data (local folder)
        tracking_uri = self.settings.MLFLOW_TRACKING_URI
        Path(tracking_uri).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)

        # ── Set experiment ─────────────────────────────────────────────────
        # Experiment = named group of runs. All WorkspaceVision runs go here.
        mlflow.set_experiment(self.settings.MLFLOW_EXPERIMENT_NAME)

        logger.info(
            f"[Tracker] MLflow initialised. "
            f"URI: {tracking_uri} | "
            f"Experiment: {self.settings.MLFLOW_EXPERIMENT_NAME}"
        )


    # ──────────────────────────────────────────────────────────────────────────
    def start_run(self, run_name: Optional[str] = None):
        """
        Starts a new MLflow run.
        Each run = one inference request.

        Args:
            run_name: Optional label e.g. "image_analysis" or "video_frame_42"
        """
        if self._run_active:
            logger.warning("[Tracker] Run already active — ending previous run first.")
            self.end_run()

        mlflow.start_run(run_name=run_name or f"inference_{int(time.time())}")
        self._run_active = True
        logger.debug("[Tracker] MLflow run started.")


    # ──────────────────────────────────────────────────────────────────────────
    def end_run(self):
        """Ends the current MLflow run."""
        if self._run_active:
            mlflow.end_run()
            self._run_active = False
            logger.debug("[Tracker] MLflow run ended.")


    # ──────────────────────────────────────────────────────────────────────────
    @contextmanager
    def run_context(self, run_name: Optional[str] = None):
        """
        Context manager for automatic run start/end.
        Ensures run is always closed even if an exception occurs.

        Usage:
            with tracker.run_context(run_name="image_analysis"):
                result = pipeline.analyze(image)
                tracker.log_inference(result)
        """
        self.start_run(run_name=run_name)
        try:
            yield self
        finally:
            self.end_run()


    # ──────────────────────────────────────────────────────────────────────────
    def log_inference(self, result, source: str = "unknown"):
        """
        Logs all metrics from one AnalysisResult to MLflow.

        Logs the following:
          PARAMS (fixed config values):
            - model_dino, model_sam2, model_blip
            - device, dtype, source

          METRICS (numeric values for charting):
            - total_time         : end-to-end latency in seconds
            - detection_time     : DINO inference time
            - segmentation_time  : SAM2 inference time
            - captioning_time    : BLIP inference time
            - object_count       : number of detected objects
            - avg_confidence     : mean detection confidence
            - min_confidence     : lowest detection confidence
            - max_confidence     : highest detection confidence

        Args:
            result : AnalysisResult from PerceptionPipeline.analyze()
            source : Label for where this came from — "api", "video", "test"
        """
        if not self._run_active:
            logger.warning("[Tracker] No active run — call start_run() first.")
            return

        try:
            # ── Log model identifiers as params ───────────────────────────
            # Params are strings — logged once per run, not per step
            mlflow.log_params({
                "model_dino"  : self.settings.GROUNDING_DINO_MODEL,
                "model_sam2"  : self.settings.SAM2_MODEL,
                "model_blip"  : self.settings.BLIP_MODEL,
                "device"      : self.settings.DEVICE,
                "dtype"       : self.settings.MODEL_PRECISION,
                "source"      : source,
            })

            # ── Log timing metrics ────────────────────────────────────────
            # Metrics are floats — can be charted over time in MLflow UI
            timings = result.timings or {}
            mlflow.log_metrics({
                "total_time"        : result.total_time,
                "detection_time"    : timings.get("detection",    0.0),
                "segmentation_time" : timings.get("segmentation", 0.0),
                "captioning_time"   : timings.get("captioning",   0.0),
                "prepare_time"      : timings.get("prepare_image",0.0),
            })

            # ── Log detection metrics ─────────────────────────────────────
            detections = result.detections or []
            object_count = len(detections)
            mlflow.log_metric("object_count", object_count)

            if detections:
                confidences = [d.confidence for d in detections]
                mlflow.log_metrics({
                    "avg_confidence" : sum(confidences) / len(confidences),
                    "min_confidence" : min(confidences),
                    "max_confidence" : max(confidences),
                })

                # Log each detected label as a tag (searchable in MLflow UI)
                labels = list({d.label for d in detections})
                mlflow.set_tag("detected_labels", ", ".join(sorted(labels)))

            # ── Log image metadata ────────────────────────────────────────
            if result.image_meta:
                orig = result.image_meta.get("original_size", (0, 0))
                final= result.image_meta.get("final_size",    (0, 0))
                mlflow.log_params({
                    "image_original_size": f"{orig[0]}x{orig[1]}",
                    "image_final_size"   : f"{final[0]}x{final[1]}",
                })

            # ── Log error tag if pipeline had issues ──────────────────────
            if result.error:
                mlflow.set_tag("error", result.error[:250])

            logger.info(
                f"[Tracker] Logged: {object_count} objects | "
                f"{result.total_time:.2f}s total"
            )

        except Exception as e:
            logger.error(f"[Tracker] Failed to log metrics: {e}")


    # ──────────────────────────────────────────────────────────────────────────
    def log_chat(
        self,
        question    : str,
        response_obj: any,
        source      : str = "api",
    ):
        """
        Logs chatbot interaction metrics to MLflow.

        Args:
            question     : User's question string
            response_obj : ChatResponse from chatbot.chat()
            source       : "api" or "test"
        """
        if not self._run_active:
            logger.warning("[Tracker] No active run — call start_run() first.")
            return

        try:
            token_usage = response_obj.token_usage or {}
            mlflow.log_metrics({
                "prompt_tokens"     : token_usage.get("prompt_tokens",     0),
                "completion_tokens" : token_usage.get("completion_tokens", 0),
                "total_tokens"      : token_usage.get("total_tokens",      0),
            })
            mlflow.log_params({
                "chat_model"   : self.settings.OPENAI_CHAT_MODEL,
                "rag_used"     : str(response_obj.rag_used),
                "scene_used"   : str(response_obj.scene_used),
                "source"       : source,
            })
            mlflow.set_tag("question_preview", question[:100])

        except Exception as e:
            logger.error(f"[Tracker] Failed to log chat metrics: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_tracker_instance: Optional[InferenceTracker] = None


def get_tracker() -> InferenceTracker:
    """Returns singleton InferenceTracker. Creates on first call."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = InferenceTracker()
    return _tracker_instance
