# backend/models/scene_graph.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Builds a structured scene graph from detection + caption results.
#   A scene graph answers: WHAT objects exist, WHERE they are, and
#   HOW they relate to each other spatially.
#
# WHY A SCENE GRAPH?
#   Raw detections give you: label="chair", box=[120,300,410,580]
#   A scene graph gives you: "chair is below the desk, left of the monitor"
#   This structured representation is what the RAG chatbot queries.
#
# OUTPUT STRUCTURE:
#   SceneGraph
#     ├── objects[]       → list of SceneObject (one per detection)
#     ├── relationships[] → spatial pairs e.g. "monitor is above keyboard"
#     └── summary         → one-sentence workspace description
#
# SPATIAL LOGIC:
#   Relationships are computed from bounding box centres:
#     - left/right : compare x centres
#     - above/below: compare y centres (y increases downward in image coords)
#     - near/far   : compare distance between centres vs image diagonal
# ─────────────────────────────────────────────────────────────────────────────


import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SceneObject:
    """
    Represents one detected object in the scene graph.

    Attributes:
        id         : Unique index e.g. 0, 1, 2 (position in objects list)
        label      : Object class name e.g. "monitor"
        confidence : Detection confidence 0.0–1.0
        box        : Bounding box [x1, y1, x2, y2] in pixels
        caption    : BLIP-generated description e.g. "a black monitor on a desk"
        mask_area  : Pixel count of SAM2 mask (0 if segmentation was skipped)
        center     : (cx, cy) computed from box — centre point in pixels
        region     : Coarse spatial region: "top-left", "center", "bottom-right" etc.
    """
    id         : int
    label      : str
    confidence : float
    box        : List[float]
    caption    : str                   = ""
    mask_area  : int                   = 0
    center     : tuple                 = field(default_factory=tuple)
    region     : str                   = ""


    def to_dict(self) -> dict:
        return {
            "id"         : self.id,
            "label"      : self.label,
            "confidence" : round(self.confidence, 4),
            "box"        : [round(v, 2) for v in self.box],
            "caption"    : self.caption,
            "mask_area"  : self.mask_area,
            "center"     : self.center,
            "region"     : self.region,
        }


@dataclass
class SpatialRelationship:
    """
    Represents a directional spatial relationship between two objects.

    Example:
        subject="monitor", relation="is above", target="keyboard"
        → readable as: "monitor is above keyboard"

    Attributes:
        subject_id  : id of the subject SceneObject
        subject     : label of the subject
        relation    : spatial relation string e.g. "is above", "is left of"
        target_id   : id of the target SceneObject
        target      : label of the target
    """
    subject_id : int
    subject    : str
    relation   : str
    target_id  : int
    target     : str


    def to_sentence(self) -> str:
        """Returns a human-readable sentence: 'monitor is above keyboard'."""
        return f"{self.subject} {self.relation} {self.target}"


    def to_dict(self) -> dict:
        return {
            "subject_id" : self.subject_id,
            "subject"    : self.subject,
            "relation"   : self.relation,
            "target_id"  : self.target_id,
            "target"     : self.target,
            "sentence"   : self.to_sentence(),
        }


@dataclass
class SceneGraph:
    """
    Complete structured representation of a workspace scene.

    Attributes:
        objects       : All detected SceneObjects
        relationships : All spatial SpatialRelationships between objects
        summary       : Auto-generated one-line workspace description
        image_size    : (width, height) of the analysed image
        object_count  : Total number of detected objects
    """
    objects       : List[SceneObject]          = field(default_factory=list)
    relationships : List[SpatialRelationship]  = field(default_factory=list)
    summary       : str                        = ""
    image_size    : tuple                      = field(default_factory=tuple)
    object_count  : int                        = 0


    def to_dict(self) -> dict:
        return {
            "object_count"  : self.object_count,
            "summary"       : self.summary,
            "image_size"    : self.image_size,
            "objects"       : [o.to_dict() for o in self.objects],
            "relationships" : [r.to_dict() for r in self.relationships],
        }


    def to_text(self) -> str:
        """
        Converts the scene graph to a plain text block for RAG context.
        This is the text that gets embedded and stored in ChromaDB.

        Format:
            Scene summary: ...
            Objects detected: chair, monitor, desk, keyboard
            Spatial relationships:
              - monitor is above keyboard
              - chair is left of desk
        """
        lines = []
        lines.append(f"Scene summary: {self.summary}")
        lines.append(f"Objects detected: {', '.join(o.label for o in self.objects)}")

        if self.relationships:
            lines.append("Spatial relationships:")
            for r in self.relationships:
                lines.append(f"  - {r.to_sentence()}")

        if self.objects:
            lines.append("Object details:")
            for o in self.objects:
                lines.append(
                    f"  - {o.label} (confidence: {o.confidence:.2f}, "
                    f"region: {o.region}, caption: {o.caption})"
                )

        return "\n".join(lines)


# ── SceneGraphBuilder ─────────────────────────────────────────────────────────

class SceneGraphBuilder:
    """
    Builds a SceneGraph from detection, segmentation, and caption results.

    Usage:
        builder = SceneGraphBuilder(image_size=(800, 534))
        graph   = builder.build(detections, seg_results, caption_results)
        print(graph.to_text())
    """

    # Distance threshold for "near" relationship
    # Objects within 20% of the image diagonal are considered "near"
    NEAR_THRESHOLD = 0.20

    def __init__(self, image_size: tuple):
        """
        Args:
            image_size: (width, height) of the image being analysed.
                        Used to compute relative spatial positions.
        """
        self.image_width  = image_size[0]
        self.image_height = image_size[1]
        # Image diagonal used to normalise distances
        self.diagonal = math.sqrt(self.image_width**2 + self.image_height**2)

        logger.info(f"[SceneGraphBuilder] Image size: {image_size}")


    # ──────────────────────────────────────────────────────────────────────────
    def build(
        self,
        detections      : list,   # List[Detection] from detector.py
        seg_results     : list,   # List[SegmentationResult] from segmenter.py (can be [])
        caption_results : list,   # List[CaptionResult] from captioner.py (can be [])
        scene_caption   : str = "",
    ) -> SceneGraph:
        """
        Builds a complete SceneGraph from pipeline outputs.

        All three input lists are optional — the builder degrades gracefully:
          - No seg_results   → mask_area stays 0
          - No caption_results → caption stays ""
          - No scene_caption → summary auto-generated from object labels

        Args:
            detections      : Detection objects from Detector
            seg_results     : SegmentationResult objects from Segmenter
            caption_results : CaptionResult objects from Captioner
            scene_caption   : Full-image caption from captioner.caption_scene()

        Returns:
            SceneGraph with objects, relationships, and summary filled in
        """

        if not detections:
            logger.warning("[SceneGraphBuilder] No detections — empty graph.")
            return SceneGraph(
                summary    = "No objects detected in the workspace.",
                image_size = (self.image_width, self.image_height),
            )

        # ── Step 1: Build lookup maps for seg and caption results ──────────
        # Map by index — seg_results[i] and caption_results[i] match detections[i]
        seg_map     = {i: r for i, r in enumerate(seg_results)}
        caption_map = {i: r for i, r in enumerate(caption_results)}

        # ── Step 2: Build SceneObject list ─────────────────────────────────
        objects = []
        for i, det in enumerate(detections):
            center = self._compute_center(det.box)
            region = self._compute_region(center)

            # Pull mask_area from seg result if available
            mask_area = seg_map[i].mask_area if i in seg_map else 0

            # Pull caption from caption result if available
            caption = caption_map[i].caption if i in caption_map else ""

            obj = SceneObject(
                id         = i,
                label      = det.label,
                confidence = det.confidence,
                box        = det.box,
                caption    = caption,
                mask_area  = mask_area,
                center     = center,
                region     = region,
            )
            objects.append(obj)

        # ── Step 3: Compute spatial relationships ──────────────────────────
        relationships = self._compute_relationships(objects)

        # ── Step 4: Build summary ──────────────────────────────────────────
        if scene_caption:
            summary = scene_caption
        else:
            # Auto-generate from detected labels if no BLIP scene caption
            labels  = [o.label for o in objects]
            summary = self._generate_summary(labels)

        # ── Step 5: Assemble final SceneGraph ──────────────────────────────
        graph = SceneGraph(
            objects       = objects,
            relationships = relationships,
            summary       = summary,
            image_size    = (self.image_width, self.image_height),
            object_count  = len(objects),
        )

        logger.info(
            f"[SceneGraphBuilder] Built graph: "
            f"{len(objects)} objects, {len(relationships)} relationships."
        )
        return graph


    # ──────────────────────────────────────────────────────────────────────────
    def _compute_center(self, box: List[float]) -> tuple:
        """
        Computes the centre point (cx, cy) of a bounding box.

        Args:
            box: [x1, y1, x2, y2] bounding box

        Returns:
            (cx, cy) tuple of floats
        """
        x1, y1, x2, y2 = box
        return (round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1))


    # ──────────────────────────────────────────────────────────────────────────
    def _compute_region(self, center: tuple) -> str:
        """
        Assigns a coarse spatial region label to an object based on its
        centre point relative to the image dimensions.

        Divides the image into a 3x3 grid:
            top-left    | top-center    | top-right
            middle-left | center        | middle-right
            bottom-left | bottom-center | bottom-right

        Args:
            center: (cx, cy) centre point

        Returns:
            Region label string e.g. "top-left", "center", "bottom-right"
        """
        cx, cy = center

        # Horizontal thirds
        if cx < self.image_width / 3:
            h_zone = "left"
        elif cx < 2 * self.image_width / 3:
            h_zone = "center"
        else:
            h_zone = "right"

        # Vertical thirds
        if cy < self.image_height / 3:
            v_zone = "top"
        elif cy < 2 * self.image_height / 3:
            v_zone = "middle"
        else:
            v_zone = "bottom"

        # Combine zones — "center"+"center" → just "center"
        if v_zone == "middle" and h_zone == "center":
            return "center"
        elif v_zone == "middle":
            return f"middle-{h_zone}"
        else:
            return f"{v_zone}-{h_zone}"


    # ──────────────────────────────────────────────────────────────────────────
    def _compute_relationships(
        self,
        objects: List[SceneObject],
    ) -> List[SpatialRelationship]:
        """
        Computes all pairwise spatial relationships between objects.

        For every pair (A, B), determines:
          - is A left of / right of B?
          - is A above / below B?
          - is A near B? (if within NEAR_THRESHOLD × diagonal)

        Only generates relationships where the spatial difference is
        meaningful (> 5% of image dimension) to avoid noise.

        Args:
            objects: List of SceneObjects with computed centers

        Returns:
            List of SpatialRelationship objects
        """
        relationships = []

        # Minimum pixel difference to call something "left of" or "above"
        # Avoids noise: objects almost side-by-side don't get left/right label
        min_h_diff = self.image_width  * 0.05   # 5% of image width
        min_v_diff = self.image_height * 0.05   # 5% of image height

        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects):
                if i >= j:
                    # Only process each pair once (i < j)
                    continue

                cx_a, cy_a = obj_a.center
                cx_b, cy_b = obj_b.center

                h_diff = cx_a - cx_b   # positive → A is right of B
                v_diff = cy_a - cy_b   # positive → A is below B (y increases down)

                # ── Horizontal relationship ────────────────────────────────
                if abs(h_diff) > min_h_diff:
                    if h_diff < 0:
                        # A is left of B
                        relationships.append(SpatialRelationship(
                            subject_id=obj_a.id, subject=obj_a.label,
                            relation="is left of",
                            target_id=obj_b.id,  target=obj_b.label,
                        ))
                    else:
                        # A is right of B
                        relationships.append(SpatialRelationship(
                            subject_id=obj_a.id, subject=obj_a.label,
                            relation="is right of",
                            target_id=obj_b.id,  target=obj_b.label,
                        ))

                # ── Vertical relationship ──────────────────────────────────
                if abs(v_diff) > min_v_diff:
                    if v_diff < 0:
                        # A is above B (smaller y = higher in image)
                        relationships.append(SpatialRelationship(
                            subject_id=obj_a.id, subject=obj_a.label,
                            relation="is above",
                            target_id=obj_b.id,  target=obj_b.label,
                        ))
                    else:
                        # A is below B
                        relationships.append(SpatialRelationship(
                            subject_id=obj_a.id, subject=obj_a.label,
                            relation="is below",
                            target_id=obj_b.id,  target=obj_b.label,
                        ))

                # ── Proximity relationship ─────────────────────────────────
                distance = math.sqrt(h_diff**2 + v_diff**2)
                if distance / self.diagonal < self.NEAR_THRESHOLD:
                    relationships.append(SpatialRelationship(
                        subject_id=obj_a.id, subject=obj_a.label,
                        relation="is near",
                        target_id=obj_b.id,  target=obj_b.label,
                    ))

        return relationships


    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _generate_summary(labels: List[str]) -> str:
        """
        Auto-generates a summary sentence from detected object labels.
        Used when no BLIP scene caption is available.

        Examples:
            ["monitor", "keyboard"] → "Workspace contains: monitor, keyboard."
            []                      → "No objects detected."

        Args:
            labels: List of object label strings

        Returns:
            Summary sentence
        """
        if not labels:
            return "No objects detected in the workspace."

        unique = list(dict.fromkeys(labels))   # deduplicate, preserve order

        if len(unique) == 1:
            return f"Workspace contains a {unique[0]}."
        elif len(unique) == 2:
            return f"Workspace contains a {unique[0]} and a {unique[1]}."
        else:
            joined = ", ".join(unique[:-1])
            return f"Workspace contains: {joined}, and {unique[-1]}."
