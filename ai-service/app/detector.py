"""BISINDO (Indonesian sign language) detection with a YOLO11 model exported to ONNX.

The network is an object detector trained by Syizuril
(https://huggingface.co/Syizuril/YOLO11-BISINDO-Detection) and converted to a 480x480
int8 ONNX graph by ``export_yolo.py``. It predicts one letter per frame and returns
"No detection" by itself when no hand is visible, so no separate hand gate is needed.
"""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Kept verbatim because it is part of the public API contract of the original service.
NO_DETECTION = "Tidak Terdeteksi"

# Head output layout per anchor: centre x, centre y, width, height, then one score per class.
_BOX_COLUMNS = 4

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8"
_JPEG_START_OF_FRAME = range(0xC0, 0xD0)
_JPEG_NON_FRAME_MARKERS = (0xC4, 0xC8, 0xCC)  # DHT, JPG, DAC share the marker range


def image_size(data: bytes) -> tuple[int, int] | None:
    """Read (width, height) from JPEG/PNG headers without decoding the pixels.

    Decoding happens before any size check otherwise, so a small crafted file could
    expand into a multi-gigabyte bitmap. Unsupported formats return ``None``.
    """
    if data.startswith(_PNG_SIGNATURE):
        if len(data) < 24:
            return None
        return struct.unpack(">II", data[16:24])

    if not data.startswith(_JPEG_SIGNATURE):
        return None

    index = 2
    while index + 9 <= len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker == 0xFF:  # fill byte
            index += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:  # standalone markers
            index += 2
            continue

        segment_length = int.from_bytes(data[index + 2 : index + 4], "big")
        if marker in _JPEG_START_OF_FRAME and marker not in _JPEG_NON_FRAME_MARKERS:
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            return width, height
        if segment_length < 2:
            return None
        index += 2 + segment_length

    return None


def _to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (cx, cy, w, h) boxes to (x1, y1, x2, y2)."""
    half = boxes[:, 2:4] / 2
    return np.hstack([boxes[:, 0:2] - half, boxes[:, 0:2] + half])


def _intersection_over_union(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU between one xyxy box and an array of xyxy boxes."""
    top_left = np.maximum(box[:2], others[:, :2])
    bottom_right = np.minimum(box[2:], others[:, 2:])
    overlap = np.clip(bottom_right - top_left, 0, None).prod(axis=1)
    area_box = np.clip(box[2:] - box[:2], 0, None).prod()
    area_others = np.clip(others[:, 2:] - others[:, :2], 0, None).prod(axis=1)
    return overlap / np.maximum(area_box + area_others - overlap, np.finfo(np.float32).eps)


def _non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS, returning kept indices ordered by descending score."""
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        current = order[0]
        kept.append(int(current))
        if order.size == 1:
            break
        order = order[1:][
            _intersection_over_union(boxes[current], boxes[order[1:]]) <= iou_threshold
        ]
    return kept


class BisindoDetector:
    """Thin, thread-safe wrapper around a single ONNX Runtime session."""

    def __init__(
        self,
        model_path: Path,
        input_size: int = 480,
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        max_image_pixels: int = 8_000_000,
    ) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_image_pixels = max_image_pixels
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, frame: bytes) -> str:
        """Return the detected letter for an encoded image frame, or ``NO_DETECTION``."""
        if not frame:
            return NO_DETECTION

        size = image_size(frame)
        if size is None or size[0] * size[1] > self.max_image_pixels:
            return NO_DETECTION

        image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return NO_DETECTION

        resized = cv2.resize(image, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch = rgb.transpose(2, 0, 1)[np.newaxis, ...]

        # Shape (1, 4 + num_classes, anchors); transpose to one row per anchor.
        raw = self._session.run(None, {self._input_name: batch})[0][0].T
        return self._best_letter(raw)

    def _best_letter(self, raw: np.ndarray) -> str:
        class_scores = raw[:, _BOX_COLUMNS:]
        classes = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), classes]

        candidates = scores >= self.confidence_threshold
        if not candidates.any():
            return NO_DETECTION

        boxes = _to_xyxy(raw[candidates, :_BOX_COLUMNS])
        scores, classes = scores[candidates], classes[candidates]
        best = _non_max_suppression(boxes, scores, self.iou_threshold)[0]
        return LETTERS[int(classes[best])]
