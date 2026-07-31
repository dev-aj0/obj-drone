"""Computer vision input from the Raspberry Pi Camera Module V2."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera(ABC):
    """Abstract camera interface."""

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the latest BGR frame, or None if unavailable."""
        ...

    def frames(self) -> Generator[np.ndarray, None, None]:
        while True:
            frame = self.read()
            if frame is not None:
                yield frame


class PiCamera(Camera):
    """Raspberry Pi Camera Module V2 via picamera2."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._camera = None

    def start(self) -> None:
        from picamera2 import Picamera2

        self._camera = Picamera2()
        config = self._camera.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.fps},
        )
        self._camera.configure(config)
        self._camera.start()
        logger.info("Pi camera started at %dx%d @ %d fps", self.width, self.height, self.fps)

    def stop(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera = None

    def read(self) -> np.ndarray | None:
        if self._camera is None:
            return None
        rgb = self._camera.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class USBCamera(Camera):
    """Fallback USB or V4L2 camera for development off the Pi."""

    def __init__(self, device: int = 0, width: int = 640, height: int = 480) -> None:
        self.device = device
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info("USB camera %d opened at %dx%d", self.device, self.width, self.height)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None


def create_camera(
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    prefer_pi: bool = True,
) -> Camera:
    """Instantiate Pi camera on aarch64, USB camera elsewhere."""
    import platform

    if prefer_pi and platform.machine() == "aarch64":
        try:
            cam = PiCamera(width, height, fps)
            cam.start()
            return cam
        except Exception as exc:
            logger.warning("Pi camera unavailable (%s), falling back to USB", exc)
    cam = USBCamera(width=width, height=height)
    cam.start()
    return cam
