"""Computer vision input from the Raspberry Pi Camera Module V2 (Sony IMX219)."""

from __future__ import annotations

import logging
import platform
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Camera could not be opened or configured."""


class Camera(ABC):
    """Abstract camera interface. All implementations return BGR frames."""

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


# libcamera AfMode values. Hard-coded so the module still imports on machines
# without libcamera; the real enum is used when it is importable.
AF_MANUAL, AF_AUTO, AF_CONTINUOUS = 0, 1, 2
AF_STATE_NAMES = {0: "idle", 1: "scanning", 2: "focused", 3: "failed"}
FOCUS_MODES = ("continuous", "auto", "manual")


class PiCamera(Camera):
    """CSI camera (IMX219, IMX519, …) via picamera2.

    Frames are returned in OpenCV's native BGR order.

    Autofocus modules default to AfMode=Manual with the lens parked at 1.0
    dioptre (≈1 m), so everything at another distance looks soft unless focus
    is configured explicitly.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        focus_mode: str = "continuous",
        lens_position: float = 0.0,
        tuning_file: str | Path | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.focus_mode = (focus_mode or "continuous").lower()
        self.lens_position = float(lens_position)
        self.tuning_file = Path(tuning_file) if tuning_file else None
        self._camera = None
        self.focus_applied: str | None = None

        if self.focus_mode not in FOCUS_MODES:
            raise ValueError(
                f"Unknown focus_mode {focus_mode!r} (expected one of {FOCUS_MODES})"
            )

    def start(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:  # pragma: no cover - depends on host
            raise CameraError(
                "picamera2 is not importable. On Raspberry Pi OS install it with "
                "'sudo apt install python3-picamera2 python3-libcamera' and create the "
                "venv with 'python3 -m venv --system-site-packages .venv'. "
                "picamera2 cannot be pip-installed into an isolated venv because "
                f"libcamera has no PyPI wheel. ({exc})"
            ) from exc

        # A custom tuning file is how autofocus gets enabled on sensors whose
        # stock Raspberry Pi OS tuning has no rpi.af block (e.g. the IMX519).
        # Note the LIBCAMERA_RPI_TUNING_FILE env var does NOT work here —
        # picamera2 passes its own tuning through, overriding it.
        tuning = None
        if self.tuning_file is not None:
            if self.tuning_file.is_file():
                tuning = Picamera2.load_tuning_file(
                    self.tuning_file.name, dir=str(self.tuning_file.parent)
                )
                logger.info("Using camera tuning file %s", self.tuning_file)
            else:
                # Not fatal: the camera works fine on the stock tuning, it just
                # may not have autofocus. Failing here would ground the aircraft
                # over a focus file.
                logger.warning(
                    "Tuning file %s not found — using the stock tuning. If this "
                    "camera needs it for autofocus, run scripts/enable_autofocus.sh",
                    self.tuning_file,
                )

        try:
            self._camera = Picamera2(tuning=tuning) if tuning else Picamera2()
        except Exception as exc:  # pragma: no cover - depends on hardware
            raise CameraError(
                f"Could not open the CSI camera ({exc}). Check the ribbon cable seating "
                "and that 'rpicam-hello --list-cameras' detects it."
            ) from exc

        frame_us = int(1_000_000 / self.fps) if self.fps > 0 else 33333
        # picamera2's "RGB888" is, confusingly, stored B,G,R in memory — i.e. exactly
        # OpenCV's BGR layout. Do NOT colour-convert the result of capture_array().
        # queue=False makes capture_array() return a freshly captured frame instead of
        # a stale queued one, which matters for closed-loop control latency.
        config = self._camera.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_us, frame_us)},
            buffer_count=4,
            queue=False,
        )
        self._camera.configure(config)
        self._camera.start()

        actual = self._camera.camera_configuration()["main"]["size"]
        if tuple(actual) != (self.width, self.height):
            logger.warning(
                "Camera returned %dx%d instead of the requested %dx%d",
                actual[0],
                actual[1],
                self.width,
                self.height,
            )
            self.width, self.height = int(actual[0]), int(actual[1])
        self._configure_focus()
        logger.info(
            "Camera started at %dx%d @ %d fps (BGR, focus=%s)",
            self.width,
            self.height,
            self.fps,
            self.focus_applied or "fixed",
        )

    def _configure_focus(self) -> None:
        """Apply the requested focus mode, if the module has a movable lens."""
        assert self._camera is not None
        if "AfMode" not in self._camera.camera_controls:
            if self.focus_mode != "manual":
                logger.info(
                    "Fixed-focus camera — focus_mode=%s ignored", self.focus_mode
                )
            self.focus_applied = "fixed"
            return

        try:
            from libcamera import controls as libcontrols

            manual = libcontrols.AfModeEnum.Manual
            auto = libcontrols.AfModeEnum.Auto
            continuous = libcontrols.AfModeEnum.Continuous
        except Exception:  # pragma: no cover - depends on host
            manual, auto, continuous = AF_MANUAL, AF_AUTO, AF_CONTINUOUS

        try:
            if self.focus_mode == "manual":
                self._camera.set_controls(
                    {"AfMode": manual, "LensPosition": self.lens_position}
                )
                distance = (
                    "infinity" if self.lens_position <= 0 else f"{1.0 / self.lens_position:.2f} m"
                )
                self.focus_applied = f"manual @ {self.lens_position:.2f} dioptre ({distance})"
            elif self.focus_mode == "auto":
                self._camera.set_controls({"AfMode": auto})
                # One-shot: focus now, then hold — no hunting mid-flight.
                if hasattr(self._camera, "autofocus_cycle"):
                    self._camera.autofocus_cycle()
                self.focus_applied = "auto (one-shot)"
            else:
                self._camera.set_controls({"AfMode": continuous})
                self.focus_applied = "continuous"
        except Exception as exc:
            logger.warning("Could not set focus mode %s: %s", self.focus_mode, exc)
            self.focus_applied = "unset"

    def focus_info(self) -> dict[str, float | str | None]:
        """Live focus telemetry: lens position, AF state, and sharpness figure.

        FocusFoM is libcamera's contrast figure-of-merit. A value that stays
        near zero across the whole lens range means the scene has no detail to
        focus on, not that the lens is broken.
        """
        if self._camera is None:
            return {}
        try:
            meta = self._camera.capture_metadata()
        except Exception:
            return {}
        state = meta.get("AfState")
        return {
            "lens_position": meta.get("LensPosition"),
            "af_state": AF_STATE_NAMES.get(state, state),
            "focus_fom": meta.get("FocusFoM"),
        }

    def set_lens_position(self, dioptres: float) -> None:
        """Move the lens manually. 0.0 = infinity, higher = closer."""
        if self._camera is None:
            return
        if "LensPosition" not in self._camera.camera_controls:
            logger.info("Fixed-focus camera — cannot set lens position")
            return
        try:
            from libcamera import controls as libcontrols

            manual = libcontrols.AfModeEnum.Manual
        except Exception:  # pragma: no cover - depends on host
            manual = AF_MANUAL
        self._camera.set_controls({"AfMode": manual, "LensPosition": float(dioptres)})
        self.lens_position = float(dioptres)
        self.focus_applied = f"manual @ {dioptres:.2f} dioptre"

    def stop(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop()
                self._camera.close()
            except Exception:  # pragma: no cover - best effort on shutdown
                logger.debug("Error closing Pi camera", exc_info=True)
            self._camera = None

    def read(self) -> np.ndarray | None:
        if self._camera is None:
            return None
        # Already BGR — see the format note in start().
        return self._camera.capture_array()


class USBCamera(Camera):
    """Fallback USB / V4L2 camera for development off the Pi."""

    def __init__(
        self, device: int = 0, width: int = 640, height: int = 480, fps: int = 30
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"Could not open video device {self.device}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Keep the grab queue short so reads return current frames.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
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


def picamera2_available() -> bool:
    """True if picamera2 can be imported in this interpreter."""
    try:
        import picamera2  # noqa: F401
    except Exception:
        return False
    return True


def create_camera(
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    backend: str = "auto",
    focus_mode: str = "continuous",
    lens_position: float = 0.0,
    tuning_file: str | Path | None = None,
) -> Camera:
    """Create a camera.

    backend:
      "picamera2" — require the CSI camera; raise CameraError if unavailable.
      "usb"       — force the V4L2/USB path.
      "auto"      — use the CSI camera on aarch64 when picamera2 imports,
                    otherwise fall back to USB.

    "auto" is convenient for development but will silently fly on the wrong
    camera if the CSI stack is broken. Set backend: picamera2 on the aircraft.

    focus_mode / lens_position are ignored by fixed-focus and USB cameras.
    """
    backend = (backend or "auto").lower()

    if backend == "picamera2":
        cam = PiCamera(width, height, fps, focus_mode, lens_position, tuning_file)
        cam.start()
        return cam

    if backend == "usb":
        cam = USBCamera(width=width, height=height, fps=fps)
        cam.start()
        return cam

    if backend != "auto":
        raise ValueError(f"Unknown camera backend {backend!r}")

    if platform.machine() == "aarch64" and picamera2_available():
        try:
            cam = PiCamera(width, height, fps, focus_mode, lens_position, tuning_file)
            cam.start()
            return cam
        except CameraError as exc:
            logger.warning("CSI camera unavailable (%s) — falling back to USB", exc)

    cam = USBCamera(width=width, height=height, fps=fps)
    cam.start()
    return cam
