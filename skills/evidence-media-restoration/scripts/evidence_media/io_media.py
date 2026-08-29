from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .core import FrameRecord, read_image, sha256_file


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def inspect_source(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path.resolve())}
    if path.is_file():
        info.update({"bytes": path.stat().st_size, "sha256": sha256_file(path), "suffix": path.suffix.lower()})
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        info.update({"kind": "image_sequence", "image_count": len(files)})
        return info
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = read_image(path)
        info.update({"kind": "image", "width": image.shape[1], "height": image.shape[0]})
        return info
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open media: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).rstrip("\x00")
    info.update({
        "kind": "video",
        "backend": "opencv",
        "codec_fourcc": codec,
        "fps_reported": fps,
        "frame_count_reported": count,
        "duration_reported_seconds": count / fps if fps > 0 else None,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "exact_pts_available": False,
    })
    cap.release()
    try:
        import av  # type: ignore
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            context = stream.codec_context
            try:
                gop_size = int(context.gop_size or 0)
            except (AttributeError, RuntimeError, ValueError):
                gop_size = None
            try:
                has_b_frames = bool(context.has_b_frames)
            except (AttributeError, RuntimeError, ValueError):
                has_b_frames = None
            info.update({
                "pyav_available": True,
                "codec_name": context.name,
                "codec_long_name": context.codec.long_name,
                "time_base": str(stream.time_base),
                "average_rate": str(stream.average_rate),
                "gop_size": gop_size,
                "has_b_frames": has_b_frames,
                "exact_pts_available": True,
            })
    except Exception as exc:
        info.update({"pyav_available": False, "pyav_note": str(exc)})
    return info


def decode_frames(path: Path, start: float | None, end: float | None, stride: int = 1, prefer_pyav: bool = True) -> tuple[list[FrameRecord], dict[str, Any]]:
    if path.is_dir() or path.suffix.lower() in IMAGE_SUFFIXES:
        files = sorted(p for p in (path.iterdir() if path.is_dir() else [path]) if p.suffix.lower() in IMAGE_SUFFIXES)
        records = []
        for i, p in enumerate(files):
            if i % max(1, stride):
                continue
            image = read_image(p)
            records.append(FrameRecord(i, image, source_name=p.name, backend="image-sequence"))
        return records, inspect_source(path)
    if prefer_pyav:
        try:
            return _decode_pyav(path, start, end, stride)
        except Exception:
            pass
    return _decode_opencv(path, start, end, stride)


def _decode_pyav(path: Path, start: float | None, end: float | None, stride: int) -> tuple[list[FrameRecord], dict[str, Any]]:
    import av  # type: ignore
    records: list[FrameRecord] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if start and stream.time_base:
            container.seek(int(start / float(stream.time_base)), stream=stream, any_frame=False, backward=True)
        ordinal = 0
        accepted = 0
        for frame in container.decode(stream):
            t = float(frame.time) if frame.time is not None else None
            if start is not None and t is not None and t < start:
                ordinal += 1
                continue
            if end is not None and t is not None and t >= end:
                break
            if accepted % max(1, stride) == 0:
                records.append(FrameRecord(
                    ordinal=ordinal,
                    image=frame.to_ndarray(format="bgr24"),
                    time_seconds=t,
                    pts=int(frame.pts) if frame.pts is not None else None,
                    key_frame=bool(frame.key_frame),
                    pict_type=str(frame.pict_type) if frame.pict_type is not None else None,
                    backend="pyav",
                ))
            ordinal += 1
            accepted += 1
    info = inspect_source(path)
    info["decode_backend"] = "pyav"
    info["decoded_count"] = len(records)
    return records, info


def _decode_opencv(path: Path, start: float | None, end: float | None, stride: int) -> tuple[list[FrameRecord], dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    start_frame = max(0, int((start or 0.0) * fps))
    end_frame = int(end * fps) if end is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    records: list[FrameRecord] = []
    idx = start_frame
    while idx < end_frame:
        ok, image = cap.read()
        if not ok:
            break
        if (idx - start_frame) % max(1, stride) == 0:
            records.append(FrameRecord(idx, image, time_seconds=idx / fps, backend="opencv"))
        idx += 1
    cap.release()
    info = inspect_source(path)
    info["decode_backend"] = "opencv"
    info["decoded_count"] = len(records)
    info["decode_warning"] = "PTS/frame type are not available; timestamps are FPS-derived."
    return records, info
