#!/usr/bin/env python3
"""Stream an MP4 through adaptive YOLO detection and create an annotated MP4."""
# python3 detect-video.py input.mp4 output.mp4

from __future__ import annotations
import time
import argparse
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO


# YOLO_MODEL = "./trained-weights/best-drone-nano.pt"
YOLO_MODEL = "./drone-detection-weights.pt"
MIN_FRAME_ANNOTATE = 15
MAX_POSITION_CHANGE = 0.05
MAX_SIZE_CHANGE = 0.20
CONSOLODATE_BOUNDING_BOX_OVERLAP = 0.10

BoundingBox = tuple[float, float, float, float]
time_start = time.time()

def bounding_box_iou(first_box: BoundingBox, second_box: BoundingBox) -> float:
    """Return the intersection-over-union overlap for two bounding boxes."""
    intersection_width = max(
        0.0, min(first_box[2], second_box[2]) - max(first_box[0], second_box[0])
    )
    intersection_height = max(
        0.0, min(first_box[3], second_box[3]) - max(first_box[1], second_box[1])
    )
    intersection_area = intersection_width * intersection_height
    first_area = max(0.0, first_box[2] - first_box[0]) * max(
        0.0, first_box[3] - first_box[1]
    )
    second_area = max(0.0, second_box[2] - second_box[0]) * max(
        0.0, second_box[3] - second_box[1]
    )
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def union_bounding_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    """Return the rectangular union of one or more bounding boxes."""
    if not boxes:
        raise ValueError("At least one bounding box is required for a union")

    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def consolidate_bounding_boxes(boxes: list[BoundingBox]) -> list[BoundingBox]:
    """Replace each overlapping group with its rectangular union."""
    consolidated = list(boxes)

    while len(consolidated) > 1:
        parents = list(range(len(consolidated)))

        def find(box_index: int) -> int:
            while parents[box_index] != box_index:
                parents[box_index] = parents[parents[box_index]]
                box_index = parents[box_index]
            return box_index

        merged_any = False
        for first_index in range(len(consolidated)):
            for second_index in range(first_index + 1, len(consolidated)):
                if (
                    bounding_box_iou(
                        consolidated[first_index], consolidated[second_index]
                    )
                    <= CONSOLODATE_BOUNDING_BOX_OVERLAP
                ):
                    continue

                first_parent = find(first_index)
                second_parent = find(second_index)
                if first_parent != second_parent:
                    parents[second_parent] = first_parent
                    merged_any = True

        if not merged_any:
            break

        groups: dict[int, list[BoundingBox]] = {}
        for box_index, box in enumerate(consolidated):
            groups.setdefault(find(box_index), []).append(box)

        consolidated = [union_bounding_boxes(group) for group in groups.values()]

    return consolidated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw red YOLO detection boxes on every frame of an MP4."
    )
    parser.add_argument("input_video", type=Path, help="path to the input MP4 file")
    parser.add_argument("output_video", type=Path, help="path for the annotated MP4 file")
    return parser.parse_args()


def get_video_info(input_video: Path) -> tuple[str, int, int]:
    """Return the frame rate and decoded frame dimensions."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=avg_frame_rate,width,height:"
                "stream_tags=rotate:stream_side_data=rotation"
            ),
            "-of",
            "json",
            str(input_video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {input_video}")

    stream = streams[0]
    frame_rate = stream.get("avg_frame_rate", "")

    try:
        if not frame_rate or Fraction(frame_rate) <= 0:
            raise ValueError
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"Could not determine frame rate for {input_video}") from error

    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not determine frame size for {input_video}")

    rotation = stream.get("tags", {}).get("rotate", 0)
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = side_data["rotation"]
            break
    if abs(int(float(rotation))) % 180 == 90:
        width, height = height, width

    return frame_rate, width, height


def annotate_frames(
    frames: list[Any],
    model: Any,
    first_frame_boxes: list[BoundingBox] | None = None,
) -> tuple[list[Any], list[BoundingBox], int]:
    """Adaptively detect objects and return annotated in-memory frames.

    YOLO initially runs every ``MIN_FRAME_ANNOTATE`` frames. When two sampled
    frames have similar boxes, the boxes between them are interpolated. When
    their positions or sizes differ considerably, YOLO runs on the midpoint
    and the two resulting ranges are checked recursively.
    """
    from PIL import ImageDraw

    if not frames:
        return [], [], 0
    if MIN_FRAME_ANNOTATE < 1:
        raise ValueError("MIN_FRAME_ANNOTATE must be at least 1")

    frame_size = frames[0].size

    detected_boxes: dict[int, list[BoundingBox]] = {}
    boxes_for_frame: dict[int, list[BoundingBox]] = {}
    if first_frame_boxes is not None:
        boxes_for_frame[0] = first_frame_boxes

    def detect(frame_index: int) -> list[BoundingBox]:
        """Return cached boxes, running YOLO only when no boxes exist yet."""
        # Interpolated frames are stored in boxes_for_frame. Checking this
        # cache first guarantees that YOLO never runs for those frames.
        if frame_index in boxes_for_frame:
            return boxes_for_frame[frame_index]

        result = model.predict(
            source=frames[frame_index], 
            imgsz=1280,
            verbose=False,
        )[0]
        boxes = []
        if result.boxes is not None:
            boxes = consolidate_bounding_boxes(
                [tuple(box) for box in result.boxes.xyxy.cpu().tolist()]
            )
        detected_boxes[frame_index] = boxes
        boxes_for_frame[frame_index] = boxes
        return boxes

    def match_similar_boxes(
        start_boxes: list[BoundingBox],
        end_boxes: list[BoundingBox],
    ) -> list[tuple[BoundingBox, BoundingBox]] | None:
        """Match boxes by center and return matches only when all are similar."""
        if len(start_boxes) != len(end_boxes):
            return None
        if not start_boxes:
            return []

        frame_width, frame_height = frame_size
        frame_diagonal = max((frame_width**2 + frame_height**2) ** 0.5, 1.0)
        remaining_end_boxes = list(end_boxes)
        matches: list[tuple[BoundingBox, BoundingBox]] = []

        for start_box in start_boxes:
            start_x1, start_y1, start_x2, start_y2 = start_box
            start_center = ((start_x1 + start_x2) / 2, (start_y1 + start_y2) / 2)

            end_box = min(
                remaining_end_boxes,
                key=lambda box: (
                    ((box[0] + box[2]) / 2 - start_center[0]) ** 2
                    + ((box[1] + box[3]) / 2 - start_center[1]) ** 2
                ),
            )
            remaining_end_boxes.remove(end_box)

            end_x1, end_y1, end_x2, end_y2 = end_box
            end_center = ((end_x1 + end_x2) / 2, (end_y1 + end_y2) / 2)
            center_change = (
                (end_center[0] - start_center[0]) ** 2
                + (end_center[1] - start_center[1]) ** 2
            ) ** 0.5 / frame_diagonal

            start_width = max(start_x2 - start_x1, 1.0)
            start_height = max(start_y2 - start_y1, 1.0)
            end_width = max(end_x2 - end_x1, 1.0)
            end_height = max(end_y2 - end_y1, 1.0)
            width_change = abs(end_width - start_width) / max(
                start_width, end_width
            )
            height_change = abs(end_height - start_height) / max(
                start_height, end_height
            )

            if (
                center_change > MAX_POSITION_CHANGE
                or width_change > MAX_SIZE_CHANGE
                or height_change > MAX_SIZE_CHANGE
            ):
                return None

            matches.append((start_box, end_box))

        return matches

    def interpolate_boxes(
        matches: list[tuple[BoundingBox, BoundingBox]], ratio: float
    ) -> list[BoundingBox]:
        return [
            tuple(
                start_coordinate
                + (end_coordinate - start_coordinate) * ratio
                for start_coordinate, end_coordinate in zip(start_box, end_box)
            )
            for start_box, end_box in matches
        ]

    def annotate_range(start_index: int, end_index: int) -> None:
        """Recursively sample a range or interpolate its inner boxes."""
        start_boxes = detect(start_index)
        end_boxes = detect(end_index)
        matches = match_similar_boxes(start_boxes, end_boxes)

        if matches is not None:
            frame_count = end_index - start_index
            for frame_index in range(start_index + 1, end_index):
                ratio = (frame_index - start_index) / frame_count
                boxes_for_frame[frame_index] = interpolate_boxes(matches, ratio)
            return

        if end_index - start_index <= 1:
            return

        middle_index = (start_index + end_index) // 2
        detect(middle_index)
        annotate_range(start_index, middle_index)
        annotate_range(middle_index, end_index)

    sample_indices = list(range(0, len(frames), MIN_FRAME_ANNOTATE))
    if sample_indices[-1] != len(frames) - 1:
        sample_indices.append(len(frames) - 1)

    for frame_index in sample_indices:
        detect(frame_index)
    for start_index, end_index in zip(sample_indices, sample_indices[1:]):
        annotate_range(start_index, end_index)

    annotated_frames = []
    for frame_index, frame in enumerate(frames):
        image = frame.copy()
        draw = ImageDraw.Draw(image)
        line_width = max(2, round(min(image.size) * 0.005))

        for x1, y1, x2, y2 in boxes_for_frame[frame_index]:
            draw.rectangle(
                (round(x1), round(y1), round(x2), round(y2)),
                outline="red",
                width=line_width,
            )

        annotated_frames.append(image)

    return annotated_frames, boxes_for_frame[len(frames) - 1], len(detected_boxes)


def read_raw_frame(stream: BinaryIO, frame_byte_count: int) -> bytes | None:
    """Read exactly one raw frame, returning None at a clean end of stream."""
    frame_data = bytearray()
    while len(frame_data) < frame_byte_count:
        chunk = stream.read(frame_byte_count - len(frame_data))
        if not chunk:
            if frame_data:
                raise RuntimeError("FFmpeg returned an incomplete raw video frame")
            return None
        frame_data.extend(chunk)
    return bytes(frame_data)


def stream_video(
    input_video: Path,
    output_video: Path,
    frame_rate: str,
    width: int,
    height: int,
    model: Any,
) -> None:
    """Decode, annotate, and encode raw frames without temporary image files."""
    from PIL import Image

    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_video),
            "-map",
            "0:v:0",
            "-pix_fmt",
            "rgb24",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            frame_rate,
            "-i",
            "pipe:0",
            "-i",
            str(input_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_video),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert decoder.stdout is not None
    assert decoder.stderr is not None
    assert encoder.stdin is not None
    assert encoder.stderr is not None

    frame_byte_count = width * height * 3
    frame_buffer: list[Any] = []
    first_frame_boxes: list[BoundingBox] | None = None
    total_frames = 0
    total_yolo_calculations = 0

    def write_frames(frames: list[Any]) -> None:
        nonlocal total_frames
        for frame in frames:
            encoder.stdin.write(frame.tobytes())
            total_frames += 1
            print(f"Annotated frame {total_frames}", end="\r", flush=True)
            frame.close()

    try:
        while True:
            frame_data = read_raw_frame(decoder.stdout, frame_byte_count)
            if frame_data is None:
                break

            frame_buffer.append(Image.frombytes("RGB", (width, height), frame_data))
            if len(frame_buffer) != MIN_FRAME_ANNOTATE + 1:
                continue

            annotated, last_boxes, yolo_calculations = annotate_frames(
                frame_buffer, model, first_frame_boxes
            )
            total_yolo_calculations += yolo_calculations
            write_frames(annotated[:-1])

            for frame in frame_buffer[:-1]:
                frame.close()
            annotated[-1].close()
            frame_buffer = [frame_buffer[-1]]
            first_frame_boxes = last_boxes

        if not frame_buffer:
            raise RuntimeError(f"No video frames were decoded from {input_video}")

        annotated, _, yolo_calculations = annotate_frames(
            frame_buffer, model, first_frame_boxes
        )
        total_yolo_calculations += yolo_calculations
        write_frames(annotated)
        for frame in frame_buffer:
            frame.close()

        encoder.stdin.close()
        decoder_error = decoder.stderr.read().decode(errors="replace").strip()
        encoder_error = encoder.stderr.read().decode(errors="replace").strip()
        decoder_status = decoder.wait()
        encoder_status = encoder.wait()

        if decoder_status != 0:
            raise RuntimeError(f"FFmpeg decoder failed: {decoder_error}")
        if encoder_status != 0:
            raise RuntimeError(f"FFmpeg encoder failed: {encoder_error}")
    except BaseException:
        if not encoder.stdin.closed:
            encoder.stdin.close()
        for process in (decoder, encoder):
            if process.poll() is None:
                process.terminate()
        for process in (decoder, encoder):
            process.wait()
        raise
    finally:
        decoder.stdout.close()
        decoder.stderr.close()
        encoder.stderr.close()

    print(
        f"\nYOLO calculated {total_yolo_calculations}/{total_frames} frames; "
        "remaining boxes were interpolated."
    )


def validate_paths(input_video: Path, output_video: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be installed and available on PATH")
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")
    if input_video.suffix.lower() != ".mp4":
        raise ValueError(f"Input video must be an MP4 file: {input_video}")
    if output_video.suffix.lower() != ".mp4":
        raise ValueError(f"Output video must use the .mp4 extension: {output_video}")
    if input_video.resolve() == output_video.resolve():
        raise ValueError("Input and output video paths must be different")


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The Python dependencies are missing; run: pip install -r requirements.txt"
        ) from error

    input_video = args.input_video.expanduser().resolve()
    output_video = args.output_video.expanduser().resolve()
    validate_paths(input_video, output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    frame_rate, width, height = get_video_info(input_video)
    model = YOLO(YOLO_MODEL)

    print(
        f"Streaming {width}x{height} frames from {input_video} "
        f"at {frame_rate} fps..."
    )
    stream_video(input_video, output_video, frame_rate, width, height, model)

    print(f"Saved annotated video to {output_video}")
    time_end = time.time()
    print(f"Total time taken: {time_end - time_start:.2f} seconds")


if __name__ == "__main__":
    main()
