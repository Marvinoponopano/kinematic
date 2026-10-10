"""Measure how much the hand reference jitters in *your* setup.

Hold your hand in front of the camera, keep it still, and curl/uncurl your fingers a few
times. This reports the per-frame displacement of the wrist and palm (normalized frame
units) next to `pointer_deadzone`, which is the number that decides whether finger
movement can nudge the cursor.

It never creates an output adapter, so it cannot move your mouse — it only reads
landmarks and prints statistics. Use it to tune `pointer_deadzone`:
it should sit above the jitter you see here and below deliberate hand movement.

    python3 tools/measure_stability.py --seconds 15
"""

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402

from kinematic.capture import ThreadedCamera  # noqa: E402
from kinematic.config import HandMouseConfig  # noqa: E402
from kinematic.gesture import hand_scale, landmark_point, palm_center, point_distance, wrist_point  # noqa: E402
from kinematic.gesture import INDEX_TIP  # noqa: E402


def describe(name: str, values: List[float]) -> str:
    if not values:
        return f"{name:<12} (no samples)"
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return (
        f"{name:<12} mean {statistics.fmean(values):.5f}  median {statistics.median(values):.5f}  "
        f"p95 {p95:.5f}  max {ordered[-1]:.5f}  (n={len(values)})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure hand-reference jitter with no mouse output.")
    parser.add_argument("--seconds", type=float, default=15.0, help="how long to sample (default 15)")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = parser.parse_args()

    config = HandMouseConfig()
    camera = ThreadedCamera(
        index=args.camera,
        width=config.capture_width,
        height=config.capture_height,
        fps=config.capture_fps,
        buffer_size=config.capture_buffer_size,
    )
    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        model_complexity=config.model_complexity,
    )
    print(f"sampling for {args.seconds:.0f}s - hold your hand in frame and curl your fingers\n")

    deltas: Dict[str, List[float]] = {"wrist": [], "palm": [], "index tip": [], "hand scale": []}
    curl_deltas: List[float] = []
    previous = None
    frames = 0
    with_hand = 0
    last_seq = -1
    deadline = time.monotonic() + args.seconds

    while time.monotonic() < deadline:
        seq, frame = camera.read()
        if frame is None or seq == last_seq:
            time.sleep(0.002)
            continue
        last_seq = seq

        frames += 1
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if not results.multi_hand_landmarks:
            previous = None  # tracking lost: re-anchor, do not measure across the gap
            continue

        with_hand += 1
        landmarks = results.multi_hand_landmarks[0].landmark
        scale = hand_scale(landmarks)
        now = {
            "wrist": wrist_point(landmarks),
            "palm": palm_center(landmarks),
            "index tip": landmark_point(landmarks, INDEX_TIP),
            "hand scale": scale,
        }
        if previous is not None:
            for key in ("wrist", "palm", "index tip"):
                deltas[key].append(point_distance(previous[key], now[key]))
            deltas["hand scale"].append(abs(previous["hand scale"] - now["hand scale"]))
            # Curl proxy: fingertip distance from the wrist in hand-widths. When this
            # changes, the fingers are moving even if the hand is not.
            previous_curl = point_distance(previous["index tip"], previous["wrist"]) / previous["hand scale"]
            curl = point_distance(now["index tip"], now["wrist"]) / scale
            curl_deltas.append(abs(curl - previous_curl))
        previous = now

    camera.release()
    hands.close()

    print(f"frames {frames}  with a hand {with_hand} ({0 if not frames else 100 * with_hand / frames:.0f}%)\n")
    if with_hand < 10:
        print("not enough hand samples - put your hand in front of the camera and run it again")
        return 1

    print("per-frame displacement, normalized frame units (smaller = more stable):")
    for key in ("wrist", "palm", "index tip", "hand scale"):
        print("  " + describe(key, deltas[key]))

    print(f"\nconfigured pointer_deadzone: {config.pointer_deadzone:.5f}")
    for key in ("wrist", "palm", "index tip"):
        values = deltas[key]
        if not values:
            continue
        above = sum(1 for value in values if value >= config.pointer_deadzone)
        print(f"  {key:<10} frames above the deadzone: {above}/{len(values)} ({100 * above / len(values):.1f}%)")

    if curl_deltas:
        print(f"\nfinger-curl change per frame: mean {statistics.fmean(curl_deltas):.5f} hand-widths")
    print(
        "\nRule of thumb: set pointer_deadzone a little above the p95 of the reference you use\n"
        "(wrist or palm), and well below the displacement of a deliberate hand movement."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
