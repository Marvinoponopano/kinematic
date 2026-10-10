"""Internal landmark representation.

MediaPipe hands the pipeline a protobuf (`NormalizedLandmarkList`, and a separate
`ClassificationList` per hand for handedness) whose shape then leaks into every layer that
touches a hand. Capture converts it exactly **once**, at the boundary, into the plain types
below: gestures, geometry, filters and the preview then work on ordinary Python objects with
no protobuf import and no MediaPipe call — which is what lets those layers be tested with
synthetic hands and no camera attached.

Nothing here touches the mouse or the screen; it is a data boundary.
"""

from dataclasses import dataclass
from typing import Any, Iterator, List, Sequence, Tuple

Point = Tuple[float, float]

#: MediaPipe hands always report 21 landmarks; anything shorter is a partial detection.
HAND_LANDMARK_COUNT = 21


@dataclass(frozen=True)
class Landmark:
    """One normalized landmark: x/y in 0..1 of the frame, z in the model's own units."""

    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class HandLandmarks:
    """One detected hand, in project-owned types.

    Supports ``len()``, iteration and indexing, so the pure geometry helpers in
    :mod:`kinematic.gesture` accept it exactly as they accept any landmark sequence.
    """

    landmarks: Tuple[Landmark, ...]
    handedness: str = "unknown"
    score: float = 0.0

    @property
    def is_complete(self) -> bool:
        """True when a full 21-point hand was reported (the geometry needs that)."""
        return len(self.landmarks) >= HAND_LANDMARK_COUNT

    def __len__(self) -> int:
        return len(self.landmarks)

    def __iter__(self) -> Iterator[Landmark]:
        return iter(self.landmarks)

    def __getitem__(self, index: int) -> Landmark:
        return self.landmarks[index]

    def point(self, index: int) -> Point:
        landmark = self.landmarks[index]
        return (landmark.x, landmark.y)

    def points(self) -> List[Point]:
        return [(landmark.x, landmark.y) for landmark in self.landmarks]


def hand_from_mediapipe(hand_landmarks: Any, handedness: str = "unknown", score: float = 0.0) -> HandLandmarks:
    """Convert one MediaPipe hand (anything with a ``.landmark`` sequence) to our type."""
    raw: Sequence[Any] = hand_landmarks.landmark
    return HandLandmarks(
        landmarks=tuple(
            Landmark(x=float(landmark.x), y=float(landmark.y), z=float(getattr(landmark, "z", 0.0)))
            for landmark in raw
        ),
        handedness=str(handedness),
        score=float(score),
    )


def hands_from_results(results: Any) -> List[HandLandmarks]:
    """Convert a MediaPipe ``process()`` result into zero or more :class:`HandLandmarks`.

    Duck-typed on purpose: the fakes used by the tests only need
    ``multi_hand_landmarks`` (and optionally ``multi_handedness``).
    """
    raw_hands = getattr(results, "multi_hand_landmarks", None) or []
    handedness = getattr(results, "multi_handedness", None) or []

    hands: List[HandLandmarks] = []
    for index, raw_hand in enumerate(raw_hands):
        label, score = "unknown", 0.0
        if index < len(handedness):
            classification = getattr(handedness[index], "classification", None) or []
            if classification:
                label = str(getattr(classification[0], "label", "unknown"))
                score = float(getattr(classification[0], "score", 0.0))
        hands.append(hand_from_mediapipe(raw_hand, label, score))
    return hands
