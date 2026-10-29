import argparse
import time
from collections import deque

import cv2
from deepface import DeepFace


EMOTION_COLORS = {
    "angry": (70, 70, 235),
    "disgust": (70, 170, 70),
    "fear": (180, 100, 220),
    "happy": (60, 210, 90),
    "sad": (230, 150, 70),
    "surprise": (40, 220, 240),
    "neutral": (220, 220, 220),
}


def analyze_face(frame):
    """Return DeepFace face-analysis results; failures are handled by the caller."""
    return DeepFace.analyze(
        img_path=frame,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="opencv",
        silent=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Display estimated facial-expression categories from your webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--analyze-every", type=int, default=5,
        help="Run emotion analysis every N frames (default: 5)"
    )
    parser.add_argument(
        "--scale", type=float, default=0.75,
        help="Resize frame before analysis to improve speed (default: 0.75)"
    )
    args = parser.parse_args()

    if args.analyze_every < 1:
        parser.error("--analyze-every must be at least 1")
    if not 0.2 <= args.scale <= 1.0:
        parser.error("--scale must be between 0.2 and 1.0")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print("Error: Could not open the camera. Check camera permissions or try --camera 1.")
        return

    print("Camera started. Press Q or Esc in the video window to quit.")
    print("First analysis may take longer while the model initializes.")

    frame_count = 0
    last_results = []
    last_analysis_time = 0.0
    fps_samples = deque(maxlen=20)
    previous_time = time.time()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Error: Could not read a frame from the camera.")
                break

            frame_count += 1
            display = frame.copy()

            if frame_count == 1 or frame_count % args.analyze_every == 0:
                try:
                    analysis_frame = cv2.resize(
                        frame, None, fx=args.scale, fy=args.scale,
                        interpolation=cv2.INTER_AREA
                    ) if args.scale < 1.0 else frame
                    results = analyze_face(analysis_frame)
                    if isinstance(results, dict):
                        results = [results]
                    last_results = results or []
                    last_analysis_time = time.time()
                except Exception as exc:
                    # Keep the camera feed alive if a frame cannot be analyzed.
                    print(f"Analysis warning: {exc}")
                    last_results = []

            # DeepFace regions refer to the resized analysis frame, so scale boxes
            # back up to the original camera-frame dimensions.
            scale_back = 1.0 / args.scale
            for result in last_results:
                region = result.get("region") or {}
                x = int(region.get("x", 0) * scale_back)
                y = int(region.get("y", 0) * scale_back)
                w = int(region.get("w", 0) * scale_back)
                h = int(region.get("h", 0) * scale_back)

                emotions = result.get("emotion") or {}
                if not emotions:
                    continue
                emotion = max(emotions, key=emotions.get).lower()
                confidence = float(emotions[emotion])
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))

                if w > 0 and h > 0:
                    cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                    label = f"{emotion.capitalize()} | {confidence:.1f}%"
                    text_y = max(28, y - 10)
                    cv2.rectangle(
                        display, (x, text_y - 24),
                        (x + min(max(w, 190), 310), text_y + 4), color, -1
                    )
                    cv2.putText(
                        display, label, (x + 6, text_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2,
                        cv2.LINE_AA
                    )

                    # Compact confidence bars for each expression category.
                    bar_x = x
                    bar_y = min(display.shape[0] - 12, y + h + 20)
                    for name, score in sorted(
                        emotions.items(), key=lambda item: item[1], reverse=True
                    ):
                        if bar_y > display.shape[0] - 8:
                            break
                        cv2.putText(
                            display, f"{name}: {score:.0f}%",
                            (bar_x, bar_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, color, 1, cv2.LINE_AA
                        )
                        bar_y += 17

            now = time.time()
            elapsed = max(now - previous_time, 1e-6)
            fps_samples.append(1.0 / elapsed)
            previous_time = now
            fps = sum(fps_samples) / len(fps_samples)

            cv2.rectangle(display, (8, 8), (390, 62), (25, 25, 25), -1)
            cv2.putText(
                display, f"Facial expression demo | FPS: {fps:.1f}",
                (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA
            )
            status = "Analyzing..." if time.time() - last_analysis_time < 2 else "Waiting for face / analysis"
            cv2.putText(
                display, status, (18, 51), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (190, 220, 190), 1, cv2.LINE_AA
            )

            cv2.imshow("Facial Expression Recognition (Q/Esc to quit)", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
