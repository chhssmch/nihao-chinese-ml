import os
import warnings
warnings.filterwarnings("ignore")

import cv2
from PIL import Image, ImageDraw, ImageFont

# Переиспользуем словарь и модель из ml_image
from ml_image.main import COCO_ZH, get_model


# Пути к шрифтам с поддержкой CJK
_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int = 20):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _detect_on_frame(frame_path: str) -> list[dict]:
    model = get_model()
    results = model(frame_path, verbose=False)

    items = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf)
            if conf < 0.4:
                continue
            label_en = model.names[int(box.cls)]
            label_zh = COCO_ZH.get(label_en, label_en)
            xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
            items.append({
                "label_en": label_en,
                "label_zh": label_zh,
                "confidence": round(conf, 3),
                "box": xyxy,
            })
    return items


def _draw_on_frame(frame_path: str, items: list[dict], output_path: str) -> None:
    """Рисует боксы и китайские подписи на кадре."""
    img = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(20)

    for item in items:
        x1, y1, x2, y2 = item["box"]
        draw.rectangle([x1, y1, x2, y2], outline="#a4161a", width=3)

        label = item["label_zh"]
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        padding = 4
        label_y = max(0, y1 - text_h - 2 * padding)
        draw.rectangle(
            [x1, label_y, x1 + text_w + 2 * padding, label_y + text_h + 2 * padding],
            fill="#d4a017",
        )
        draw.text((x1 + padding, label_y + padding), label, fill="#7a0f13", font=font)

    img.save(output_path)


def analyze_video(video_path: str, frames_dir: str, fps: int = 1) -> dict:
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видеофайл.")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps else 0

    step = max(1, int(video_fps / fps))

    objects_counter = {} 
    annotated_frames = []

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            temp_path = os.path.join(frames_dir, f"_tmp_{saved_idx}.jpg")
            cv2.imwrite(temp_path, frame)

            items = _detect_on_frame(temp_path)

            if items:
                for item in items:
                    key = item["label_en"]
                    if key not in objects_counter:
                        objects_counter[key] = {
                            "label_en": item["label_en"],
                            "label_zh": item["label_zh"],
                            "count": 0,
                            "max_confidence": 0.0,
                        }
                    objects_counter[key]["count"] += 1
                    objects_counter[key]["max_confidence"] = max(
                        objects_counter[key]["max_confidence"],
                        item["confidence"],
                    )

                annotated_path = os.path.join(frames_dir, f"frame_{saved_idx}.jpg")
                _draw_on_frame(temp_path, items, annotated_path)
                annotated_frames.append(annotated_path)

            if os.path.exists(temp_path):
                os.remove(temp_path)

            saved_idx += 1

        frame_idx += 1

    cap.release()

    unique_objects = list(objects_counter.values())
    unique_objects.sort(key=lambda x: x["count"], reverse=True)

    return {
        "unique_objects": unique_objects,
        "annotated_frames": annotated_frames,
        "total_frames_processed": saved_idx,
        "duration_seconds": round(duration, 1),
    }