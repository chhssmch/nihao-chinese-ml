import warnings
import os
from PIL import Image, ImageDraw, ImageFont
warnings.filterwarnings("ignore")

_model = None

# Словарь COCO (80 классов) → китайский
COCO_ZH = {
    "person": "人",
    "bicycle": "自行车",
    "car": "汽车",
    "motorcycle": "摩托车",
    "airplane": "飞机",
    "bus": "公共汽车",
    "train": "火车",
    "truck": "卡车",
    "boat": "船",
    "traffic light": "红绿灯",
    "fire hydrant": "消防栓",
    "stop sign": "停车标志",
    "parking meter": "停车计时器",
    "bench": "长凳",
    "bird": "鸟",
    "cat": "猫",
    "dog": "狗",
    "horse": "马",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "大象",
    "bear": "熊",
    "zebra": "斑马",
    "giraffe": "长颈鹿",
    "backpack": "背包",
    "umbrella": "雨伞",
    "handbag": "手提包",
    "tie": "领带",
    "suitcase": "行李箱",
    "frisbee": "飞盘",
    "skis": "滑雪板",
    "snowboard": "单板滑雪板",
    "sports ball": "运动球",
    "kite": "风筝",
    "baseball bat": "棒球棒",
    "baseball glove": "棒球手套",
    "skateboard": "滑板",
    "surfboard": "冲浪板",
    "tennis racket": "网球拍",
    "bottle": "瓶子",
    "wine glass": "酒杯",
    "cup": "杯子",
    "fork": "叉子",
    "knife": "刀",
    "spoon": "勺子",
    "bowl": "碗",
    "banana": "香蕉",
    "apple": "苹果",
    "sandwich": "三明治",
    "orange": "橙子",
    "broccoli": "西兰花",
    "carrot": "胡萝卜",
    "hot dog": "热狗",
    "pizza": "披萨",
    "donut": "甜甜圈",
    "cake": "蛋糕",
    "chair": "椅子",
    "couch": "沙发",
    "potted plant": "盆栽",
    "bed": "床",
    "dining table": "餐桌",
    "toilet": "马桶",
    "tv": "电视",
    "laptop": "笔记本电脑",
    "mouse": "鼠标",
    "remote": "遥控器",
    "keyboard": "键盘",
    "cell phone": "手机",
    "microwave": "微波炉",
    "oven": "烤箱",
    "toaster": "烤面包机",
    "sink": "水槽",
    "refrigerator": "冰箱",
    "book": "书",
    "clock": "钟表",
    "vase": "花瓶",
    "scissors": "剪刀",
    "teddy bear": "泰迪熊",
    "hair drier": "吹风机",
    "toothbrush": "牙刷",
}

def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
    return _model


def detect_objects(image_path: str, confidence_threshold: float = 0.3) -> list[dict]:
    model = get_model()
    results = model(image_path, verbose=False)

    items = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf)
            if conf < confidence_threshold:
                continue

            cls_id = int(box.cls)
            label_en = model.names[cls_id]
            label_zh = COCO_ZH.get(label_en, label_en)

            # box.xyxy — [x1, y1, x2, y2] в пикселях
            xyxy = box.xyxy[0].tolist()

            items.append({
                "label_en": label_en,
                "label_zh": label_zh,
                "confidence": round(conf, 3),
                "box": [round(v, 1) for v in xyxy],
            })

    items.sort(key=lambda x: x["confidence"], reverse=True)
    return items

# Пути к шрифтам с поддержкой CJK под разные ОС
_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",               
    "/System/Library/Fonts/Hiragino Sans GB.ttc",      
    "C:/Windows/Fonts/msyh.ttc",                        
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   
]


def _load_font(size: int = 24):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_boxes(image_path: str, items: list[dict], output_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(24)

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