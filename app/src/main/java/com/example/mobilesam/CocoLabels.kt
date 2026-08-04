package com.example.mobilesam

/**
 * COCO 80 class labels, shared by all detectors/segmenters (class order is the
 * same across YOLOv8n and YOLO-seg models).
 */
object CocoLabels {
    val names = arrayOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    )

    fun labelFor(classId: Int): String =
        names.getOrElse(classId) { "obj" }

    /** Chinese name for common classes; falls back to English for the rest. */
    private val chinese = mapOf(
        0 to "人", 1 to "自行车", 2 to "汽车", 3 to "摩托车", 4 to "飞机", 5 to "公交", 6 to "火车",
        7 to "卡车", 8 to "船", 13 to "长凳", 14 to "鸟", 15 to "猫", 16 to "狗", 17 to "马",
        18 to "羊", 19 to "牛", 20 to "大象", 21 to "熊", 22 to "斑马", 23 to "长颈鹿",
        24 to "背包", 25 to "雨伞", 26 to "手提包", 27 to "领带", 28 to "行李箱",
        29 to "飞盘", 32 to "球", 34 to "棒球棒", 36 to "滑板", 39 to "瓶子",
        40 to "酒杯", 41 to "杯子", 42 to "叉子", 43 to "刀", 44 to "勺子", 45 to "碗",
        46 to "香蕉", 47 to "苹果", 48 to "三明治", 49 to "橙子", 50 to "西兰花",
        51 to "胡萝卜", 52 to "热狗", 53 to "披萨", 54 to "甜甜圈", 55 to "蛋糕",
        56 to "椅子", 57 to "沙发", 58 to "盆栽", 59 to "床", 60 to "餐桌", 61 to "马桶",
        62 to "电视", 63 to "笔记本", 64 to "鼠标", 65 to "遥控器", 66 to "键盘",
        67 to "手机", 68 to "微波炉", 69 to "烤箱", 71 to "水槽", 72 to "冰箱",
        73 to "书", 74 to "时钟", 75 to "花瓶", 76 to "剪刀", 77 to "泰迪熊", 79 to "牙刷",
    )

    fun chineseFor(classId: Int): String =
        chinese[classId] ?: labelFor(classId)
}
