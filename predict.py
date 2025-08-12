import os
from collections import Counter

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine.results import Results

CLASS_COLORS = {
    0: (255, 0, 0),  # Red
    1: (128, 0, 128),  # Purple
    2: (0, 0, 255),  # Blue
    3: (155, 210, 0),  # Yellow
    4: (255, 165, 0),  # Orange
    5: (0, 255, 0),  # Lime Green
    6: (0, 155, 155),  # Cyan
    7: (255, 105, 180),  # Hot Pink
    8: (139, 69, 19),  # Saddle Brown
    9: (75, 0, 130),  # Indigo
    10: (0, 128, 128),  # Teal
    11: (255, 20, 147),  # Deep Pink
    12: (255, 69, 0),  # Red-Orange
    13: (173, 255, 47),  # Green-Yellow
    14: (128, 128, 0),  # Olive
    15: (0, 100, 0),  # Dark Green
    16: (0, 0, 139),  # Dark Blue
    17: (255, 0, 255),  # Magenta
    18: (0, 191, 255),  # Deep Sky Blue
}
CLASSES = {
    0: "CANTINE",
    1: "HIGH-TABLE",
    2: "LOUNGE",
    3: "MEETING-TABLE",
    4: "PHONEBOOTH",
    5: "PLANTS",
    6: "PRIVATE-DESK",
    7: "PRIVATE-OFFICE",
    8: "REST-FOR-1",
    9: "REST-FOR-2",
    10: "ROUND-TABLE",
    11: "STORAGE",
}


def load_model(model_name: str):
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        model_name,
        "weights",
        "best.pt",
    )
    model = YOLO(model_path)
    model.fuse()
    return model


def count_names(class_ids: list[str]):
    name_counts = Counter(class_ids)
    return dict(name_counts)


def predict_model(model: YOLO, image_path: str, conf: float = 0.4) -> list[Results]:
    return model.predict(  # type: ignore
        source=image_path,
        conf=conf,
        # iou=iou_treshold,
    )


def _get_boxes(results: list[Results]):
    boxes_cord: list[NDArray[np.floating]] = []
    confidences: list[NDArray[np.floating]] = []
    class_ids: list[NDArray[np.integer]] = []
    for result in results:
        if not result.boxes:
            continue
        boxes = result.boxes.cpu().numpy()
        boxes_cord.append(boxes.xyxy)
        confidences.append(boxes.conf)
        class_ids.append(boxes.cls)
    return boxes_cord, confidences, class_ids


def _draw_legend(image_height: float, image_width: float):

    global count_names_global, classes_global

    legend_width = int(image_width * 0.25)
    legend = np.ones((image_height, legend_width, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5 * image_height / 640
    text_thickness = int(np.floor(2 * image_height / 640))
    legend_y = int(np.floor(20 * image_height / 640))
    line_height = int(np.floor(20 * image_height / 640))

    for class_, val in count_names_global.items():
        class_name = classes_global.get(class_)
        if val == 0:
            continue
        cv2.putText(
            img=legend,
            text=f"{class_name}: {val}",
            org=(10, legend_y),
            fontFace=font,
            fontScale=font_scale,
            color=CLASS_COLORS.get(class_, (0, 0, 255)),
            thickness=text_thickness,
        )
        legend_y += line_height

    return legend


def paint_predictions(results: list[Results], image_path: str):

    boxes_cord, confidences, class_ids = _get_boxes(results=results)

    image = cv2.imread(image_path)
    image_h, image_w = image.shape[:2]

    image_name = image_path.split("/")[-1]

    global boxes, confidences_global, class_ids_global, classes_global, image_global, removed_boxes, count_names_global
    boxes = boxes_cord[0]
    confidences_global = confidences[0]

    class_ids_global = class_ids[0]
    classes_global = CLASSES
    image_global = image.copy()
    count_names_global = count_names(class_ids[0])
    removed_boxes = []

    def mouse_callback(event, x, y, param, flags):
        global boxes, confidences_global, class_ids_global, image_global, removed_boxes, count_names_global
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, box in enumerate(boxes):
                if box[0] <= x <= box[2] and box[1] <= y <= box[3]:

                    removed_boxes.append((boxes[i], confidences_global[i], class_ids_global[i]))
                    boxes = np.delete(boxes, i, axis=0)
                    count_names_global.update(
                        {class_ids_global[i]: count_names_global.get(class_ids_global[i]) - 1}
                    )
                    confidences_global = np.delete(confidences_global, i)
                    class_ids_global = np.delete(class_ids_global, i)
                    break
            redraw_image()

    def key_callback():
        global boxes, confidences_global, class_ids_global, removed_boxes
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("z") and removed_boxes:

                last_removed = removed_boxes.pop()
                boxes = np.vstack((boxes, last_removed[0]))
                confidences_global = np.append(confidences_global, last_removed[1])
                class_ids_global = np.append(class_ids_global, last_removed[2])
                count_names_global.update(
                    {last_removed[2]: count_names_global.get(last_removed[2]) + 1}
                )
                redraw_image()
            elif key == 27:
                break
            elif key == ord("q"):
                exit()

    def redraw_image():
        image_temp = image_global.copy()
        scale_factor = 1140 / 1118
        for i, box in enumerate(boxes):
            start_point = (int(box[0].item()), int(box[1].item()))
            end_point = (int(box[2].item()), int(box[3].item()))
            color = CLASS_COLORS.get(class_ids_global[i], (0, 0, 255))
            box_thickness = 3
            cv2.rectangle(image_temp, start_point, end_point, color, box_thickness)
            text_conf = str(confidences_global[i])
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.5
            text_thickness = 1
            text_size = cv2.getTextSize(text_conf, font, font_scale, text_thickness)[0]
            text_x = start_point[0] + (end_point[0] - start_point[0] - text_size[0]) // 2
            text_y = start_point[1] - 4
            c1, c2 = int(scale_factor * (453 - 397)), int(scale_factor * (177 - 85))
            c3, c4 = int(scale_factor * (1283 - 397)), int(scale_factor * (901 - 85))
            cv2.putText(
                image_temp,
                text_conf,
                (text_x, text_y),
                font,
                font_scale,
                color,
                text_thickness,
            )
            cv2.circle(image_temp, center=(c1, c2), radius=10, color=(0, 0, 255), thickness=-1)
            cv2.circle(image_temp, center=(c3, c4), radius=10, color=(0, 0, 255), thickness=-1)
        legend = _draw_legend(image_h, image_w)
        full_image = np.hstack((image_temp, legend))
        cv2.imshow(image_name, full_image)

    cv2.imshow(image_name, image)
    cv2.setMouseCallback(image_name, mouse_callback)
    key_callback()


if __name__ == "__main__":

    model_name = ""
    image_file = ""
    conf = 0.4

    model = load_model(model_name=model_name)

    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    image_path = os.path.join(image_dir, image_file)

    results = predict_model(model=model, image_path=image_path, conf=conf)

    paint_predictions(results, image_path)
    cv2.destroyAllWindows()
