import os
import cv2
from collections import Counter

import numpy as np
from ultralytics import YOLO

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
    1: "DOOR-DOUBLE",
    2: "DOOR-SINGLE",
    3: "ELEVATOR",
    4: "HIGH-TABLE",
    5: "LOUNGE",
    6: "MEETING-TABLE",
    7: "PHONEBOOTH",
    8: "PLANTS",
    9: "PRIVATE-DESK",
    10: "PRIVATE-OFFICE",
    11: "REST-FOR-2",
    12: "ROUND-TABLE",
    13: "SINK",
    14: "STAIRS",
    15: "STORAGE",
    16: "TOILET",
    17: "WASHBASIN",
    18: "WINDOW",
}


def load_model(model_path):
    model = YOLO(model_path)
    model.fuse()
    return model


def count_names(class_ids):
    name_counts = Counter(class_ids)
    return dict(name_counts)


def predict_model(model: YOLO, image, conf=0.4):
    results = model.predict(image, conf=conf)
    # iou=iou_threshold,  # IoU threshold for NMS
    return results


def _get_boxes(results):
    boxes_cord = []
    confidences = []
    class_ids = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        boxes_cord.append(boxes.xyxy)
        confidences.append(boxes.conf)
        class_ids.append(boxes.cls)
    # json_data = json.loads(results[0].tojson())

    # boxes.xywh	Tensor | ndarray	Boxes in [x, y, width, height] format.
    # boxes.xyxyn	Tensor | ndarray	Normalized [x1, y1, x2, y2] boxes relative to orig_shape.
    # boxes.xywhn	Tensor | ndarray	Normalized [x, y, width, height] boxes relative to orig_shape.
    return boxes_cord, confidences, class_ids


def _draw_legend(image_height, image_width):
    # Create a white canvas for the legend
    global count_names_global, classes_global

    legend_width = int(image_width * 0.25)  # Width of the legend canvas in pixels
    legend = (
        np.ones((image_height, legend_width, 3), dtype=np.uint8) * 255
    )  # White canvas

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5 * image_height / 640
    text_thickness = int(np.floor(2 * image_height / 640))
    legend_y = int(
        np.floor(20 * image_height / 640)
    )  # Start drawing text 20 pixels down from the top of the canvas
    line_height = int(
        np.floor(20 * image_height / 640)
    )  # Height between each line of text in the legend

    for class_, val in count_names_global.items():
        class_name = classes_global.get(class_)
        if val == 0:
            continue
        cv2.putText(
            legend,
            f"{class_name}: {val}",
            (10, legend_y),
            font,
            font_scale,
            CLASS_COLORS.get(class_, (0, 0, 255)),
            text_thickness,
        )
        legend_y += line_height

    return legend


def paint_predictions(results, image_path):
    boxes_cord, confidences, class_ids = _get_boxes(results)
    image = cv2.imread(image_path)
    image_h, image_w = image.shape[:2]
    print(image_h, image_w)

    image_name = image_path.split("/")[-1]

    global boxes, confidences_global, class_ids_global, classes_global, image_global, removed_boxes, count_names_global
    boxes = boxes_cord[0]
    confidences_global = confidences[0]

    class_ids_global = class_ids[0]
    classes_global = CLASSES
    image_global = image.copy()
    count_names_global = count_names(class_ids[0])
    removed_boxes = []  # Stack to keep track of removed boxes for undo

    def mouse_callback(event, x, y, param, flags):
        global boxes, confidences_global, class_ids_global, image_global, removed_boxes, count_names_global
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, box in enumerate(boxes):
                if (
                    box[0] <= x <= box[2] and box[1] <= y <= box[3]
                ):  # Check if click is inside the box
                    # Save the removed box to the stack
                    removed_boxes.append(
                        (boxes[i], confidences_global[i], class_ids_global[i])
                    )
                    boxes = np.delete(boxes, i, axis=0)  # Remove the box
                    count_names_global.update(
                        {
                            class_ids_global[i]: count_names_global.get(
                                class_ids_global[i]
                            )
                            - 1
                        }
                    )
                    confidences_global = np.delete(confidences_global, i)
                    class_ids_global = np.delete(class_ids_global, i)
                    break
            redraw_image()  # Redraw the image without the clicked box

    def key_callback():
        global boxes, confidences_global, class_ids_global, removed_boxes
        while True:
            key = cv2.waitKey(0) & 0xFF
            if (
                key == ord("z") and removed_boxes
            ):  # If 'z' is pressed and there are removed boxes
                last_removed = removed_boxes.pop()  # Get the most recent removed box
                # Add the removed box back to the list
                boxes = np.vstack((boxes, last_removed[0]))
                confidences_global = np.append(confidences_global, last_removed[1])
                class_ids_global = np.append(class_ids_global, last_removed[2])
                count_names_global.update(
                    {last_removed[2]: count_names_global.get(last_removed[2]) + 1}
                )
                redraw_image()  # Redraw the image with the restored box
            elif key == 27:  # ESC key to exit
                break
            elif key == ord("q"):
                exit()  # Save the results to a JSON file

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
            text_x = (
                start_point[0] + (end_point[0] - start_point[0] - text_size[0]) // 2
            )
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
            cv2.circle(
                image_temp, center=(c1, c2), radius=10, color=(0, 0, 255), thickness=-1
            )
            cv2.circle(
                image_temp, center=(c3, c4), radius=10, color=(0, 0, 255), thickness=-1
            )
        legend = _draw_legend(image_h, image_w)
        full_image = np.hstack((image_temp, legend))
        cv2.imshow(image_name, full_image)

    cv2.imshow(image_name, image)
    cv2.setMouseCallback(image_name, mouse_callback)
    key_callback()


if __name__ == "__main__":

    model_name = ""
    image_file = ""

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        model_name,
        "weights",
        "best.pt",
    )
    model = load_model(model_path)
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

    conf = 0.4

    print("Image file: " + image_file + "\n")
    image_path = os.path.join(image_dir, image_file)

    results = predict_model(model, image_path, conf=conf)
    paint_predictions(results, image_path)
    cv2.destroyAllWindows()
