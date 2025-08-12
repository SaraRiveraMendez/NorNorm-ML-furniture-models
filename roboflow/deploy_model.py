from roboflow import Roboflow
import os

TOKEN = os.getenv("ROBOFLOW_API_KEY")
VERSION = 5
MODEL_NAME = "furni-set-detect3"

if __name__ == "__main__":
    rf = Roboflow(api_key=TOKEN)
    project = rf.workspace().project("full-set-menu")

    model_path = f"../models/{MODEL_NAME}"
    project.version(VERSION).deploy(model_type="yolov8", model_path=model_path)

    print("Done!")
