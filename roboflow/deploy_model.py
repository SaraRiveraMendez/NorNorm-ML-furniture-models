import os

from roboflow import Roboflow

TOKEN = os.getenv("48MGvsQcK2gpWRbHeOnn")
VERSION = 5
MODEL_NAME = "best.pt"


def ge():
    pass


def de():
    pass


if __name__ == "__main__":
    rf = Roboflow(api_key=TOKEN)
    project = rf.workspace().project("best.pt")

    model_path = f"../../../weights/{MODEL_NAME}"
    project.version(VERSION).deploy(model_type="yolov12", model_path=model_path)

    print("Done!")
