import os
import sys
import urllib.request


_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/"
    "latest/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(_MODEL_DIR, "hand_landmarker.task")


def download_model():
    os.makedirs(_MODEL_DIR, exist_ok=True)

    if os.path.isfile(_MODEL_PATH):
        print(f"Model already exists at: {_MODEL_PATH}")
        return

    print("Downloading hand_landmarker.task ...")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"Downloaded successfully to: {_MODEL_PATH}")
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    download_model()
