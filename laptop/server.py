import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# ---- Load MiDaS Model ----
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device).eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

def run_midas(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    output_display = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    output_display = cv2.applyColorMap(output_display, cv2.COLORMAP_JET)
    return output_display

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file.stream)
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    depth_result = run_midas(frame)

    # encode depth map ke base64 biar bisa dikirim ke HP
    _, buffer = cv2.imencode('.jpg', depth_result)
    encoded = base64.b64encode(buffer).decode("utf-8")
    return jsonify({"depth_map": encoded})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)