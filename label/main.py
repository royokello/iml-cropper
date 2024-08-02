import argparse
import csv
from flask import Flask, render_template, request, jsonify, send_from_directory
import os

import torch

from predict import predict
from utils import get_labels, get_model_by_latest, get_model_by_name, save_labels

app = Flask(__name__)
working_dir = ""
image_dir = ""
image_files = []
model = None
labels: dict[int, list[int]] = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_next_img_id(img_id: int, positive_step: bool, labels: dict[int, list[int]]) -> int:
    """
    Get the next image ID based on the current image ID and step direction.
    If positive_step is True, move to the next image ID with labels, otherwise move to the previous.
    If the end of the list is reached, wrap around to the beginning or end accordingly.
    """
    labeled_ids = sorted(labels.keys())
    if not labeled_ids:
        return img_id
    
    if img_id in labeled_ids:
        current_index = labeled_ids.index(img_id)
    else:
        current_index = min(range(len(labeled_ids)), key=lambda i: abs(labeled_ids[i] - img_id))

    if positive_step:
        next_index = (current_index + 1) % len(labeled_ids)
    else:
        next_index = (current_index - 1) % len(labeled_ids)

    return labeled_ids[next_index]


@app.route('/')
def index():
    global current_index, image_files, labels
    total_images = len(image_files)
    total_labels = len(labels)
    return render_template('index.html', total_images=total_images, total_labels=total_labels)

@app.route('/image/<int:img_id>')
def get_image(img_id):
    global image_dir
    return send_from_directory(image_dir, f"{img_id}.png")

@app.route('/next_labeled_image', methods=['POST'])
def next_labeled_image():
    global labels
    print("next_labeled_image ...")
    data = request.json
    if data:
        next_img_id = get_next_img_id(img_id=int(data["id"]), positive_step=data["step"], labels=labels)
        return jsonify(img_id=next_img_id)
    else:
        return jsonify(img_id=1)

@app.route('/label', methods=['POST'])
def label_image():
    global labels, working_dir
    data = request.json
    if data:
        labels[data["id"]] = [data['x1'], data['y1'], data['x2'], data['y2']]
        save_labels(directory=working_dir, labels=labels)
        return jsonify(success=True)
    else:
        return jsonify(success=False)

@app.route('/predict_crop/<int:img_id>')
def predict_crop(img_id):
    global image_dir, model, device
    model = get_model_by_name(device=device, )
    img_path = os.path.join(image_dir, f"{img_id}.png")
    prediction = predict(device=device, model=model, image_path=img_path)
    print(" * prediction: {prediction}")
    return jsonify(prediction=prediction)


def label(working_directory: str):
    global working_dir, image_files, current_index, image_dir, model, labels, device
    working_dir = working_directory
    image_dir = os.path.join(working_dir, '256p')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_index = 0
    labels = get_labels(directory=working_dir)
    model_dir = os.path.join(working_dir, 'model')
    model = get_model_by_latest(device=device, directory=model_dir)
    app.run(debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the image labeling Flask app.")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Directory where the image files are located.")
    
    args = parser.parse_args()
    label(args.working_dir)
