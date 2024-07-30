import argparse
import csv
from flask import Flask, render_template, request, jsonify, send_from_directory
import os

from utils import get_labels, load_model, save_labels

app = Flask(__name__)
working_dir = ""
image_dir = ""
image_files = []
model = None
labels: dict[int, list[int]] = {}

def get_labeled_image_indices() -> list[int]:
    """
    Retrieve and sort the indices of labeled images.
    """
    csv_file_path = os.path.join(working_dir, 'labels.csv')
    if not os.path.exists(csv_file_path):
        return []

    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        labeled_indices = [int(row[0]) for row in reader]
    labeled_indices.sort()
    return labeled_indices

def find_nearest_index(indices, current_index, step):
    """Find nearest index in sorted list of labeled indices depending on direction."""
    if step > 0:  # next labeled image
        for index in indices:
            if index > current_index:
                return index
        return indices[0] if indices else current_index  # loop to start if not found
    else:  # previous labeled image
        for index in reversed(indices):
            if index < current_index:
                return index
        return indices[-1] if indices else current_index  # loop to end if not found

@app.route('/')
def index():
    global current_index, image_files
    total_images = len(image_files)
    return render_template('index.html', total_images=total_images)

@app.route('/image/<int:i>')
def get_image(i):
    global image_dir
    return send_from_directory(image_dir, f"{i}.png")

@app.route('/next_labeled_image/<int:i>')
def next_labeled_image(i):
    labeled_indices = get_labeled_image_indices()
    next_index = find_nearest_index(labeled_indices, i, 1)
    return jsonify(index=next_index)

@app.route('/prev_labeled_image/<int:i>')
def prev_labeled_image(i):
    labeled_indices = get_labeled_image_indices()
    prev_index = find_nearest_index(labeled_indices, i, -1)
    return jsonify(index=prev_index)


@app.route('/label', methods=['POST'])
def label_image():
    global labels, working_dir

    data = request.json

    if data:
        id = int(data['id'])
        sample = [int(data['x1']), int(data['x2']), int(data['y1']), int(data['y2'])]

        labels[id] = sample
        
        save_labels(directory=working_dir, labels=labels)

        return jsonify(success=True)
    
    else:
        return jsonify(success=False)

# @app.route('/predict_crop/<int:i>')
# def predict_crop(i):
#     global image_dir, model
#     img_path = os.path.join(image_dir, f"{i}.png")
#     prediction = predict(model, img_path)
#     return jsonify(prediction)


def label(working_directory: str):
    global working_dir, image_files, current_index, image_dir, model, labels
    working_dir = working_directory
    image_dir = os.path.join(working_dir, '256p')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_index = 0
    labels = get_labels(directory=working_dir)
    model = load_model(working_dir)
    app.run(debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the image labeling Flask app.")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Directory where the image files are located.")
    
    args = parser.parse_args()
    label(args.working_dir)
