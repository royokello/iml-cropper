from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from PIL import Image
import numpy as np

app = Flask(__name__)
working_dir = ""
image_files = []
current_index = 0

@app.route('/')
def index():
    return render_template('index.html', image_files=image_files, current_index=current_index)

@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory(working_dir, filename)

@app.route('/next_image', methods=['POST'])
def next_image():
    global current_index
    current_index = (current_index + 1) % len(image_files)
    return jsonify(index=current_index, filename=image_files[current_index])

@app.route('/prev_image', methods=['POST'])
def prev_image():
    global current_index
    current_index = (current_index - 1) % len(image_files)
    return jsonify(index=current_index, filename=image_files[current_index])

@app.route('/label', methods=['POST'])
def label_image():
    data = request.json
    filename = data['filename']
    label = data['label']
    # Save the label to a file or database
    with open(os.path.join(working_dir, 'labels.txt'), 'a') as f:
        f.write(f"{filename},{label}\n")
    return jsonify(success=True)

def label(working_directory: str):
    global working_dir, image_files, current_index
    working_dir = working_directory
    image_files = [f for f in os.listdir(working_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_index = 0
    app.run(debug=True)

if __name__ == '__main__':
    working_directory = 'path_to_your_working_directory'
    label(working_directory)
