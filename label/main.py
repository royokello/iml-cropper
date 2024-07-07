import argparse
import csv
from flask import Flask, render_template, request, jsonify, send_from_directory
import os

app = Flask(__name__)
working_dir = ""
image_files = []
current_index = 0

def get_labeled_image_indices():
    """Retrieve and sort the indices of labeled images."""
    csv_file_path = os.path.join(working_dir, 'labels.csv')
    if not os.path.exists(csv_file_path):
        return []

    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        labeled_indices = [image_files.index(row[0]) for row in reader if row[0] in image_files]
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
    return render_template('index.html', image_files=image_files, current_index=current_index)

@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory(working_dir, filename)

@app.route('/next_image', methods=['POST'])
def next_image():
    global current_index
    current_index = (current_index + 1) % len(image_files)  # Loop back to the start if at the end
    return jsonify(index=current_index, filename=image_files[current_index])

@app.route('/prev_image', methods=['POST'])
def prev_image():
    global current_index
    current_index = (current_index - 1 + len(image_files)) % len(image_files)  # Loop back to the end if at the start
    return jsonify(index=current_index, filename=image_files[current_index])

@app.route('/next_labeled_image', methods=['POST'])
def next_labeled_image():
    labeled_indices = get_labeled_image_indices()
    global current_index
    current_index = find_nearest_index(labeled_indices, current_index, 1)
    return jsonify(index=current_index, filename=image_files[current_index])

@app.route('/prev_labeled_image', methods=['POST'])
def prev_labeled_image():
    labeled_indices = get_labeled_image_indices()
    global current_index
    current_index = find_nearest_index(labeled_indices, current_index, -1)
    return jsonify(index=current_index, filename=image_files[current_index])


@app.route('/label', methods=['POST'])
def label_image():
    data = request.json

    if data:
        filename = data['filename']
        labels = [data['top-left'], data['top-right'], data['bottom-left'], data['bottom-right']]

        csv_file_path = os.path.join(working_dir, 'labels.csv')

        # Check if file exists and read all data, update if filename exists
        records = []
        header = ['filename', 'top-left', 'top-right', 'bottom-left', 'bottom-right']
        file_exists = os.path.exists(csv_file_path)

        if file_exists:
            with open(csv_file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                next(reader)
                updated = False
                for row in reader:
                    if row[0] == filename:
                        records.append([filename] + labels)
                        updated = True
                    else:
                        records.append(row)
                if not updated:
                    records.append([filename] + labels)
        else:
            records.append([filename] + labels)

        # Write updated data back to CSV
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)
            writer.writerows(records)

        return jsonify(success=True)
    
    else:
        return jsonify(success=False)

def label(working_directory: str):
    global working_dir, image_files, current_index
    working_dir = working_directory
    image_files = [f for f in os.listdir(working_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_index = 0
    app.run(debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the image labeling Flask app.")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Directory where the image files are located.")
    
    args = parser.parse_args()
    label(args.working_dir)
