import cv2
from PIL import Image
import base64

import os

from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, Response, url_for, render_template_string, send_from_directory

from BlazeposeOpenvino import BlazeposeOpenvino, POSE_DETECTION_MODEL, LANDMARK_MODEL_FULL

OUTPUT_IMG = "output.jpg"

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


## IMAGE
def process_image(img):

    print("-- opening ", img)
    #frame = Image.open(img)

    ht = BlazeposeOpenvino(input_src=img,
                pd_xml=POSE_DETECTION_MODEL,
                pd_device="CPU",
                pd_score_thresh=0.5, pd_nms_thresh=0.3,
                lm_xml=LANDMARK_MODEL_FULL,
                lm_device="CPU",
                lm_score_threshold=0.5,
                use_gesture=False,
                smoothing=True,
                filter_window_size=5,
                filter_velocity_scale=10,
                show_3d=False,
                crop=False,
                multi_detection=False,
                force_detection=False,
                output=OUTPUT_IMG)

    return ht.run()

##

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            #
            processed_img = process_image(filepath)
            #
            return render_template('display.html',filename=processed_img)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='localhost',port=9999, debug=True, threaded=True)

