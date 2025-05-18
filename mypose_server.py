import cv2
from PIL import Image
import base64

import os

from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, Response, url_for, render_template_string, send_from_directory

from BlazeposeOpenvino import BlazeposeOpenvino, POSE_DETECTION_MODEL, LANDMARK_MODEL_FULL

OUTPUT_IMG = "output.jpg"

#VIDEO_SOURCE = "../videos/mixkit-cargo-ship-arriving-to-container-terminal-30979-hd-ready.mp4"
#VIDEO_SOURCE = "./video/wheelchair-exercises.mp4"
#VIDEO_SOURCE = "./video/sports-runner.mp4"

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#@app.route('/')
#def home():
#    return render_template('index.html')

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
                pose_correction=True,
                smoothing=True,
                filter_window_size=5,
                filter_velocity_scale=10,
                show_3d=False,
                crop=False,
                multi_detection=False,
                force_detection=False,
                output=OUTPUT_IMG)

    img_ht = ht.run()

    #print(ht.get_posture_feedback_history())

    return img_ht, ht.get_posture_feedback_history()[0]

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
            (processed_img, posture_feedback_history) = process_image(filepath)
            color_value='btn-success'
            if posture_feedback_history['posture_quality'] < 50:
                color_value = 'btn-danger'
            elif posture_feedback_history['posture_quality'] < 75:
                color_value = 'btn-warning'
            #
            return render_template('display.html',
                                   filename=processed_img,
                                   overall_posture_color_value=color_value,
                                   overall_posture=posture_feedback_history['overall_posture'],
                                   posture_quality=int(posture_feedback_history['posture_quality']),
                                   posture_status=posture_feedback_history['posture_status'],
                                   feedback=posture_feedback_history['feedback'])

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='localhost',port=9999, debug=True, threaded=True)

