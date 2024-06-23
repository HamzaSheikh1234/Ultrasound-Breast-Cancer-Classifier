from imports import *
import random
import numpy as np
from flask import send_from_directory

app = Flask(__name__)

# Load the pre-trained model
model = keras.models.load_model('best_model1.keras')

# Configure the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_array = load_and_preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            probability = predict(img_array=img_array, model=model)
            if probability > 0.5:
                result = 'Malignant'
            else:
                result = 'Benign'
            probability = probability[0]
            probability = probability * 100
            probability = np.round(probability, 2)
            probability = str(probability) + '%'
            return redirect(url_for('result', filename=filename, result=result, probability=probability))
    return render_template('upload.html')

@app.route('/result')
def result():
    filename = request.args.get('filename')
    result = request.args.get('result')
    probability = request.args.get('probability')
    return render_template('result.html', filename=filename, result=result, probability=probability)

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return f"File uploaded successfully: {filename}"

@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == '__main__':
    app.run(debug=True)