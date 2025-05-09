from flask import Flask, request, render_template
import torch
from model import CNNMultimodalModel
from utils import predict_single_image, generate_gradcam_heatmap
import os

app = Flask(__name__)

# Load model
MODEL_PATH = 'best_resnet18_multimodal.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNMultimodalModel(
    backbone_name='resnet18',
    text_feature_dim=2,
    hidden_dim=256,
    num_classes=2,
    dropout_rate=0.3
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        age = int(request.form['age'])
        gender_str = request.form['gender']
        gender = 1 if gender_str.lower() == 'male' else 0

        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

        predicted_class, probabilities = predict_single_image(
            model, image_path, gender, age, device
        )
        heatmap_path = generate_gradcam_heatmap(model, image_path, gender, age, device)

        return render_template('result.html',
                               prediction=predicted_class,
                               prob_0=probabilities[0],
                               prob_1=probabilities[1],
                               heatmap_file=os.path.basename(heatmap_path)
                               )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)
