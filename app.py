import sys
import time
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from flask import Flask, request, jsonify, render_template

application = Flask(__name__)

app = application

# Load the VQA model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vqa', methods=['POST'])
def vqa():
    # Get image and question from the request
    image_file = request.files['image']
    question = request.form['question']

    # Read the image and preprocess input data
    raw_image = Image.open(image_file).convert('RGB')
    
    # Preprocess the image
    inputs = processor(raw_image, question, return_tensors="pt")

    # Perform generation
    start = time.perf_counter()
    out = model.generate(inputs["input_ids"], pixel_values=inputs["pixel_values"])
    end = time.perf_counter() - start

    # Postprocess result
    answer = processor.decode(out[0], skip_special_tokens=True)

    # Return the answer as JSON response
    return jsonify({"answer": answer, "processing_time": end})

if __name__ == '__main__':
    app.run(debug=True)


    
