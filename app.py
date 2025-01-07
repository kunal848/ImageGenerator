from flask import Flask, request, jsonify, send_file, render_template
from diffusers import StableDiffusionPipeline
import torch

app = Flask(__name__)

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
pipe = pipe.to("cpu")  # Ensure the model uses the CPU

@app.route('/')
def home():
    # Serve the frontend HTML
    return render_template("index.html")

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.json.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Generate the image
    image = pipe(prompt).images[0]
    image_path = "output.png"
    image.save(image_path)

    return send_file(image_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
