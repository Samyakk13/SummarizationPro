from flask import Flask, request, jsonify, render_template
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

app = Flask(__name__)

# Load the model and tokenizer from local directories
model = PegasusForConditionalGeneration.from_pretrained('./model')
tokenizer = PegasusTokenizer.from_pretrained('./tokenizer')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data['text']
    max_length = data.get('max_length',300)

    print(f"Received max_length: {max_length}")

    # Tokenize and generate summary
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=1024)
    input_tokens = inputs['input_ids'].shape[1]
    print(f"length of 1 token: {input_tokens}")

    adjusted_max_length = min(input_tokens+max_length,1024)
    
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=adjusted_max_length,  # Dynamic max_length based on slider value
        min_length=100,  # A reasonable minimum length
        num_beams=6,  # Beam search to improve summary quality
        early_stopping=True , # Stop when the summary is ready
        temperature=1.0 
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)
