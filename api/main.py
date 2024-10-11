from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

CORS(app)

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    response = model.generate(input_ids, max_length=100, num_return_sequences=1, early_stopping=True)
    return tokenizer.decode(response[0], skip_special_tokens=True)

@app.route('/', methods=['GET'])
def home():
    data = "Hello World"
    return jsonify({'data': data})

@app.route('/predict', methods=['POST'])
def predict():
    print("Request data: ", request.json)
    response = generate_response(request.json['message'])
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
