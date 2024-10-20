from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import ssl

app = Flask(__name__)
CORS(app)

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# dictionary for different sessions with conversation history
sessions = {}

def generate_response(prompt, history):
    prompt = ' '.join(history + [prompt])
    
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=512, return_tensors='pt')
    
    response = model.generate(input_ids, max_length=512, max_new_tokens=25, num_return_sequences=1, do_sample=True, temperature=.5, early_stopping=True)
    
    return tokenizer.decode(response[0], skip_special_tokens=True)

@app.route('/', methods=['GET'])
def home():
    data = "Hello World"
    return jsonify({'data': data})

@app.route('/predict', methods=['POST'])
def predict():
    print("Request data: ", request.json)
    message = request.json['message']
    session_key = request.json['sessionKey']
    
    if session_key not in sessions:
        sessions[session_key] = []

    sessions[session_key].append(message)
    conversation_history = sessions[session_key]
    response = generate_response(message, conversation_history)
    sessions[session_key].append(response)

    print(sessions)
    return jsonify({'response': response})

if __name__ == "__main__":
    dev_mode = False
    if dev_mode:
        app.run(debug=True, port=5000)
    else:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain('/etc/letsencrypt/live/kellerflint.com/fullchain.pem', '/etc/letsencrypt/live/kellerflint.com/privkey.pem')
        # app.run(debug=True)
        app.run(host='0.0.0.0', port=5000, ssl_context=context)