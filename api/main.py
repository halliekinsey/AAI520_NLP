from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import ssl

dev_mode = True

app = Flask(__name__)
CORS(app)

# Load pre-trained model and tokenizer
model_path = 'tuned_convo_model'
if dev_mode:
    model_path = 'C:/Users/kflin/Downloads/tuned_convo_model'

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# dictionary for different sessions with conversation history
sessions = {}

def generate_response(prompt, history, input_max_length=200, generated_max_length=25):
    prompt = ' '.join(history + [prompt])

    # Tokenize the prompt
    inputs = tokenizer(prompt, truncation=True, max_length=input_max_length, return_tensors='pt')
    input_length = inputs['input_ids'].shape[1]

    # Generate text
    outputs = model.generate(
        inputs['input_ids'], 
        max_length=generated_max_length + input_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
        do_sample=True,
        top_p=.95,
        top_k=50,
        temperature=0.2,
    )

    # Decode the generated text
    print(f"Generated output: {outputs[0]}")
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=False)
    print(f"Generated text: {generated_text}")
    return generated_text


@app.route('/', methods=['GET'])
def home():
    data = "Hello World"
    return jsonify({'data': data})

@app.route('/predict', methods=['POST'])
def predict():
    print("Request data: ", request.json)
    message = request.json['message']
    session_key = request.json['sessionKey']
    character1 = request.json['character1']
    character2 = request.json['character2']
    genres = request.json['genres']
    
    if session_key not in sessions:
        genre_str = ','.join(genres)
        message = f"<genre:{genre_str}><char:{character1}><char:{character2}>\n{message}"
        sessions[session_key] = []

        print("New session created")

    print("Message: ", message)

    sessions[session_key].append(message)
    conversation_history = sessions[session_key]
    response = generate_response(message, conversation_history)
    sessions[session_key].append(response)

    print(sessions)
    return jsonify({'response': response})

if __name__ == "__main__":
    if dev_mode:
        app.run(debug=True, port=5000)
    else:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain('/etc/letsencrypt/live/kellerflint.com/fullchain.pem', '/etc/letsencrypt/live/kellerflint.com/privkey.pem')
        # app.run(debug=True)
        app.run(host='0.0.0.0', port=5000, ssl_context=context)