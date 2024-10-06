from flask import Flask, request, jsonify
from main import get_fusion_answer

app = Flask(__name__)

# Route for handling queries from the front-end
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query', '')
    
    if user_query:
        # Use RAG model to generate a response
        response = get_fusion_answer(user_query)
        return jsonify({'response': response})
    return jsonify({'response': 'No query provided.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
