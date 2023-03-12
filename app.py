from flask import Flask, render_template, request
from utils.general import Load_model,suggestor
import torch
from transformers import AutoTokenizer




app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Load_model(device)
tokenizer = AutoTokenizer.from_pretrained("model/code-search-net-tokenizer-mod")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/suggestions')
def generate_suggestions():
    prompt = request.args.get('code', '')
    print(prompt)

    # real-----
    suggestion = suggestor(prompt.strip(),model,tokenizer,device,max_lenght=25)

    suggestion = suggestion.replace('\n', '<br>')
    suggestion_html= [f'<li class="list-group-item">{suggestion}</li>']
    return {'suggestions': suggestion_html}


if __name__ == '__main__':
    app.run(debug=True)
