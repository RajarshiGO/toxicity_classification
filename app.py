from flask import Flask, render_template, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import numpy as np
import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from utils import LSTMClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
app = Flask(__name__)
app.config['SECRET_KEY'] = "Rajarshi"

class CommentForm(FlaskForm):
    comment = StringField("Type or paste the text below that you want to check for toxicity:", validators = [DataRequired()])
    submit = SubmitField("Submit")

vocab = torch.load("./vocab.pt")
vocab_size = len(vocab)
embed_len = 50
hidden_dim = 128
n_layers=2

tokenizer = get_tokenizer('basic_english')
model = LSTMClassifier(vocab_size, device)
model = model.to(device)
model.load_state_dict(torch.load("./model.pth", map_location = device))
max_words = 2000
classes = np.array(["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"])
@app.route('/', methods = ['GET', 'POST'])
def showform():
    comment = None
    report = None
    pred = None
    form = CommentForm()
    if form.validate_on_submit():
        comment = form.comment.data
        form.comment.data = ""
        text_data = vocab(tokenizer(comment))
        text_data = text_data +([0]* (max_words-len(text_data))) if len(text_data)<max_words else text_data[:max_words]
        text_data = torch.tensor(text_data, dtype = torch.int32)
        text_data = torch.unsqueeze(text_data, 0)
        text_data = text_data.to(device)
        pred = torch.squeeze(model(text_data)).cpu().detach().numpy()
        pred = np.where(pred>0.5)
        report = list(classes[pred])
        if(len(report) == 0):
            report = ["Non-Toxic"]
        flash("Report generated successfully!! Click on home if you want to get another prediction.")
    return render_template("index.html", report = report, form = form)

@app.route('/instructions', methods = ['GET'])
def show_instructions():
    return render_template("instructions.html")

@app.route('/about', methods = ['GET'])
def show_about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run(debug=True)
