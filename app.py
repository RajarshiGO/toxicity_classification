# Import flask and related packages 
from flask import Flask, render_template, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import numpy as np
# Import torch and tokenizer from torchtext
import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
# Import model from utils file
from utils import LSTMClassifier

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize the app instance
app = Flask(__name__)
app.config['SECRET_KEY'] = "Rajarshi"

# Define flask-WTForm class
class CommentForm(FlaskForm):
    comment = StringField("Type or paste the text below that you want to check for toxicity:", validators = [DataRequired()])
    submit = SubmitField("Submit")

# Load vocabulary from saved file and define the model parameters
vocab = torch.load("./vocab.pt")
vocab_size = len(vocab)
embed_len = 50
hidden_dim = 128
n_layers=2

# Initialize tokenizer and load the model
tokenizer = get_tokenizer('basic_english')
model = LSTMClassifier(vocab_size, device)
model = model.to(device)
model.load_state_dict(torch.load("./model.pth", map_location = device))
# Set the maximum words
max_words = 1000
# Construct a numpy array of the different classes
classes = np.array(["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"])
# Define the default route of the flask app
@app.route('/', methods = ['GET', 'POST'])
def showform():
    # Initialize the form parameters and the form object
    comment = None
    report = None
    pred = None
    form = CommentForm()
    # Validate form submission and extract the data
    if form.validate_on_submit():
        comment = form.comment.data
        form.comment.data = ""
        # Processs the input data
        text_data = vocab(tokenizer(comment))
        text_data = text_data +([0]* (max_words-len(text_data))) if len(text_data)<max_words else text_data[:max_words]
        text_data = torch.tensor(text_data, dtype = torch.int32)
        text_data = torch.unsqueeze(text_data, 0)
        text_data = text_data.to(device)
        pred = torch.squeeze(model(text_data)).cpu().detach().numpy()
        pred = np.where(pred>0.5)
        # Return the classification if the comment is toxic
        report = list(classes[pred])
        # If the comment is classified into none of the classes then label it as non-toxic and return it to the client
        if(len(report) == 0):
            report = ["Non-Toxic"]
        flash("Report generated successfully!! Click on home if you want to get another prediction.")
    return render_template("index.html", report = report, form = form)

# Route to show instructions page
@app.route('/instructions', methods = ['GET'])
def show_instructions():
    return render_template("instructions.html")

# Route to show about page
@app.route('/about', methods = ['GET'])
def show_about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run(debug=True)
