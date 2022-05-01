# IMPORTS - ALL ARE REQUIRED DO NOT REMOVE
import os
import pickle
import sklearn
import numpy as np
import pandas as pd

from PIL import Image
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from collections import Counter
from matplotlib import pyplot as plt
from flask import Flask, render_template, request, flash
from werkzeug.utils import redirect

app = Flask(__name__)

# CONSTANTS
app.secret_key = "super secret key"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
all_stop_words = pickle.load(open('models/all_stop_words.pkl', 'rb'))
classifier_model = pickle.load(open('models/classifier.pkl', 'rb'))
vec = pickle.load(open('models/vectorizer.pkl', 'rb'))
top_spam_words = pickle.load(open('models/top_spam_words.pkl', 'rb'))
top_ham_words = pickle.load(open('models/top_ham_words.pkl', 'rb'))

UPLOAD_FOLDER = 'uploads'
CLOUD_IMAGE = 'static/images/wordcloud/word_cloud.png'
WORD_CLOUD_CREATED_PATH = 'static/images/wordcloud/created_wordcloud.png'

SPAM_ICON = "/static/images/Spam.png"
NON_SPAM_ICON = "/static/images/NonSpam.png"
loading_gif = "/static/styles/throbber.gif"


# FUNCTIONS
def top_words(string):
    count = (Counter(string.split()).most_common(5))
    return count


def check_lists(list1, list2):
    result = any(elem in list1 for elem in list2)
    return result


def email_body_generator(uploaded_filepath):
    stream = open(uploaded_filepath, encoding='latin-1')
    is_body = False
    lines = []
    # extracts email body
    for line in stream:
        if is_body:
            lines.append(line)
        elif line == '\n':
            is_body = True
    stream.close()
    email_body = '\n'.join(lines)
    return email_body


def clean_msg_nohtml(message, stop_words=set(all_stop_words)):
    # Remove HTML tags
    soup = BeautifulSoup(message, 'html.parser')
    cleaned_text = soup.get_text()

    # splits up string to individual words
    words = word_tokenize(cleaned_text)

    filtered_words = []

    for word in words:
        if word not in stop_words and word.isalpha():
            filtered_words.append(word)

    # returns array
    return " ".join(filtered_words)


def convert_to_list(string):
    li = list(string.split(" "))
    return li


def wordcloud_generator(string):
    icon = Image.open(CLOUD_IMAGE)
    image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
    image_mask.paste(icon, box=icon)

    # converts the image object to an array
    rgb_array = np.array(image_mask)

    word_cloud = WordCloud(mask=rgb_array, background_color='white',
                           max_words=100, colormap='rainbow',
                           max_font_size=300)

    word_cloud.generate(string)

    plt.figure(figsize=[130, 110])
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(WORD_CLOUD_CREATED_PATH)


@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if os.path.exists(WORD_CLOUD_CREATED_PATH):
            os.remove(WORD_CLOUD_CREATED_PATH)
        content = request.files['userinput']

        if content.filename == '':
            flash('No selected file')
            return redirect(request.url)
        else:
            content.save(os.path.join(UPLOAD_FOLDER, content.filename))
            path = os.path.join(UPLOAD_FOLDER, content.filename)
            test_email_body = email_body_generator(path)
            cleaned_text = clean_msg_nohtml(test_email_body)
            wordcloud_generator(cleaned_text)
            number_of_words = cleaned_text.split()
            text_to_list = convert_to_list(cleaned_text)
            most_common = top_words(cleaned_text)
            df = pd.Series(text_to_list)
            # Vectorize the data:
            test_verify_classifier = vec.transform(df)
            # predicting the data with the classifier:
            prediction = classifier_model.predict(test_verify_classifier)[0]

        if prediction != 1:
            same_values = set(top_ham_words).intersection(text_to_list)
            extracted_words = '      |  '.join([str(item) for item in same_values])
            return render_template('results.html', text=extracted_words, message="Not Spam",
                                   most_common=most_common[0:5], number_of_words=len(number_of_words), results=NON_SPAM_ICON,
                                   word_cloud_image=WORD_CLOUD_CREATED_PATH)
        else:
            same_values = set(top_spam_words).intersection(text_to_list)
            extracted_words = '      |  '.join([str(item) for item in same_values])
            return render_template('results.html', text=extracted_words, message="Spam",
                                   most_common=most_common[0:5], number_of_words=len(number_of_words), results=SPAM_ICON,
                                   word_cloud_image=WORD_CLOUD_CREATED_PATH)

    return render_template('home.html', loading_gif=loading_gif)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
