# Libraries
from gensim.models import Word2Vec
from flask import Flask, render_template, request
import pickle
from joblib import load
from itertools import chain
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import numpy as np
from gensim.matutils import sparse2full
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
import csv


######################

# Loading in models and files
app = Flask(__name__)
document_dict = Dictionary.load('/dictionary_tokenized_articles.dict')
tfidf_model = TfidfModel.load('tfidf_model.model')

# Load Task 2_3 Model
pkl_filename = "model.pkl"
model = load(pkl_filename)
description_model = Word2Vec.load('W2v.model')
description_model_wv = description_model.wv

# Load csv of jobs from task 2_3
job_list = []
with open('job_lists.csv', mode='r', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file)
    for row in reader:
        job_list.append(row)

######################

# Functions
def docvecs(embeddings, docs):
    """
    Generates document vectors by summing up the embeddings
    of the words in the document and normalizing the vector.
    (Similar to one-hot encoding)

    Args:
        embeddings: Word embedding model (e.g. Word2Vec)
        docs: List of tokenized documents

    Returns:
        _type_: _description_
    """
    # Create a matrix of zeros with the same shape as token_dictionary
    vecs = np.zeros((len(docs), embeddings.vector_size))
    # Fill in the embeddings for the words that are present in the Word2Vec model
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        # if valid_keys is not empty
        if valid_keys:  
            # get the embedding vector of the word and store it inside index of the matrix
            docvec = np.vstack([embeddings[term] for term in valid_keys])
            # sum the vectors of all the words in the document
            docvec = np.sum(docvec, axis=0)
            # normalize the vector
            vecs[i,:] = docvec
    return vecs

######################

# Pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        f_title = request.form['title']
        f_content = request.form['description']
        
        # Tokenize and obtain its Word2Vec embedding
        tokenized_data = word_tokenize(f_content)
        dense_vector = docvecs(description_model_wv, [tokenized_data])[0]
        
        # Predict the label using the logistic regression model
        y_pred = model.predict([dense_vector])
        y_pred = y_pred[0]
        
        # Set a test message
        predicted_message = f"The category of this job is: {y_pred}."
        return render_template('index.html', predicted_message=predicted_message, title=f_title, description=f_content)
    else:
        return render_template('index.html')
 
@app.route('/jobs')
def jobs():
    category = request.args.get('category', None)
    
    if category:
        filtered_jobs = [job for job in job_list if job['category'] == category]
    else:
        filtered_jobs = job_list
    
    return render_template('jobs.html', jobs=filtered_jobs)
    # return render_template('jobs.html', jobs=job_list)

@app.route('/jobs/<string:job_id>')
def job_detail(job_id):
    job = next((job for job in job_list if job["id"] == job_id), None)
    if job:
        return render_template('jobs_detail.html', job=job)
    else:
        return "Job not found", 404

@app.route('/category/<string:category_name>')
def category_jobs(category_name):
    category_jobs = [job for job in job_list if job["category"] == category_name]
    return render_template('index.html', jobs=category_jobs)


@app.route('/create_listing', methods=['GET', 'POST'])
def create_listing():
    if request.method == 'POST':
        # Extract form data
        
        company = request.form['company']
        title = request.form['title']
        description = request.form['description']
        
        # Tokenize and obtain its Word2Vec embedding
        f_content = request.form['description']
        tokenized_data = word_tokenize(f_content)
        dense_vector = docvecs(description_model_wv, [tokenized_data])[0]
        
        # Predict the label using the logistic regression model
        y_pred = model.predict([dense_vector])
        y_pred = y_pred[0]
        recommended_category = [y_pred]
        
        # Show the recommended categories to the user and let them confirm or adjust
        return render_template('confirm_category.html', company = company, title=title, description=description, recommended_categories=recommended_category)

    # If it's a GET request, show the form
    return render_template('create_listing.html')

@app.route('/confirm_category', methods=['POST'])
def confirm_category():
    category = request.form['category']
    company = request.form['company']
    title = request.form['title']
    description = request.form['description']

    # Save the new job listing with the confirmed categories to your database (or in this case, add to the 'jobs' list)
    job_id = f"Job_{len(job_list)+1}"  # generate a new unique job id

    new_job = {'id': job_id, 'category':category, 'title': title, 'description': description, 'company': company}
    job_list.append(new_job)
    
    return render_template('job_posted.html', job_id=job_id)
