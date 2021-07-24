import numpy as np
from flask import Flask,render_template,request
import pickle
from tensorflow.keras import models
from util import ManDist
import pandas as pd
    

app = Flask(__name__)
model = models.load_model('./data/SiameseLSTM.h5',custom_objects={'ManDist': ManDist})


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/similarity',methods=['POST'])
def similarity():
    sentence1 = request.form["sentence1"]
    sentence2 = request.form["sentence2"]

    import pandas as pd

    import tensorflow as tf

    from util import make_w2v_embeddings
    from util import split_and_zero_padding
    from util import ManDist


    # File paths
    TEST_CSV = './data/test-20.csv'

    # Load training set
    test_df = pd.read_csv(TEST_CSV)
    new_row = {'sentence1':sentence1, 'sentence2':sentence2}
    test_df = test_df.append(new_row, ignore_index = True)
    for q in ['sentence1', 'sentence2']:
        test_df[q + '_n'] = test_df[q]


    # Make word2vec embedding
    embedding_dim = 300
    max_seq_length = 20
    test_df, embeddings = make_w2v_embeddings(test_df, embedding_dim=embedding_dim, empty_w2v=False)

    # Split to dicts and append zero padding.
    X_test = split_and_zero_padding(test_df, max_seq_length)

    # Make sure everything is ok
    assert X_test['left'].shape == X_test['right'].shape


    model = tf.keras.models.load_model('./data/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})

    prediction = model.predict([X_test['left'], X_test['right']]) * 100

    return render_template('index.html', prediction_text='Similarity :  {}%'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
