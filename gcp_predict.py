from util import ManDist
import h5py
import gcsfs
from keras.models import load_model

PROJECT_NAME = 'mlstmkeras'
CREDENTIALS = 'mlstmkeras-361e03a2a40f.json'
MODEL_PATH = 'mlstm_bucked/SiameseLSTM.h5'
from keras.utils import CustomObjectScope

FS = gcsfs.GCSFileSystem(project=PROJECT_NAME,
                         token=CREDENTIALS)
with FS.open(MODEL_PATH, 'rb') as model_file:
    model_gcs = h5py.File(model_file, 'r')
    myModel = load_model(model_gcs,custom_objects={'ManDist': ManDist})

myModel.summary()

sentenceA = "I love eating vegetables"
sentenceB = "vegetables are good for health"

myModel.predict(sentenceA, sentenceB)

