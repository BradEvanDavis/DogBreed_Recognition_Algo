import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SECRET_KEY = '12345'
    UPLOADED_PHOTOS_DEST = os.path.join(basedir, 'uploads') # you'll need to create a folder named uploads
