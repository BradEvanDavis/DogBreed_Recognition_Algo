import os
from flask import Flask, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from models.pytorch_model import *
from models.keras_model import *

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'uploads') # you'll need to create a folder named uploads

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image only!'), FileRequired('File was empty!')])
    submit = SubmitField('Upload')
    # doghuman = {}


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    doghuman='hello'
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)
        filename = 'uploads/' + filename
        model = build_model()
        doghuman = str(run_app(filename, model))
        return render_template('index.html', form=form, file_url=file_url, doghuman=doghuman)
    else:
        file_url = None
        return render_template('index.html', form=form, file_url=file_url, doghuman=doghuman)


if __name__ == '__main__':
    app.run()