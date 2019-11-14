[//]: # (Image References)
[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Motivation / Definition:

In this project, a pipeline and webapp were created to process real-world, user-supplied images.  Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

This model explores state-of-the-art CNN models for classification and localization.  For training the algorithm please refer to  the jupyter notebook - here design decisions about training the algorithm used for the web app are explored.  The webapp, built using flask, allows the user to explore the algorithm and it's respective results when trained using a VGG19 network with transfer learning.  

![Sample Output][image1]

## Results and Analysis:

To train the model successfully I leveraged a pretrained VGG19 model where the final layer was replaced with outputs required for the dog breed recognition task.  VGG19 was chosen to take advantage of its state of the art performance as demonstrated in other image recognition tasks.

To accomplish high accuracy, all weights were locked in from the pretrained model and back prop was only used a modified version of the last layer to train it for this specific recognition task - doing this took the greatest advantage of transfer learning from previous tasks. Specifically, the last linear layer was replaced with a softmax with 133 categories rather then the original 1000 to be consistent with labeled dog breeds from the training set.

For training, cross-entropy loss was chosen as the loss metric and the amsgrad variant of Adam was chosen for gradient descent.  Using these metrics, training of the final layer was conducted for 10 epochs – overall less training was required for the VGG19 model relative to a model that was manually defined since it included already trained conv and fc layers. Taking advantage of the pretraining resulted in a model that required less training but also resulted in higher accuracy.  After training only the last layer of the VGG19 model for 10 epochs, with a learning rate set to 1e-3.  The model achieved an average validation loss score of 0.385 with an accuracy of 88%.  

Lastly, and for inclusion in the webapp, the dog breed recognizing model was combined with results from a pretrained VGG16 model and the CV2 face detector.  The original pretrained model was used to determine if the picture being presented to it was either a dog or not a dog by identifying – this was accomplished by identifying a non-dog from the original 1000 categories included in the VGG16 pretrained model.  If a dog was detected the model proceeded to identify its specific breed.  If the image did not determine that a dog was present then using the CV2 face detector the image was tested for the presence of a human face - if it was verified that a face was present the dog breed detector then named the dog breed that the human most closely resembles.  If both original tests were false then the algorithm determines that another object is present (neither do nor human). 


## Required Libraries Include:

torch, torchvision, tensorflow==1.13.1, keras==2.2.4, numpy, cv2, flask, flask_uploads, flask_wtf, wtforms, scikit-learn, glob, random, matplotlib, tqdm, os, PIL, collections, and time.  Please refer to requirements.txt in the github repo.


## Jupyter Notebook instructions:

1. Clone the repository and navigate to the downloaded folder.
2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.   
4. Install packages from requirements.txt
5. Open a terminal window and navigate to the project folder then open dog_app.ipynb to see how training was done.


## Webapp Instructions:

1. For the Webapp go to the flask_webapp folder and then type export FLASK_APP="dog_recognition_webapp.py" into your terminal
2. Once in the flask_webapp folder and after you've defined an environment variable for FLASK_APP type "flask run" into your terminal to launch the webap locally.
3. Navigate to localhost:5000 to view the launched webapp (other ports can be defined using the -p option in the flask run command)
4. Upload images to the webapp to be evaluated using the VGG19 model trained to recognize dog breeds.

## Conclusions:

Outputs leveraging the pretrained VGG19 model to obtain 88% accuracy were better than expected given that when attempting to create a model from scratch after 20 epochs it only approached 20% accuracy.  After reflection, a few possible points of improvement to focus on include the following: 

1) hyperparameter tuning
2) number of epochs trained
3) training additional layers in the classifier (besides just the last layer), and 
4) the method used for gradient descent. 

In order to further improve the models demonstrated in this notebook and webapp further training beyond 10 epochs will only serve to further improve the reliability of the modified VGG19 model. Furthermore, additional modifications can be made to the VGG19 model in order to improve efficiency since the model was quite memory intensive - these changes include training at half precision rather than full precision, and implementing distributed learning across multiple GPUs rather than implementing the model across GPUs in a serial manner. Training in serial increased available memory, however it also increases training time vs a single GPU. Distributing training using ring all-reduce or a similar distributed methodology would result in much greater training efficiency and would ultimately result in a more responsive model. Webapp performance could also be increased by decreasing the size of the model by converting it to 8bits and/or trimming nodes that do not sacrifice overall performance.
