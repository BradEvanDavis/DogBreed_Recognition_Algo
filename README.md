[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview

In this project, a pipeline and webapp were created to process real-world, user-supplied images.  Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

Along with exploring state-of-the-art CNN models for classification and localization, the jupyter notebook provides the opportunity to initiate important design decisions about training the algorithm that will be usedt for the web app.  The webapp on the otherhand, allows the user to explore the algorithm and it's respective results when trained using a VGG19 network with transfer learning.


## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.   
4. Install packages from requirements.txt.
5. Open a terminal window and navigate to the project folder then open dog_app.ipynb to see how training was done.
6. For the Webapp go to the flask_webapp folder and then type export FLASK_APP="dog_recognition_webapp.py" into your terminal
7. Once in the flask_webapp folder and after you've defined an environment variable for FLASK_APP type "flask run" into your terminal to launch the webapp
8. Navigate to localhost:5000 to view the launched webapp (other ports can be defined using the -p option in the flask run command)
8. Test your images!
