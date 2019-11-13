import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from utils.Utils import *
from models.keras_model import *


def test(loaders, model, criterion):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    model.to('cpu')

    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        data, target = data.to('cpu'), target.to('cpu')
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % ( 100. * correct / total, correct, total))


def load_state_dict(model, path):
    states = torch.load(path)
    model.load_state_dict(states)


def build_model(state_dict='../models/vgg19-dogs.pth'):
    model_transfer = models.vgg19(pretrained=True)
    model_transfer.classifier[6] = nn.Linear(4096, 133)
    load_state_dict(model_transfer, state_dict)
    return model_transfer


def torch_dog_detector(img_path):
    torch_dog = models.vgg16(pretrained=True)
    img = load_image(img_path)
    prediction = torch_dog(img)
    prediction = torch.argmax(prediction)
    return ((prediction <= 268) & (prediction >= 151))


def predict_dog_breed(img_path, model):
    class_names = torch.load('../models/class_names.pth')
    model.eval()
    model.to('cpu')
    output = F.softmax(model(load_image(img_path)), dim=1)
    output = torch.argmax(output)
    return class_names[output]


def run_app(img_path, model):
    ## handle cases for a human face, dog, and neither
    if torch_dog_detector(img_path):
        return "Dog Detected - I predict that this dog is a " + str(predict_dog_breed(img_path, model))
    elif face_detector(img_path):
        return "Human Detected - You look like a " + str(predict_dog_breed(img_path, model))
    else:
        return "Something else detected!"
