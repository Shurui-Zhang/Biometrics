#!pip install removebg

import cv2 as cv
import torchvision.models.segmentation as segmentation
import numpy
import torch

from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from removebg import RemoveBg
from matplotlib import pyplot

from google.colab import drive
drive.mount('/content/drive')


def removeBackground():
    """
    This method is called only when the dataset is reconstructed
    """
    rmbg = RemoveBg("R2MCHXvXFFX6QE86aYuKN2Ef", "error.log")
    path = "/content/drive/MyDrive/Colab Notebooks/BIOM/biometrics/training"
    for pic in os.listdir(path):
        rmbg.remove_background_from_img_file("%s/%s" % (path, pic))

def getLoaderDataset(path_train, path_test):
    """
    Converts dataset to torchvision.datasets.ImageFolder and torch.utils.data.DataLoader object
    """
    transform = transforms.Compose([
        transforms.ToTensor()  # convert to tensor
    ])

    train_dataset = ImageFolder(path_train, transform)
    train_loader = DataLoader(train_dataset, batch_size=1)

    test_dataset = ImageFolder(path_test, transform)
    test_loader = DataLoader(test_dataset, batch_size=1)

    return train_loader, test_loader, train_dataset, test_dataset


def process_data(train_loader, test_loader):
    """
    Convert the data to ndarray
    """
    matrix_train_image = []
    matrix_train_label = []
    matrix_test_image = []
    matrix_test_label = []
    for train_image, train_label in train_loader:
        numpy_train_image = train_image.numpy()[0].transpose(1, 2, 0)
        numpy_train_label = train_label.numpy()
        matrix_train_image.append(numpy_train_image)
        matrix_train_label.append(numpy_train_label)

    for test_image, test_label in test_loader:
        numpy_test_image = test_image.numpy()[0].transpose(1, 2, 0)
        numpy_test_label = test_label.numpy()
        matrix_test_image.append(numpy_test_image)
        matrix_test_label.append(numpy_test_label)

    matrix_train_image = numpy.asarray(matrix_train_image)
    matrix_train_label = numpy.asarray(matrix_train_label)

    matrix_test_image = numpy.asarray(matrix_test_image)
    matrix_test_label = numpy.asarray(matrix_test_label)
    return matrix_train_image, matrix_train_label, matrix_test_image, matrix_test_label


def segment(image):
    """
    Semantic Segmentation
    """
    preprocess_input = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()

    predictions = model(preprocess_input(image).unsqueeze(0))['out'][0]
    preds = predictions.argmax(0).byte().cpu().numpy()
    return preds  # numpy.ndarray

def processImage(predictions):
    """
    Convert the image to a binary image
    """
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i][j] != 0:
                predictions[i][j] = 255
    return predictions

def cut(image):
    """
    Crop the image
    """
    height_min = (image.sum(axis=1) != 0).argmax()
    height_max = ((image.sum(axis=1) != 0).cumsum()).argmax()
    width_min = (image.sum(axis=0) != 0).argmax()
    width_max = ((image.sum(axis=0) != 0).cumsum()).argmax()
    head_top = image[height_min, :].argmax()

    size = height_max - height_min
    temp = numpy.zeros((size, size))
    temp = numpy.zeros((size, size))

    l1 = head_top - width_min
    r1 = width_max - head_top
    temp[:, (size // 2 - l1):(size // 2 + r1)] = image[height_min:height_max, width_min:width_max]

    temp = torch.from_numpy(temp)
    transform = transforms.Compose([
        transforms.CenterCrop((340, 160))
    ])

    temp = transform(temp)
    temp = temp.numpy()
    return temp

def calculateHuMoments(images):
    """
    Calculate the Hu Moments of all images in the traing set or test set
    """
    hu_moments_list = []
    for image in images:
        moments = cv.moments(image)
        hu_moments = cv.HuMoments(moments)  # numpy.ndarray (7, 1)
        hu_moments = hu_moments.flatten()
        hu_moments_list.append(hu_moments)
    return hu_moments_list  # list

def getFeatures(hu_moments_list):
    """
    The Hu Moments of the side-on view and the front-on view of an object are pieced together
    as the feature vector of this object.
    """
    features = []
    hu_moments = None
    for i in range(len(hu_moments_list)):
        if i % 2 == 0:
            hu_moments = hu_moments_list[i]
        else:
            feature = numpy.hstack((hu_moments, hu_moments_list[i]))
            features.append(feature)
            hu_moments = None
    features = numpy.asarray(features)
    return features

def modifyLabels(labels):
    """
    Merge the labels of side-on view and front-on view of each group in the training set into one.
    """
    result = []
    for i in range(len(labels)):
        if i % 2 == 0:
            result.append(labels[i])

    result = numpy.asarray(result)
    return result

def knn(k, matrix_train_features, matrix_train_label, matrix_test_features, classes):
    """
    using k-nearest-neighbour classifier to classify the images in test set
    """
    neigh = KNeighborsClassifier(k, weights='distance')
    neigh.fit(matrix_train_features, matrix_train_label)
    predictions = neigh.predict(matrix_test_features)

    results = []
    for i in range(len(predictions)):
        class_name = classes[predictions[i]]
        results.append(class_name)
    return results


#testing code
path_train = "/content/drive/MyDrive/Colab Notebooks/BIOM/biom_set_test/training"
path_test = "/content/drive/MyDrive/Colab Notebooks/BIOM/biom_set_test/test"
train_loader, test_loader, train_dataset, test_dataset = getLoaderDataset(path_train, path_test)
matrix_train_image, matrix_train_label, matrix_test_image, matrix_test_label = process_data(train_loader, test_loader)

train_images_o = []
train_images = []
for train_image in matrix_train_image:
  train_images_o.append(train_image)
  predictions = segment(train_image)
  image = processImage(predictions)
  train_images.append(image)

test_images_o = []
test_images = []
for test_image in matrix_test_image:
  test_images_o.append(test_image)
  predictions = segment(test_image)
  image = processImage(predictions)
  test_images.append(image)

processed_train_images = []
processed_test_images = []
for image in train_images:
  processed_image = cut(image)
  processed_train_images.append(processed_image)

for image in test_images:
  processed_image = cut(image)
  processed_test_images.append(processed_image)

train_hu_moments_list = calculateHuMoments(processed_train_images)
matrix_train_features = getFeatures(train_hu_moments_list)


test_hu_moments_list = calculateHuMoments(processed_test_images)
matrix_test_features = getFeatures(test_hu_moments_list)

classes = train_dataset.classes
matrix_train_label = modifyLabels(matrix_train_label)

results = knn(1, matrix_train_features, matrix_train_label, matrix_test_features, classes)
print(results)





























