from pathlib import Path
import numpy as np
import skimage
import os
from sklearn import svm, metrics
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split


from skimage.io import imread
from skimage.transform import resize

folder_path = "/home/amit/PycharmProjects/imageclassifierwithsvm/sample_images"
test_data_path = "/home/amit/PycharmProjects/imageclassifierwithsvm/Test_data"

def load_image_files(container_path, dimension=(64, 64)):
    """
    Load image files with categories as subfolder names
    which performs like scikit-learn sample dataset

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to

    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

def input_prediction(img_path):
"""
The function is to predict the class of the images in the Test_data folder
for each file the predicted class is printed in the output.

"""

    d = img_path
    for path in os.listdir(d):
        full_path = os.path.join(d, path)
        if os.path.isfile(full_path):
            dimension = (64, 64)
            img = skimage.io.imread(full_path)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_dicdata = (img_resized.flatten())
            flat_dicdata = np.array(flat_dicdata).reshape(1, -1)
            prediction = clf.predict(flat_dicdata)
            print(prediction)


image_dataset = load_image_files(folder_path)

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3, random_state=109)

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

#Remove the below comment to print the classification report for the SVM trained metrics
#print("Classification report for SVM classifier- \n{}\n".format(metrics.classification_report(y_test, y_pred)))

#function to get the prediction for the images inside the test data.
input_prediction(test_data_path)


