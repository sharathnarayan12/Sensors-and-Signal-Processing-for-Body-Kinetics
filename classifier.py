import numpy as np
from glob import glob
from os.path import basename
from sklearn.ensemble import RandomForestClassifier
from micromlgen import port

fileNames = ["/content/drive/MyDrive/new data/Abnormalhit data.csv","/content/drive/MyDrive/new data/Nh.csv"]

def load_features(files):
    dataset = None
    classmap = {}
    for class_idx, filename in enumerate(files):
        class_name = basename(filename)[:-4]
        classmap[class_idx] = class_name
        samples = np.genfromtxt(filename, delimiter=',',filling_values=0.0)
        labels = np.ones((len(samples), 1)) * class_idx
        samples = np.hstack((samples, labels))
        dataset = samples if dataset is None else np.vstack((dataset, samples))

    return dataset, classmap

def get_classifier(features):
    X, y = features[:, :-1], features[:, -1]

    return RandomForestClassifier(20, max_depth=10).fit(X, y)

if __name__ == '__main__':
    features, classmap = load_features(fileNames)
    classifier = get_classifier(features)
    c_code = port(classifier, classmap=classmap)
    
    print("Writing to a file")

    modelFile = open("ModelK.h", "w")
    modelFile.write(c_code)
    modelFile.close()

    print("Model file created")
