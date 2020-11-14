import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

def visualize_dataset():
    features = iris['data']
    labels = iris['target']
    names = iris.target_names
    colors = ['b', 'r', 'g']

    _, (sepal, petal) = plt.subplots(ncols=2, figsize=(12, 5))


    for i, clr in enumerate(colors):

        sepal_x = features[:,1][labels==i]
        sepal_y = features[:,2][labels==i]
        petal_x = features[:,0][labels==i]
        petal_y = features[:,1][labels==i]
        sepal.scatter(sepal_x, sepal_y, c = clr)
        petal.scatter(petal_x, petal_y, c = clr)

    sepal.legend(names)
    petal.legend(names)
    sepal.set_xlabel('Sepal length')
    sepal.set_ylabel('Sepal width')
    petal.set_xlabel('Petal length')
    petal.set_ylabel('Petal width')
    plt.show()

def visualize_accuracy(conf_matrix):
    cm = pd.DataFrame(conf_matrix, index = iris.target_names, columns=iris.target_names)
    sn.set(font_scale=1.2)
    sn.heatmap(cm, annot=True, annot_kws={'size': 16}, cmap="Blues")
    plt.show()
# visualize_dataset(iris['data'], iris['target'])


    