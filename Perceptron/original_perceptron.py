import numpy as np
import random as rand

class original_perceptron(object):
    def __init__(self, *, learning_rate = 0.01, max_iteration = 5000):
        self.eta = learning_rate
        self.max_iteration = max_iteration

    def train(self, features, labels):
        dimensions = len(features)
        self.w = np.zeros(len(features[0]) + 1)

        for _ in range(self.max_iteration):
            test_index = rand.randint(0, dimensions - 1)
            xi = np.array(features[test_index])
            xi = np.append(xi, 1.0)
            yi = labels[test_index]
            if np.inner(self.w, xi) * yi > 0:
                continue
            print(f'Now handling {xi}, {yi}')
            print(f'Before w is {self.w}')
            self.w += self.eta * yi * xi
            print(f'After w is {self.w}')

    def predict(self, features):
        labels = np.zeros(len(features))
        for feature in features:
            xi = np.array(feature)
            xi = np.append(xi, 1.0)
            labels.append(self._predict(xi))
        return labels

    def _predict(self, xi):
        return np.inner(self.w, xi) > 0

if __name__ == '__main__':
    features = [[3, 3], [4, 3], [1, 1]]
    labels = [1, 1, -1]
    op = original_perceptron(learning_rate = 1)
    op.train(features, labels)
    print(op.w)
