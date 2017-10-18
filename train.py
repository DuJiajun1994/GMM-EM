import pandas
import numpy as np
import math


class_num = 4
iter_num = 1000

def compute_gauss(x, u, cov):
    diff = np.mat(x - u)
    cov = np.mat(cov)
    prob = np.exp(- (diff.T * cov.I * diff) / 2) / (2 * math.pi * math.sqrt(np.linalg.det(cov)))
    return prob

def compute_posteriori(x, c, u, cov):
    prob = np.array([
        compute_gauss(x, u[cls_id], cov[cls_id]) for cls_id in range(class_num)
    ])
    prob = prob * c
    prob_sum = np.sum(prob)
    prob = prob / prob_sum
    return prob

def train_model(x):
    x = np.array(x)
    c = np.ndarray([1.0/class_num]*class_num)
    u = np.random.rand(class_num, 2)
    cov = np.random.rand(class_num, 2, 2)

    for i in range(iter_num):
        print(i)
        prob = [compute_posteriori(data, c, u, cov) for data in x]
        prob_sum = np.sum(prob, axis=0)

        # u = np.sum(prob * x, axis=0) / prob_sum
        # diff = x - u
        # cov = np.sum(prob * diff * diff, axis=0) / prob_sum
        # c = prob_sum / np.sum(prob_sum)

    print('c: {}'.format(c))
    print('u: {}'.format(u))
    print('cov: {}'.format(cov))


df = pandas.read_csv('data/train.txt', sep='\s+')
x1 = []
x2 = []
for _, row in df.iterrows():
    if row['label'] == 1:
        x1.append([row['x'], row['y']])
    else:
        x2.append([row['x'], row['y']])
print('train label 1:')
train_model(x1)
print('train label 2:')
train_model(x2)

