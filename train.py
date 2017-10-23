import pandas
import numpy as np
import json
from utils import compute_prob


class_num = 4
iter_num = 1000


def compute_posteriori_prob(x, c, u, cov):
    prob = compute_prob(x, c, u, cov)
    prob_sum = np.sum(prob)
    prob = prob / prob_sum
    return prob


def train_model(x):
    x = np.array(x)
    c = np.array([1.0/class_num] * class_num)
    u = np.random.rand(class_num, 2)
    cov = np.zeros([class_num, 2, 2])
    for cls_id in range(class_num):
        cov[cls_id, 0, 0] = 1
        cov[cls_id, 1, 1] = 1

    for i in range(iter_num):
        print(i)
        prob = np.array([
            compute_posteriori_prob(data, c, u, cov) for data in x
        ])
        prob_sum = np.sum(prob, axis=0)

        for cls_id in range(class_num):
            ym = prob_sum[cls_id]
            um = 0
            covm = 0

            for n in range(len(x)):
                um += x[n] * prob[n, cls_id]
            um /= ym

            for n in range(len(x)):
                diff = x[n] - um
                diff = np.mat(diff)
                covm += np.array(diff.T * diff) * prob[n, cls_id]

            covm /= ym

            u[cls_id] = um
            cov[cls_id] = covm

    print('c: {}'.format(c))
    print('u: {}'.format(u))
    print('cov: {}'.format(cov))

    result = {
        'c': c.tolist(),
        'u': u.tolist(),
        'cov': cov.tolist()
    }
    return result

if __name__ == '__main__':
    df = pandas.read_csv('data/train.txt',
                         dtype={
                             'x': np.float,
                             'y': np.float,
                             'label': np.int
                         },
                         sep='\s+')
    x1 = []
    x2 = []
    for _, row in df.iterrows():
        if row['label'] == 1:
            x1.append([row['x'], row['y']])
        else:
            x2.append([row['x'], row['y']])
    print('train label 1:')
    result1 = train_model(x1)
    with open('output/label1_result.json', 'w') as fid:
        json.dump(result1, fid)
    print('train label 2:')
    result2 = train_model(x2)
    with open('output/label2_result.json', 'w') as fid:
        json.dump(result2, fid)
