import pandas
import numpy as np
import json
import os
from utils import compute_prob


if __name__ == '__main__':
    test_file = os.path.join(os.getcwd(), 'data/test.txt')
    df = pandas.read_csv(test_file, sep='\s+')
    df.insert(2, 'label', 1)
    with open('output/label1_result.json') as fid:
        param1 = json.load(fid)
        c1 = np.array(param1['c'])
        u1 = np.array(param1['u'])
        cov1 = np.array(param1['cov'])
    with open('output/label2_result.json') as fid:
        param2 = json.load(fid)
        c2 = np.array(param2['c'])
        u2 = np.array(param2['u'])
        cov2 = np.array(param2['cov'])
    for index, row in df.iterrows():
        x = np.array([row['x'], row['y']])
        prob1 = compute_prob(x, c1, u1, cov1).sum()
        prob2 = compute_prob(x, c2, u2, cov2).sum()
        if prob1 > prob2:
            df.loc[index, 'label'] = 1
        else:
            df.loc[index, 'label'] = 2

    output_annotation = os.path.join(os.getcwd(), 'output/test_result.txt')
    df.to_csv(output_annotation, index=False)

