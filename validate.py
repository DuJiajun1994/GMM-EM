import pandas
import numpy as np
import json
import os
from utils import compute_prob


if __name__ == '__main__':
    val_file = os.path.join(os.getcwd(), 'data/dev.txt')
    df = pandas.read_csv(val_file,
                         dtype={
                             'x': np.float,
                             'y': np.float,
                             'label': np.int
                         },
                         sep='\s+')
    df.insert(3, 'predict_label', 1)
    df.insert(4, 'prob1', 1)
    df.insert(5, 'prob2', 1)

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

    correct_num = 0
    print('mistake examples:')
    print('x,y,label,prob1,prob2')
    for index, row in df.iterrows():
        x = np.array([row['x'], row['y']])
        prob1 = compute_prob(x, c1, u1, cov1).sum()
        prob2 = compute_prob(x, c2, u2, cov2).sum()
        if prob1 > prob2:
            predict_label = 1
        else:
            predict_label = 2
        df.loc[index, 'prob1'] = prob1
        df.loc[index, 'prob2'] = prob2
        df.loc[index, 'predict_label'] = predict_label
        if predict_label == row['label']:
            correct_num += 1
        else:
            print('{},{},{},{},{}'.format(row['x'], row['y'], row['label'], prob1, prob2))
    accuracy = float(correct_num) / len(df)
    print('accuracy: {}'.format(accuracy))

    output_annotation = os.path.join(os.getcwd(), 'output/val_result.txt')
    df.to_csv(output_annotation, index=False)

