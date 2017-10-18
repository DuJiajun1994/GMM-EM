import pandas
df = pandas.read_csv('data/train.txt', sep='\s+')
df1 = df.loc[df['label'] == 1]
df2 = df.loc[df['label'] == 2]
plt = df1.plot(kind='scatter', x='x', y='y').get_figure()
plt.savefig('output/label1_scatter.png')
plt = df2.plot(kind='scatter', x='x', y='y').get_figure()
plt.savefig('output/label2_scatter.png')