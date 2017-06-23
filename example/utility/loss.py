import numpy as np
import csv
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-p","--path" ,type=str, default='../dqn/log/loss.csv', help="the path of trajectory file")
parser.add_argument("-s", "--start", type =int,default=0, help="print the full data plot with dots")
parser.add_argument("-n", "--num", type=int, default= 100,help="the number of trajectories to be plotted. Default =100")
args = parser.parse_args()

with open(args.path,'rb') as f:
    reader = csv.DictReader(f)
    history = [e for e in reader]
    f.close()

dir_loss = [ep['Direction_loss'] for ep in history]
plt.plot(dir_loss)
plt.title('direction loss')
plt.ylabel('loss')
plt.xlabel('steps')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()