# Visualize training history
import numpy as np
import csv
import argparse
import matplotlib.pyplot as plt
def read_csv(filepath):
   with open(filepath, 'rb') as f:
      reader = csv.DictReader(f)
      history = [e for e in reader]
      f.close()
   return history


parser = argparse.ArgumentParser()
parser.add_argument("-p","--path" ,type=str, default='../dqn/log/trajectory.csv', help="the path of trajectory file")
parser.add_argument("-s", "--start", type =int,default=0, help="print the full data plot with dots")
parser.add_argument("-n", "--num", type=int, default= 100,help="the number of trajectories to be plotted. Default =100")
args = parser.parse_args()

history = read_csv(args.path)
print len(history)


start = args.start
show_num = args.num

X=[]
Y=[]
i = 0
epoch = [ep['epoch'] for ep in history]
epoch_np = np.array(epoch)
x = []
y = []
reward = []
collision = []
epoch = int(epoch[0])

for ep in history:

   if int(ep['epoch']) == int(epoch):
      x.append(float(ep['x']))
      y.append(-float(ep['y']))

      if ep['done'] == 'True':
         epoch = int(ep['epoch']) + 1
         X.append(x)
         Y.append(y)
         reward.append(ep['reward'])
         collision.append(ep['collision'])
         x = []
         y = []

epoch_max = len(X)
#plt.xlim((-550, 350))
#plt.ylim((-550, 350))
plt.ylabel('y')
plt.xlabel('x')
for i in range(start, min(start + show_num,len(X)) ):
   if collision[i] == 'True' or float(reward[i]) <=0 :
      line_tpye = 'k'
      if collision[i] == 'True':#collision point
         plt.scatter(X[i][-1], Y[i][-1], c='magenta', s=30, alpha=0.6, edgecolors='white')
   else :
      size = max(float(reward[i]),0.1) *10
      plt.scatter(X[i][-1], Y[i][-1], c='green', s=size , alpha=0.6, edgecolors='white')
      line_tpye = 'r'
   plt.scatter(X[i][0], Y[i][0], c='blue', s=30, alpha=0.6, edgecolors='white')

   plt.plot(X[i], Y[i], line_tpye)

plt.show()