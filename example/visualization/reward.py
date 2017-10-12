#!/usr/bin/env python

import os
import gym
import matplotlib
import matplotlib.pyplot as plt
import itertools
import sys
import argparse
import numpy as np
from scipy.interpolate import pchip
from gym import monitoring

class LivePlot(object):
    def __init__(self, outdir, data_key='episode_rewards', line_color='blue'):
        """
        Liveplot renders a graph of either episode_rewards or episode_lengths
        Args:
            outdir (outdir): Monitor output file location used to populate the graph
            data_key (Optional[str]): The key in the json to graph (episode_rewards or episode_lengths).
            line_color (Optional[dict]): Color of the plot.
        """
        #data_key can be set to 'episode_lengths'
        self.outdir = outdir
        self._last_data = None
        self.data_key = data_key
        self.line_color = line_color

        #styling options
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')
        plt.xlabel("steps")
        plt.ylabel("cumulated episode rewards")
        fig = plt.gcf().canvas.set_window_title('averaged_simulation_graph')
        matplotlib.rcParams.update({'font.size': 15})

    def plot(self, full=True, dots=False, average=0, interpolated=0):
        print self.outdir
        results = monitoring.load_results(self.outdir)

        data =  results[self.data_key]
        steps = results['episode_lengths']
        #print steps
        count_steps = 0
        for i in range(len(steps)):
            count_steps += steps[i]
            steps[i] = count_steps

        avg_data = []

        if full:
            plt.plot(steps,data, color='blue')
        if dots:
            plt.plot(steps,data, '.', color='black')
        if average > 0:
            average = int(average)
            for i, val in enumerate(data):
                '''if i%average==0:
                    if (i+average) < len(data)+average:
                        avg =  sum(data[i:i+average])/average
                        avg_data.append(avg)'''
                if i < average:
                    avg = np.array(data[:average]).mean()
                else:
                    avg = np.array(data[(i-average):i]).mean()
                avg_data.append(avg)
            #new_data = expand(avg_data,average)
            plt.plot(steps,avg_data, color='red', linewidth=2.5)
        if interpolated > 0:
            avg_data = []
            avg_data_points = []
            n = len(data)/interpolated
            if n == 0:
                n = 1
            data_fix = 0
            for i, val in enumerate(data):
                if i%n==0:
                    if (i+n) <= len(data)+n:
                        avg =  sum(data[i:i+n])/n
                        avg_data.append(avg)
                        avg_data_points.append(i)
                if (i+n) == len(data):
                    data_fix = n
            
            x = np.arange(len(avg_data))
            y = np.array(avg_data)

            interp = pchip(avg_data_points, avg_data)
            xx = np.linspace(0, len(data)-data_fix, 1000)
            plt.plot(xx, interp(xx), color='green', linewidth=3.5)        

        # pause so matplotlib will display
        # may want to figure out matplotlib animation or use a different library in the future
        plt.pause(0.000001)

def expand(lst, n):
    lst = [[i]*n for i in lst]
    lst = list(itertools.chain.from_iterable(lst))
    return lst

def pause():
    programPause = raw_input("Press the <ENTER> key to finish...")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path", type=str, default='../dqn/log_multi/monitor/tmp', help="the path of monitor file")
    parser.add_argument("-f", "--full", action='store_true', help="print the full data plot with lines")
    parser.add_argument("-d", "--dots", action='store_true', help="print the full data plot with dots")
    parser.add_argument("-a", "--average", type=int, nargs='?', const=100, metavar="N", help="plot an averaged graph using N as average size delimiter. Default = 50")
    parser.add_argument("-i", "--interpolated", type=int, nargs='?', const=50, metavar="M", help="plot an interpolated graph using M as interpolation amount. Default = 50")
    args = parser.parse_args()

    #print args.path
    plotter = LivePlot(args.path)

    if len(sys.argv)==1:
        # When no arguments given, plot full data graph
        plotter.plot(full=True)
    else:
        plotter.plot(full=args.full, dots=args.dots, average=args.average, interpolated=args.interpolated)

    pause()
