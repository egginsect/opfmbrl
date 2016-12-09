import numpy as np
import matplotlib as mpl
import socket
if socket.gethostname() != "Davids-MacBook-Air-2.local":
    mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
from multiprocessing import Process, Pipe

def async_plotter(generate_ax_writers, num_axes=1):
    conn, child_conn = Pipe()
    p = Process(target=plot_params, args=(child_conn, generate_ax_writers, num_axes))
    p.daemon = True
    p.start()
    return conn.send

def plot_params(conn, generate_ax_writers, num_axes):
    plt.ion()
    fig = plt.figure()
    axes = [fig.add_subplot(num_axes, 1, i+1) for i in range(num_axes)]
    params = conn.recv()
    while True:
        for i, ax_writer in enumerate(generate_ax_writers(params)):
            ax_writer(axes)
            plt.draw()
            plt.pause(0.01)
            # if i == 0 or i == 1:
            #     plt.pause(0.5)
            if conn.poll():
                params = conn.recv()
                break
