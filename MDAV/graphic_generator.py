import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def generate3Dmetric2D(path=os.path.join("File_graphic", "m_M error.csv")):
    df = pd.read_csv(path)
    Points=[]

    x = df["m"]
    y = df["M"]
    z = df["metric error"]
   
    fig, ax = plt.subplots()
    graph=ax.tricontourf(x, y, z, levels=30, cmap="turbo", antialiased=True)
    fig.colorbar(graph)
    # ax.plot3D(x, y, z, 'green')
    ax.set(xlabel='m', ylabel='M')
    ax.set_title("m vs M In metric error")
    ax.plot
    plt.show()
