#Dashboard to view past records from record.json
import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import animation
from matplotlib.animation import FuncAnimation

st.title('RepRight: Your Dashboard')

# Load the records
records = None
with open('record.json', 'r') as f:
    print("HIHI")
    records = json.load(f)

print(records)

if records is None:
    st.write('No records found')
else:
    #show plots over time for each exercise in columns
    for exercise in records:
        st.write(f'## {exercise}')
        df = pd.DataFrame(records[exercise]).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Correct'], marker='o', linestyle='-', color='green', label='Correct')
        ax.plot(df.index, df['Incorrect'], marker='o', linestyle='-', color='red', label='Incorrect')
        ax.set_title(f'{exercise} Accuracy over time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Count')
        #add legend
        ax.legend()
        
        #reduce number of x ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Weight'], marker='o', linestyle='-')
        ax.set_title(f'{exercise} Weights over time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Weight')
        #reduce number of x ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        st.pyplot(fig)
        st.write(df)

