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

st.title('AI Fitness Trainer: Your Dashboard')

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
        df['Total'] = df['Correct'] + df['Incorrect']
        df['Accuracy'] = df['Correct'] / df['Total']
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Accuracy'], marker='o', linestyle='-')
        ax.set_title(f'{exercise} Accuracy over time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Accuracy')
        st.pyplot(fig)
        st.write(df)
    # for exercise in records:
    #     st.write(f'## {exercise}')
    #     df = pd.DataFrame(records[exercise]).T
    #     df.index = pd.to_datetime(df.index)
    #     df = df.sort_index()
    #     df['Total'] = df['Correct'] + df['Incorrect']
    #     df['Accuracy'] = df['Correct'] / df['Total']
    #     fig, ax = plt.subplots()
    #     ax.plot(df.index, df['Accuracy'], marker='o', linestyle='-')
    #     ax.set_title(f'{exercise} Accuracy over time')
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('Accuracy')
    #     st.pyplot(fig)
    #     # st.write(df)
