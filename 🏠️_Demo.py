import streamlit as st

st.title('RepRight')
st.write('## Squat Analysis')
recorded_file1 = './demo_vid/output_squats.mp4'
recorded_file = 'output_sample.mp4'
sample_vid = st.empty()
sample_vid.video(recorded_file1)

st.write('## Bicep Curl Analysis')
recorded_file2 = './demo_vid/output_biceps.mp4'
sample_vid = st.empty()
sample_vid.video(recorded_file2)

st.write('## Shoulder Raise Analysis')
recorded_file3 = './demo_vid/output_raises.mp4'
sample_vid = st.empty()
sample_vid.video(recorded_file3)

