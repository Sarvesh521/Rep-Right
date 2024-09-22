import av
import os
import sys
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from aiortc.contrib.media import MediaRecorder
import time
from utils import draw_text

BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)


from utils import get_mediapipe_pose
from process_frame_BicepCurls import ProcessFrameCurls
from process_frame_Squats import ProcessFrameSquats
from process_frame_shoulder_raises import ProcessFrameRaises

from thresholds import get_thresholds_beginner, get_thresholds_pro
from Classifier import predict_image  # Import the predict_image function
from Classifier import predict_video  # Import the predict_video function


st.title('AI Fitness Trainer: Exercise Analysis')

# mode = st.radio('Select Mode', ['Beginner', 'Pro'], horizontal=True)

thresholds = get_thresholds_beginner() 



live_process_frame_squat = ProcessFrameSquats(thresholds=thresholds, flip_frame=True)
live_process_frame_curl = ProcessFrameCurls(thresholds=thresholds, flip_frame=True)
live_process_frame_raise = ProcessFrameRaises(thresholds=thresholds, flip_frame=True)
# Initialize face mesh solution
pose = get_mediapipe_pose()


if 'download' not in st.session_state:
    st.session_state['download'] = False

if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None


output_video_file = f'output_live.flv'

exercise_option = st.selectbox('Current Exercise:', ['Predict','Bicep curl', 'Squat', 'Lateral Raise'], key='exercise')

def video_frame_callback(frame: av.VideoFrame):
    frame = frame.to_ndarray(format="rgb24")  # Decode and get RGB frame
    #frame, _ = live_process_frame_curl.process(frame, pose)  # Process frame
    if exercise_option == 'Bicep curl':
        frame, _ = live_process_frame_curl.process(frame, pose)
    elif exercise_option == 'Squat':
        frame, _ = live_process_frame_squat.process(frame, pose)
    elif exercise_option == 'Lateral Raise':
        frame, _ = live_process_frame_raise.process(frame, pose)
    else:
        frame = predict_image(frame)
    
    return av.VideoFrame.from_ndarray(frame, format="rgb24")  # Encode and return BGR frame




def out_recorder_factory() -> MediaRecorder:
        return MediaRecorder(output_video_file)


ctx = webrtc_streamer(
                        key="Squats-pose-analysis",
                        video_frame_callback=video_frame_callback,
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Add this config
                        media_stream_constraints={"video": {"width": {'min':480, 'ideal':480}}, "audio": False},
                        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False),
                        out_recorder_factory=out_recorder_factory
                    )


download_button = st.empty()

if os.path.exists(output_video_file):
    with open(output_video_file, 'rb') as op_vid:
        download = download_button.download_button('Download Video', data = op_vid, file_name='output_live.flv')

        if download:
            st.session_state['download'] = True



if os.path.exists(output_video_file) and st.session_state['download']:
    os.remove(output_video_file)
    st.session_state['download'] = False
    download_button.empty()


    


