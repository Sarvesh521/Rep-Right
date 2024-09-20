import time
import cv2
import numpy as np
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line


class ProcessFrame:
    def __init__(self, thresholds, flip_frame = False):
        
        # Set if frame should be flipped or not.
        self.flip_frame = flip_frame

        # self.thresholds
        self.thresholds = thresholds["CURL"]

        # Font type.
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # line type
        self.linetype = cv2.LINE_AA

        # set radius to draw arc
        self.radius = 20

        # Colors in BGR format.
        self.COLORS = {
                        'blue'       : (0, 127, 255),
                        'red'        : (255, 50, 50),
                        'green'      : (0, 255, 127),
                        'light_green': (100, 233, 127),
                        'yellow'     : (255, 255, 0),
                        'magenta'    : (255, 0, 255),
                        'white'      : (255,255,255),
                        'cyan'       : (0, 255, 255),
                        'light_blue' : (102, 204, 255)
                      }



        # Dictionary to maintain the various landmark features.
        self.dict_features = {}
        self.left_features = {
                                'shoulder': 11,
                                'elbow'   : 13,
                                'wrist'   : 15,                    
                                'hip'     : 23,
                                'knee'    : 25,
                                'ankle'   : 27,
                                'foot'    : 31
                             }

        self.right_features = {
                                'shoulder': 12,
                                'elbow'   : 14,
                                'wrist'   : 16,
                                'hip'     : 24,
                                'knee'    : 26,
                                'ankle'   : 28,
                                'foot'    : 32
                              }

        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0

        
        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            'state_seq': [],

            'start_inactive_time': time.perf_counter(),
            'start_inactive_time_front': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            'INACTIVE_TIME_FRONT': 0.0,

            # 0 --> Bend Backwards, 1 --> Bend Forward, 2 --> Keep shin straight, 3 --> Deep squat
            'DISPLAY_TEXT' : np.full((2,), False),
            'COUNT_FRAMES' : np.zeros((2,), dtype=np.int64),

            'CURL_MORE': False,

            'INCORRECT_POSTURE': False,

            'prev_state': None,
            'curr_state':None,

            'REP_COUNT': 0,
            'IMPROPER_REP':0
            
        }
        
        self.FEEDBACK_ID_MAP = {
                                0: ('STRAIGHTEN BODY', 215, (0, 153, 255)),
                                1: ('LOWER YOUR ELBOW', 215, (0, 153, 255))
                               }

        


    def _get_state(self, elbow_angle):
        
        elbow = None        

        if self.thresholds['ELBOW_WRIST_VERT']['NORMAL'][1] <= elbow_angle <= self.thresholds['ELBOW_WRIST_VERT']['NORMAL'][0]:
            elbow = 1
        elif self.thresholds['ELBOW_WRIST_VERT']['TRANS'][1] <= elbow_angle <= self.thresholds['ELBOW_WRIST_VERT']['TRANS'][0]:
            elbow = 2
        elif self.thresholds['ELBOW_WRIST_VERT']['PASS'][1] <= elbow_angle <= self.thresholds['ELBOW_WRIST_VERT']['PASS'][0]:
            elbow = 3

        return f's{elbow}' if elbow else None



    
    def _update_state_sequence(self, state):

        if state == 's2':
            if (('s3' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2'))==0) or \
                    (('s3' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2')==1)):
                        self.state_tracker['state_seq'].append(state)
            

        elif state == 's3':
            if (state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']: 
                self.state_tracker['state_seq'].append(state)

            


    def _show_feedback(self, frame, c_frame, dict_maps, curl_more_disp):


        if curl_more_disp:
            draw_text(
                    frame, 
                    'CURL MORE', 
                    pos=(30, 80),
                    text_color=(0, 0, 0),
                    font_scale=0.6,
                    text_color_bg=(255, 255, 0)
                )  

        for idx in np.where(c_frame)[0]:
            draw_text(
                    frame, 
                    dict_maps[idx][0], 
                    pos=(30, dict_maps[idx][1]),
                    text_color=(255, 255, 230),
                    font_scale=0.6,
                    text_color_bg=dict_maps[idx][2]
                )

        return frame



    def process(self, frame: np.array, pose):
        play_sound = None
       

        frame_height, frame_width, _ = frame.shape

        # Process the image.
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            with open('offset_angle.txt', 'a') as f:
                    f.write("HELLOHELLOWDFHJK")
            ps_lm = keypoints.pose_landmarks

            nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            left_shoulder_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
            right_shoulder_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(left_shoulder_coord, right_shoulder_coord, nose_coord)

            if offset_angle > self.thresholds['OFFSET_THRESH']:
                
                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['REP_COUNT'] = 0
                    self.state_tracker['IMPROPER_REP'] = 0
                    display_inactivity = True

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shoulder_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shoulder_coord, 7, self.COLORS['magenta'], -1)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    # cv2.putText(frame, 'Resetting SQUAT_COUNT due to inactivity!!!', (10, frame_height - 90), 
                    #             self.font, 0.5, self.COLORS['blue'], 2, lineType=self.linetype)
                    play_sound = 'reset_counters'
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['REP_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )  
                

                draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_REP']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                    
                )  
                
                
                draw_text(
                    frame, 
                    'CAMERA NOT ALIGNED PROPERLY!!!!', 
                    pos=(30, frame_height-60),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                ) 
                
                
                draw_text(
                    frame, 
                    'OFFSET ANGLE: '+str(offset_angle), 
                    pos=(30, frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                ) 

                # Reset inactive times for side view.
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
                self.state_tracker['prev_state'] =  None
                self.state_tracker['curr_state'] = None
            
            # Camera is aligned properly.
            else:
                with open('offset_angle.txt', 'a') as f:
                    f.write("HELLO1")
                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter()


                dist_l_sh_elb = abs(left_elbow_coord[1]- left_shoulder_coord[1])
                dist_r_sh_elb = abs(right_elbow_coord[1] - right_shoulder_coord)[1]

                shoulder_coord = None
                elbow_coord = None
                wrist_coord = None
                hip_coord = None
                knee_coord = None
                ankle_coord = None
                foot_coord = None

                if dist_l_sh_elb > dist_r_sh_elb:
                    shoulder_coord = left_shoulder_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hip_coord = left_hip_coord
                    knee_coord = left_knee_coord
                    ankle_coord = left_ankle_coord
                    foot_coord = left_foot_coord

                    multiplier = -1
                                     
                
                else:
                    shoulder_coord = right_shoulder_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hip_coord = right_hip_coord
                    knee_coord = right_knee_coord
                    ankle_coord = right_ankle_coord
                    foot_coord = right_foot_coord

                    multiplier = 1
                    

                # ------------------- Verical Angle calculation --------------
                with open('offset_angle.txt', 'a') as f:
                    f.write("HELLO2")
                shoulder_vertical_angle = find_angle(elbow_coord, np.array([shoulder_coord[0], 0]), shoulder_coord)
                cv2.ellipse(frame, shoulder_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*shoulder_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                draw_dotted_line(frame, shoulder_coord, start=shoulder_coord[1]-80, end=shoulder_coord[1]+20, line_color=self.COLORS['blue'])




                elbow_vertical_angle = find_angle(wrist_coord, np.array([elbow_coord[0], 0]), elbow_coord)
                cv2.ellipse(frame, wrist_coord, (20, 20), 
                            angle = 0, startAngle = -90, endAngle = -90-multiplier*elbow_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3,  lineType = self.linetype)

                draw_dotted_line(frame, elbow_coord, start=elbow_coord[1]-50, end=elbow_coord[1]+20, line_color=self.COLORS['blue'])
                with open('offset_angle.txt', 'a') as f:
                    f.write("HELLO3")

                #commented this for now, will work on it when handling within wrist

                # ankle_vertical_angle = find_angle(knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
                # cv2.ellipse(frame, ankle_coord, (30, 30),
                #             angle = 0, startAngle = -90, endAngle = -90 + multiplier*ankle_vertical_angle,
                #             color = self.COLORS['white'], thickness = 3,  lineType=self.linetype)

                # draw_dotted_line(frame, ankle_coord, start=ankle_coord[1]-50, end=ankle_coord[1]+20, line_color=self.COLORS['blue'])

                # ------------------------------------------------------------
        
                
                # Join landmarks.
                cv2.line(frame, shoulder_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                # cv2.line(frame, shoulder_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                # cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                # cv2.line(frame, ankle_coord, knee_coord,self.COLORS['light_blue'], 4,  lineType=self.linetype)
                # cv2.line(frame, ankle_coord, foot_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                
                # Plot landmark points
                cv2.circle(frame, shoulder_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                # cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                # cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                # cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                # cv2.circle(frame, foot_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                with open('offset_angle.txt', 'a') as f:
                    f.write("HELLO4")
                
                
                current_state = self._get_state(int(elbow_vertical_angle))
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)



                # -------------------------------------- COMPUTE COUNTERS --------------------------------------
                print(current_state)
                if current_state == 's1':

                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['REP_COUNT']+=1
                        play_sound = str(self.state_tracker['REP_COUNT'])
                        
                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq'])==1:
                        self.state_tracker['IMPROPER_REP']+=1
                        play_sound = 'incorrect'

                    elif self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['IMPROPER_REP']+=1
                        play_sound = 'incorrect'
                        
                    
                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False


                # ----------------------------------------------------------------------------------------------------

                    with open('offset_angle.txt', 'a') as f:
                        f.write("HELLO5")


                # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

                else:
                    if shoulder_vertical_angle < self.thresholds['SHOULDER_THRESH']:
                        self.state_tracker['DISPLAY_TEXT'][1] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True                        
                                        
                    
                    if self.state_tracker['state_seq'].count('s2')==1:
                        self.state_tracker['CURL_MORE'] = True
                        

                    
                    # if (ankle_vertical_angle > self.thresholds['ANKLE_THRESH']):
                    #     self.state_tracker['DISPLAY_TEXT'][2] = True
                    #     self.state_tracker['INCORRECT_POSTURE'] = True
                    with open('offset_angle.txt', 'a') as f:
                        f.write("HELLO6")


                # ----------------------------------------------------------------------------------------------------


                
                
                # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

                display_inactivity = False
                
                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
                    self.state_tracker['start_inactive_time'] = end_time

                    if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                        self.state_tracker['REP_COUNT'] = 0
                        self.state_tracker['IMPROPER_REP'] = 0
                        display_inactivity = True

                
                else:
                    
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                # -------------------------------------------------------------------------------------------------------
                with open('offset_angle.txt', 'a') as f:
                    f.write("HELLO7")
                shoulder_text_coord_x = shoulder_coord[0] + 10
                elbow_text_coord_x = elbow_coord[0] + 15
                #ankle_text_coord_x = ankle_coord[0] + 10

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    shoulder_text_coord_x = frame_width - shoulder_coord[0] + 10
                    elbow_text_coord_x = frame_width - elbow_coord[0] + 15
                    #ankle_text_coord_x = frame_width - ankle_coord[0] + 10
               
                if 's3' in self.state_tracker['state_seq'] or current_state == 's1':
                    self.state_tracker['CURL_MORE'] = False
                
                with open('offset_angle.txt', 'a') as f:
                    f.write("HELLO8")

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']]+=1

                frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, self.state_tracker['CURL_MORE'])

                if display_inactivity:
                    # cv2.putText(frame, 'Resetting COUNTERS due to inactivity!!!', (10, frame_height - 20), self.font, 0.5, self.COLORS['blue'], 2, lineType=self.linetype)
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0
                
                cv2.putText(frame, str(int(shoulder_vertical_angle)), (shoulder_text_coord_x, shoulder_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(elbow_vertical_angle)), (elbow_text_coord_x, elbow_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                #cv2.putText(frame, str(int(ankle_vertical_angle)), (ankle_text_coord_x, ankle_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)

                with open('offset_angle.txt', 'a') as f:
                    f.write("HELLO9")
                draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['REP_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )  

                draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_REP']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                    
                )  
                with open('offset_angle.txt', 'a') as f:
                    f.write("HELLO10")
                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0    
                self.state_tracker['prev_state'] = current_state
                                  
        else:

            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['REP_COUNT'] = 0
                self.state_tracker['IMPROPER_REP'] = 0
                # cv2.putText(frame, 'Resetting SQUAT_COUNT due to inactivity!!!', (10, frame_height - 25), self.font, 0.7, self.COLORS['blue'], 2)
                display_inactivity = True

            self.state_tracker['start_inactive_time'] = end_time

            draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['REP_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )  
                

            draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_REP']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                    
                )  

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
            
            
            # Reset all other state variables
            
            self.state_tracker['prev_state'] =  None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full((3,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros((3,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()

  
        return frame, play_sound