

# Get thresholds for beginner mode
def get_thresholds_beginner():

    _ANGLE_HIP_KNEE_VERT = {
                            'NORMAL' : (0,  32),
                            'TRANS'  : (35, 65),
                            'PASS'   : (70, 95)
    }
    
    _ANGLE_ELBOW_WRIST_VERT = {
                            'NORMAL' : (180,  121),
                            'TRANS'  : (120, 46),
                            'PASS'   : (45, 0)
    }   
    _ANGLE_SHOULDER_ELBOW_VERT = {
                            'NORMAL' : (180,  150),
                            'TRANS'  : (149, 106),
                            'PASS'   : (105, 60)
    }

        
    thresholds = {
    "SQUAT": {
                    'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,

                    'HIP_THRESH'   : [10, 50],
                    'ANKLE_THRESH' : 45,
                    'KNEE_THRESH'  : [50, 70, 95],

                    'OFFSET_THRESH'    : 35.0,
                    'INACTIVE_THRESH'  : 15.0,

                    'CNT_FRAME_THRESH' : 50}
    ,
    "CURL" : {
                        'ELBOW_WRIST_VERT': _ANGLE_ELBOW_WRIST_VERT,

                        'SHOULDER_THRESH'   : 145,
                        # 'WRIST_THRESH'   : 45,
                        # 'ELBOW_THRESH': [50, 70, 95],
                        'ELBOW_THRESH': [150, 0, -20],
                        'OFFSET_THRESH'    : 35.0,
                        'INACTIVE_THRESH'  : 15.0,

                        'CNT_FRAME_THRESH' : 50
                            
                    },
    
    "RAISES" : {
                        'SHOULDER_ELBOW_VERT': _ANGLE_SHOULDER_ELBOW_VERT,

                        'SHOULDER_THRESH'   : 130,
                        # 'WRIST_THRESH'   : 45,
                        # 'ELBOW_THRESH': [50, 70, 95],
                        'ELBOW_THRESH': [150, 0, -20],

                        'OFFSET_THRESH'    : 70.0,
                        'INACTIVE_THRESH'  : 15.0,

                        'CNT_FRAME_THRESH' : 50,

                        'DIFF_THRESH' : 15, 

                        'HUNCH_SHOULDER_THRESH': 1.6,

                        'IMPROPER_START_FORM': 30
    }
    }

    return thresholds



# Get thresholds for pro mode
def get_thresholds_pro():

    _ANGLE_HIP_KNEE_VERT = {
                            'NORMAL' : (0,  32),
                            'TRANS'  : (35, 65),
                            'PASS'   : (80, 95)
                           }    

        
    thresholds = {
                    'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,

                    'HIP_THRESH'   : [15, 50],
                    'ANKLE_THRESH' : 30,
                    'KNEE_THRESH'  : [50, 80, 95],

                    'OFFSET_THRESH'    : 35.0,
                    'INACTIVE_THRESH'  : 15.0,

                    'CNT_FRAME_THRESH' : 50
                            
                 }
                 
    return thresholds