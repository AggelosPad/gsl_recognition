#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import random
from unittest import result
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
import Levenshtein
from collections import Counter
import streamlit as st
from PIL import Image

# Dictionary init
array = []
txt_words = 3321
word_indexes = []
words_to_sign = []


#Final 
global levenshtein_score
levenshtein_score = 0
global max_levenshtein_score
max_levenshtein_score =0
global count_lev
count_lev=0

#Sequence initialization
global signed_word
signed_word = ''
global count
count = 0
global words_answered
words_answered = 0
global correct_words
global total_words
total_words = 3
correct_words = 0
global check_time 
check_time = 0
global  spellit,signed,st_title,stframe,button1,message,st_image
global start_time


start_time =0
st_title = st.empty()
stframe = st.empty()
spellit = st.empty()
signed = st.empty()
st_result = st.empty()
st_final = st.empty()
st_leven = st.empty()
st_message =st.empty()
button1 = st.empty()
message = st.empty()
st_image = st.empty()

st_title.latex('\centering')
st_title.latex('Greek\:Sing \:Language \:Fingerspelling\: Recognition')

col1, col2, col3 = st.columns(3)


image2 = Image.open('logos.png')

st_image=st.image(image2)

for i in range(0,total_words):
    temp = (random.randint(0, txt_words-1))
    word_indexes.append(temp)
try: 
    with open('dict.txt','r', encoding = 'utf-8') as file:
        
        # reading each line    
        for line in file:
    
            # reading each word        
            for word in line.split():
                
                array.append(word)

        print("Dictionary loaded: dict.txt")      
except:
    print("Could not load dictionary")          

for num in word_indexes:
    words_to_sign.append(array[num])

print(words_to_sign)


global w_index
w_index = 0
global sign_this_word

sign_this_word = words_to_sign[w_index]

print(sign_this_word)

spellit.latex('\centering')
spellit.latex(r'Spell:\: ' + sign_this_word)

letter = ''
predicted_class_name = None
global letter_predict
letter_predict = []


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def next_word():
    global result
    global sign_this_word
    global signed_word
    global w_index
    global letter
    global check_time
    global words_answered
    global spellit
    global signed
    global st_result
    global correct_words
    global count_lev
    global levenshtein_score
    global max_levenshtein_score
    global total_words
    global st_leven

    if result == 'CORRECT':
        correct_words = correct_words +1
        

    result=''
    signed_word = ''
    letter = ''
    words_answered = words_answered + 1
    #print(w_index)
    if w_index < total_words-1:
        w_index = w_index + 1
        sign_this_word = words_to_sign[w_index]
    check_time=0
    count_lev =0

    if words_answered < total_words:
        spellit.latex('\centering')
        spellit.latex(r'Spell:\: ' + sign_this_word)

        signed.latex(('\centering'))
        signed.latex(r'Your\: word:\: ' + signed_word)
        st_leven.latex('')


    elif words_answered == total_words:
        spellit.latex('\centering')
        spellit.latex('')

        signed.latex(('\centering'))
        signed.latex("")

        st_final.latex('\centering')
        
        correct_words_percentage = int(correct_words/total_words * 100)
        correct_words_percentage_str = '\:[' + str(correct_words_percentage)+'\%]'

        print("Final\: score:\:" + str(correct_words)+'/'+str(total_words) + correct_words_percentage_str)
        st_final.latex("Final\: score:\:" + str(correct_words)+'/'+str(total_words) + correct_words_percentage_str)

        st_leven.latex('\centering')
        
        max_levenshtein_score_perc = int(((max_levenshtein_score-levenshtein_score)/max_levenshtein_score) *100 ) 
        print(max_levenshtein_score_perc)
        st_leven.latex("Total \:Levenshtein\: distance: \: " + str(max_levenshtein_score-levenshtein_score)+'/'+str(max_levenshtein_score) + '\:[' + str(max_levenshtein_score_perc)+'\%]')
        
   
    message.latex(" ")
    st_result.latex('')
    

def main():
    # Arguments  
    args = get_args()

    #Global variables
    global sign_this_word
    global signed_word
    global stframe
    global start_time
    global end_time
    global start_time
    global check_time
    
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    #st_title = st.empty()
    #stframe = st.empty()

    

    # Camera  
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Mediapipe loading
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Label reading
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
       
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS 
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history 
    finger_gesture_history = deque(maxlen=history_length)

    #  
    mode = 0

    #streamlit init 
    
  
    while True:
        fps = cvFpsCalc.get()

        # Key processing
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        mode = 0
        number = -1
        #number, mode = select_mode(key, mode)

        # Camera capture 
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        #Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True


        end_time = time.time()
        #print(end_time)
        if (end_time > start_time + 3) and check_time == 1:
            next_word()

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Calculation of circumscribed rectangle
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                #Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Learning data storage

                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # 指差しサイン
                    point_history.append(landmark_list[8])  # 人差指座標
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # drawing
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                #stframe.image(debug_image)

                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        #debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        #Streamlit video
        stframe.image(image)

        # Window title
        #cv.imshow('Greek Sign Language Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Key Point
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number+20, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    # Connection line
    if len(landmark_point) > 0:
        # 親指
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm

        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Point

    for index, landmark in enumerate(landmark_point):
        if index == 0:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    #Global variables 
    global signed_word
    global sign_this_word
    global w_index
    global letter_predict
    global count
    global words_answered
    global correct_words
    global total_words
    global spellit
    global signed
    global st_result
    global button1
    global check_time
    global start_time
    global message
    global max_levenshtein_score
    global levenshtein_score
    global count_lev
    global result
    global st_image
    global st_final
    global st_leven
    spellit.latex('\centering')
    spellit.latex(r'Spell:\: ' + sign_this_word)
    
    signed.latex(('\centering'))
    signed.latex(r'Your\: word:\: ' + signed_word)



    info_text = handedness.classification[0].label[0:]
    letter = ''

    frame_count = 24

    if hand_sign_text != "":
        
        letter_predict.append(hand_sign_text) 
        
        if len(letter_predict) > 12:  # arithmos frames gia thn epilogh tou grammatos
            letter = most_frequent(letter_predict)
            print("Letter: " + str(letter))
            #print(letter_predict)
            letter_predict = []
       

        if words_answered == total_words:
            
            #print("Score: " + str(correct_words/words_answered))
            signed.latex('')
            spellit.latex('')
            st_result.latex('')
            st_final.latex('\centering')
            correct_words_percentage = int(correct_words/total_words * 100)
            correct_words_percentage_str = '\:[' + str(correct_words_percentage)+'\%]'
            st_final.latex("Final\: score:\:" + str(correct_words)+'/'+str(total_words) + correct_words_percentage_str)

            st_leven.latex('\centering')
            max_levenshtein_score_perc = int(((max_levenshtein_score-levenshtein_score)/max_levenshtein_score) *100 ) 
            print(max_levenshtein_score_perc)
            st_leven.latex("Total \:Levenshtein\: distance: \: " + str(max_levenshtein_score-levenshtein_score)+'/'+str(max_levenshtein_score) + '\:[' + str(max_levenshtein_score_perc)+'\%]')

            pass    

        if signed_word == '':
            signed_word = signed_word + letter
            
            if signed_word!='':    
                print("Word to sign: " + str(sign_this_word) +" Signed word: " + str(signed_word))

           
        if len(signed_word) == len(sign_this_word) and count_lev == 0:
            
                        
            levenshtein_score = levenshtein_score + Levenshtein.distance(signed_word,sign_this_word)
            curr_leven_score = Levenshtein.distance(signed_word,sign_this_word) # score kathe leksis 3exwrista
            max_levenshtein_score =  max_levenshtein_score + len(sign_this_word)
            count_lev = 1
            
            st_leven.latex('\centering')
            curr_leven_score_perc = int((len(sign_this_word)-curr_leven_score)/len(sign_this_word) * 100)
            st_leven.latex(str(sign_this_word) + "\: Levenshtein \: distance: \:" + str(len(sign_this_word)-curr_leven_score)+'/'+str(len(sign_this_word)) + '\:[' + str(curr_leven_score_perc) + '\%]')


        if len(signed_word)>=1 and len(signed_word) <= len(sign_this_word):
            if signed_word[-1]!= letter:
                signed_word = signed_word + letter
                
                #print("Word to sign: " + str(sign_this_word) +" Signed word: " + str(signed_word))
                length = len(signed_word)

             


                if length >= len(sign_this_word):
                    
                    

                    #Periptwsh lathous apanthshs
                    if signed_word!=sign_this_word:
                        if check_time == 0:
                            
                            start_time = time.time()
                            #print(str(start_time)+'start_time')
                            check_time = 1
                        
                        result = 'WRONG'
                        st_result.latex('\centering')
                        st_result.latex(result)

                        message.latex(('\centering'))
                        correct_words_percentage = int(correct_words/(words_answered+1) * 100)
                        correct_words_percentage_str = '\:[' + str(correct_words_percentage)+'\%]'
                        st_final.latex('\centering')
                        st_final.latex("Current\: score:\:" + str(correct_words)+'/'+str(words_answered+1) + correct_words_percentage_str)
                        

                        message.latex("Next \:word \:will \:be \:up \:in\:3\:seconds")
                        #print(result)
                        
                        
                        i=0
                        if (i==1):
                        
                            result=''
                            signed_word = ''
                            letter = ''
                            words_answered = words_answered + 1
                            #print(w_index)
                            if w_index < total_words-1:
                                w_index = w_index + 1
                                sign_this_word = words_to_sign[w_index]
                            check_time=0
                    

                    #Periptwsh swsths apanthshs
                    if signed_word == sign_this_word:
                        
                        if check_time == 0:
                            
                            start_time = time.time()
                            #print(str(start_time)+'start_time')
                            check_time = 1

                        result = 'CORRECT'
                        st_result.latex('\centering')
                        st_result.latex(result)

                        correct_words_percentage = int((correct_words+1)/(words_answered+1) * 100)
                        correct_words_percentage_str = '\:[' + str(correct_words_percentage)+'\%]'
                        st_final.latex('\centering')
                        st_final.latex("Current\: score:\:" + str(correct_words+1)+'/'+str(words_answered+1) + correct_words_percentage_str)
                        
                        

                        message.latex(('\centering'))
                        message.latex("Next \:word \:will \:be \:up \:in\:3\:seconds")

                        #print(result)
                        i=0
                        if (i==1):
                            signed_word = ''
                            result = ''
                            letter = ''
                            words_answered = words_answered + 1
                            correct_words = correct_words + 1
                            if w_index < total_words:
                                w_index = w_index + 1
                                sign_this_word = words_to_sign[w_index]
                        
        
        
       
        #print(signed_word)
        info_text = info_text + ':' + hand_sign_text
   
    
    return image

def most_frequent(List):
    dict = {}
    count, itm = 0, ''
    for item in reversed(List):
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count :
            count, itm = dict[item], item
    return(itm)


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number+20), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
