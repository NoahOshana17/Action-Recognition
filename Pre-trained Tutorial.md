
## How to: Use a provided trained model, load the model, and make detections using your webcam

Walking through the jupyter notebook provided step by step will allow you to create your dataset and train models from scratch. However, loading and testing the models provided in this repository can be acconmplished by running only a few notebook cells, as shown below.


#### Step 1: Import Dependencies. This should be the first cell in the ActionRec.ipynb

```python
import mediapipe as mp
import pandas as pd
import cv2
import os
import numpy as np
import csv
from sklearn.model_selection import train_test_split
```

#### Step 2: Declaring our Mediapipe solutions. This should be cell #2 in ActionRec.ipynb

```python
mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic 
```
#### Step 3: Import Pickle library. This should be cell #21 at the end of the training phase in ActionRec.ipynb

```python
from sklearn.metrics import accuracy_score
import pickle
```

#### Step 4: Load the desired model. This should be cell #24 at the "Loading our model" section in ActionRec.ipynb

```python
with open('Action_Recognition_lr_2.pkl', 'rb') as f:
    model = pickle.load(f)
```
Note: Here you can change 'Action_Recognition_lr_2.pkl' to the name of any of the other models provided, or a model you trained yourself. 


#### Step 5: Making the detections using webcam. This should be cell #26 under the "Using webcam for real-time detections" section in ActionRec.ipynb

```python
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = holistic.process(image)


        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )

        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        
        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            
            row = pose_row + face_row
            
#             row.insert(0, class_name)
            
#             with open('coords.csv', mode='a', newline='') as f:
#                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                 csv_writer.writerow(row)

            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            #print(body_language_class, body_language_prob)
            
            
#             coords = tuple(np.multiply(
#                             np.array(
#                                     ( results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
#                                       results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
#                             , [640, 480]).astype(int))
            
#             cv2.rectangle(image, 
#                           (coords[0], coords[1]+5), 
#                           (coords[0]+len(body_language_class)*20, coords[1]-30), 
#                           (245, 117, 16), -1)
#             cv2.putText(image, body_language_class, coords, 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            
            
        except:
            pass

        cv2.imshow('Action Recognition', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```


Note: In ActionRec.ipynb, the very last cell under the section "Same as previous cell, except here we commented out the landmarks and will only show you the prediction and accuracy" can be used in replace of step 5, depending on preferences for the visualizations.
