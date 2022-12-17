# Real-time Action Recognition from Scratch


This is a quick project showing the capabilities of using machine learning for action recognition software. In this project you will see code that is used to develop a machine learning computer vision model to detect certain actions in real-time. These actions consist of: a "neutral" state, smiling, waving, and dropping my head to signal a sad emotion. This project uses renown open-source libraries such as Mediapipe and Scikit-learn, as well as many others.

In this specific project, we use the Mediapipe library. I chose to use this library over others (Openpose, Pytorch, BlazePose, etc) because of the more detailed landmarks available as well as the fast inferencing times to allow for rapid prototyping, as Mediapipe uses GPU acceleration and multi-threading techniques for you. Mediapipe's Holistic solution (link: https://google.github.io/mediapipe/solutions/holistic). I chose to use this solution over some of Mediapipe's other solutions (such as their Pose solution) due to the larger number of landmarks, especially in the facial region. With these additional landmarks, more specific facial actions, such as smiling, were easier to learn for the models. The holistic model also utilizes the landmarks in the pose solution, providing the enablement of various applications and actions to be learned. 

![image](https://user-images.githubusercontent.com/70036220/208226088-e031f899-607c-4061-aa12-e3964a91e8fd.png)


Building this model from scratch provided a few benefits compared to grabbing a pre-trained model and public dataset. This project is designed to help understand the ins and outs of pose estimation/action recognition. As a result, we must build our dataset, our model, and our method of deployment. As you look through the code, you will see the use of OpenCV to use our webcam to extract the landmark coordinates from a live video feed and export these landmarks to a .csv file. With this, we are able to use the Pandas library to split our data into our target and feature variables, and proceed to use Scikit-learn to import the ML algorithms we plan to use and train these models. You will see the implementation of 4 algorithms in this project: Random Forest Classifier, Gradient Boosting Classifier, Ridge Classifier, and Logistic Regression. This provides the opportunity to compare amongst several algorithms and choose the best solution. Lastly, we are using the Pickle library to save/export and load our trained models. 


## Video Footage of our Logistic Regression Model Running in Real-Time

#### Video #1: Initially Trained Logistic Regression Model

https://user-images.githubusercontent.com/70036220/208224872-35ce0a37-0642-489c-9bff-4452c3b12dd6.mp4

This video demontrates a logistic regression model performing action recognition. The model appears to be performing quite well for the small amount of data that is being used to train the model..... Or as we thought, see video #2. 

#### Video #2: Problems with our Initially Trained Logistic Regression Model

https://user-images.githubusercontent.com/70036220/208225169-0141056b-cda2-4516-b00b-395bbb494f4d.mp4

In this video, I have removed all of the landmarks and connections to provide a more clearer frame. In this video you will see several issues with this model. One of these issues arises with the tilting of my head. As you see me smile and proceed to move my head from left to right and vice versa, the model begins to fail in its detections. Same thing happens as I begin to wave. Initially the model detects the waving action, but as I begin to wave closer to the front of my body, the mdoel begins to fail in its detections again. 

Solution: These faults in the model performance are a result of inadequate data. Thus, we can go back to our landmark extracting steps, and provide our model more data to better learn and generalize. A further trained model is showcased in video #3, and video #4 is the further trained model without the landmarks and connections to provide a more clear picture of how the model is improving. 

#### Video #3: Further Trained Logistic Regression Model

https://user-images.githubusercontent.com/70036220/208225647-034fd696-0eb6-429c-b5e2-479dbd0f1770.mp4

#### Video #4: Our Further Trained Logistic Regression Model Without the Landmarks and Connections

https://user-images.githubusercontent.com/70036220/208225708-61fe3231-2325-446e-82cb-0a3549958167.mp4

#### What's Next?

This is just the tip of the iceberg for action recognition, pose estimation, facial recognition, as well as a whole heap of other recognition and detection capabilities. For this specific project, you can continue to add more actions to be detected and continue to add more data to further improve the weaknesses of the model. 

