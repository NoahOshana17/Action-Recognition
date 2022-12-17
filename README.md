# Real-time Action Recognition from Scratch


This is a quick project showing the capabilities of using machine learning for action recognition software. In this project you will see code that is used to develop a machine learning computer vision model to detect certain actions in real-time. These actions consist of: a "neutral" state, smiling, waving, and dropping my head to signal a sad emotion. This project uses renown open-source libraries such as Mediapipe and Scikit-learn, as well as many others.

In this specific project, we use Mediapipe's Holistic solution (link: https://google.github.io/mediapipe/solutions/holistic). I chose to use this solution over some of Mediapipe's other solutions (such as their Pose solution) due to the larger number of landmarks, especially in the facial region. With these additional landmarks, more specific facial actions, such as smiling, were easier to learn for the models. The holistic model also utilizes the landmarks in the pose solution, providing the enablement of various applications and actions to be learned. 

Building this model from scratch provided a few benefits compared to grabbing a pre-trained model and public dataset. This project is designed to help understand the ins and outs of pose estimation/action recognition. As a result, we must build our dataset, our model, and our method of deployment. As you look through the code, you will see the use of OpenCV to use our webcam to extract the landmark coordinates from a live video feed and export these landmarks to a .csv file. With this, we are able to use the Pandas library to split our data into our target and feature variables, and proceed to use Scikit-learn to import the ML algorithms we plan to use and train these models. You will see the implementation of 4 algorithms in this project: Random Forest Classifier, Gradient Boosting Classifier, Ridge Classifier, and Logistic Regression. This provides the opportunity to compare amongst several algorithms and choose the best solution. Lastly, we are using the Pickle library to save/export and load our trained models. 


## Video Footage of our Logistic Regression Model Running in Real-Time

#### Video #1: Initially Trained Logistic Regression Model

https://user-images.githubusercontent.com/70036220/208224872-35ce0a37-0642-489c-9bff-4452c3b12dd6.mp4

This video demontrates a logistic regression model performing action recognition. The model appears to be performing quite well for the small amount of data that is being used to train the model..... Or as we thought, see video #2. 

#### Video #2: Problems with our Initially Trained Logistic Regression Model

https://user-images.githubusercontent.com/70036220/208225169-0141056b-cda2-4516-b00b-395bbb494f4d.mp4

