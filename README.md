# FaceRecognition
Its a Training Project on face recognition using Machine Learning and Python+OpenCV libraries. The project is made on Anaconda Jupyter Notebook
Our aim in this project is to develop a fully functional face recognition model using the concepts of Machine Learning and Computer Vision (OpenCV) and compare their performance for identical datasets.
In the code using OpenCV I have used the cascades of Local Binary Patterns (LBPH classifiers).
In the code using Machine Learning I have used Support Vector Machine (SVM) and Principal Component Analysis (PCA).
The ‘Trainer’ codes take as input a Celebrity Face dataset with about 20 samples per class. The information gathered from this training is then tested by the ‘Testing’ code that takes a sample outside of the training data to predict the face of the Celebrity in the test sample.

## Implementation
The implementation of the project can be divided into 4 basic stages namely:

•	Dataset gathering

•	Model generation

•	Model Training

•	Model Testing

Each of these stages can be explained in greater detail as followed:

4.1 Dataset Gathering

The first stage of the project is Dataset Gathering. In this step I collected the required dataset images for all the respective classes to prepare a set of samples that can later be used in the training and testing of the models created in the following steps.
I used the ‘5 Celebrity Face’ Dataset from Kaggle which is available on the website. I modified this dataset by adding and removing some sample images from the classes and separating out the testing samples from the training samples. The dataset contains approximately 20 sample images for each class. The class ID is set as the name of the celebrity which would be displayed as the output of the prediction as well.

One image for each celebrity class is used as the testing data for testing the model.

4.2 Model Generation

Followed by gathering the dataset I prepared the codes for both models. I used references from various available models on the internet and notes from the training to prepare a model that fits the dataset and its attributes.
To avoid overfitting of the model I left values for LBPH classifier attributes at default. Also in the Machine Learning model I have set the no. of components attribute of the PCA feature extractor at 200 after obtaining the maximum number of independent features that were required for a good prediction from the dataset. Rest of the attributes for PCA and SVM were left to default.

4.3 Model Training

I trained the model using the training dataset samples. 
The LBPH classifier model goes through each and every image in the dataset and prepares its corresponding histogram and then trains itself with a unique histogram for each class. Any input image whose histogram is close to any one of the classes would be predicted to belong to that class. The data gathered from this training is saved in a Trainer.yml file. This file can be read by any subsequent LBPH Classifier program to refer and predict faces for input test samples.
The Machine Learning model that uses PCA and SVM first prepares a numerical dataset for all the training samples and stores them in a numpy array. This array is then reshaped into a one dimensional numpy array. The images, in the form of numerical data are then used further to perform Machine Learning techniques and enable face recognition.
The array is passed through the PCA object to obtain the maximum number of useful features in the data. The graphical representation of this evaluation shows that this value can b set 200. Therefore, the ‘n_components’ attribute of the PCA object is set to 200 for further operation.
After this these useful attributes are passed through the SVM classifier to learn information from these features of the training images. The classifier and the PCA object after training are saved in .pkl (pickle file) format. This file can be imported in any testing code of the Machine Learning model to avoid having to retrain the model every time a prediction needs to be made. 


4.4 Model Testing

I then tested these models that have been trained on the same training dataset using the same testing input samples.
Both the models are tested separately. The test data is input into the models and the model processes these images to check for the features that it has learnt to extract and predict from. The label or class ID (here, the name of the celebrity) is returned by the model and the output is displayed on the screen and shown in the subsequent section. 
 
## Conclusion

The results obtained from the testing of both models are depicted in the images shown below.
5.1	Output: Machine Learning Model
The output can be represented in the following tabular form:

S.NO.	CORRECT LABEL	PREDICTED LABEL

1.		Ben Afflek	Face not detected correctly, Alton John
2.		Jerry Seinfeld	Jerry Seinfeld
3.		Madonna	Ben Afflek
4.		Elton John	Elton John
5.		Mindy Kaling	Mindy Kaling
From here it is clear that the model predicted 3/5 classes correctly. This indicates 60% accuracy of the model. 
The model can further be improved by opting one of the following methods:
-	Changing some of the attribute values of the classifier to better fit the data and produce more accurate predictions.
-	Providing more training data for the model to learn from. This also enables to better fit the data and improve accuracy.
 

5.2 Output: LBPH Classifier

The output can be represented in the following tabular form:

S.NO.	CORRECT LABEL	PREDICTED LABEL

6.		Ben Afflek	Ben Afflek
7.		Jerry Seinfeld	Jerry Seinfeld
8.		Madonna	Jerry Seinfeld
9.		Elton John	Elton John
10.		Mindy Kaling	Mindy Kaling
From here it is clear that the model predicted 4/5 classes correctly. This indicates 80% accuracy of the model. 
The model can further be improved by opting one of the following methods:
-	Changing some of the attribute values of the classifier to better fit the data and produce more accurate predictions.
-	Providing more training data for the model to learn from. This also enables to better fit the data and improve accuracy.


## Inference 

It can be inferred from here that the LBPH Classifier Model showed better performance than the Machine Learning Model for the ‘5 Celebrity Face Dataset’ from Kaggle when all the classifier attributes for both the models were kept as default.
