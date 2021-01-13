# Visual diarization with support vector machines

## What is this?
This is a python project for visual speaker diarization. 
The goal is to find out who speaks and when in videos with multi-speaker conversations in Broadcast News of Greek television, 
using mouth images and Convolutional Neural Networks.

## Requirements
You need python 3.6.9 version and conda 4.6.11 environment.

## Dataset and ground truth
The gridnews directory contains the videos that were used for training and testing as well the corresponding transcriptions files.
[dataset](https://drive.google.com/drive/u/0/folders/1TO72-uN6_vexSOJdIr3HG-Hws4gyPCwT)

## Steps for running the project
### Create the ground truth
In pyannote-parser-develop/tests directory run the trs_file_parse_visual_cnn.py (creates the ground truth).

Initialize the video variable with the video name. In reference parser write the corresponding transcription file.

	
	video = 'NET20070331_thlep_1_2'

	reference = parser.read("../gridnews/NET20070331/NET20070331.trs")


Also, in trs.py should initialize the video variable with the video name and set the time limits (start, end).


	video = 'NET20070331_thlep_1_2'

	start = 1505.931
    end = 1952.862


Time limits for the 5 testing videos:
* Net20070326_thelep_1_1: start = 1663.556, end = 2244.469
* Net20070329_thelep_1_1: start = 1705.173, end = 2455.011
* Net20070330_thelep_1_1: start = 1180.084, end = 1540.335
* Net20070330_thelep_1_4: start = 2246.728, end = 2615.844
* Net20070331_thelep_1_2: start = 1505.931, end = 1952.862


### Training
In visual_diarization_cnn directory run the following scripts:

1. dlib_face_detection_centroids_cnn.py (creates the mouth features).
2. clear_mouth_features.py (ignores the incidental speakers).
	* Should change manually the ids of each main speaker to a unique id at mouth_features_train.txt.
3. create_samples_targets.py (creates samples and targets for each mouth and save them in numpy arrays).


For 1, 2, 3 scripts set the training variable to 1.


	training = 1


4. k_frames_making_dataset.py (concatenates the mouths of each speaker separately for k continuous frames in larger concatenated images).
	* I examined 3 different values of k (k = 20, k = 30, k = 49).


Set the number_of_concatennated_images variable to 20 or 30 or 49.


	number_of_concatennated_images = 30

5. For each size of the concatenated images run the following scripts in order to create and train the cnn model (I used Google Colab):
	1. cnn_model_train_concatenate_20.py (classifies the 20 concatenated mouths images to speech/non speech)
	2. cnn_model_train_concatenate_30.py (classifies the 30 concatenated mouths images to speech/non speech)
	3. cnn_model_train_concatenate_49.py (classifies the 49 concatenated mouths images to speech/non speech)

### Testing
In visual_diarization_cnn directory run the following scripts:

1. dlib_face_detection_centroids_cnn.py (creates the mouth features).
2. clear_mouth_features.py (ignores the incidental speakers).
	* Should change manually the ids of each main speaker to a unique id at mouth_features_test.txt.
3. create_samples_targets.py (creates samples and targets for each mouth and save them in numpy arrays).


For 1, 2, 3 scripts set the training variable to 0.


	training = 0
4. cnn_test_window_k.py (creates the speaker diarization hypothesis files).

Set the  number_of_mouths_in_image variable to 20 or 30 or 49.
	
	
	number_of_mouths_in_image = 30


### Evaluation
In pyannote-parser-develop/tests directory run the der_measure_cnn.py (computes the diarization error rate).

Set the video variable with the video name and the number_of_mouths_in_image to 20 or 30 or 49.

	
	video = 'NET20070331_thlep_1_2'

	number_of_mouths_in_image = 30

## Author
Charalampos Vossos
* [linkedin](https://www.linkedin.com/in/charalampos-vossos-6bbb78185/)
* email: <vossosx96@gmail.com>

## License
Copyright © 2021, Charalampos Vossos. Licensed under [MIT License](LICENSE)
