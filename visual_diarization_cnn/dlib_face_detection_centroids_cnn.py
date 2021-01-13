from __future__ import division
from imutils import face_utils
import dlib
import cv2
import numpy as np
import imutils
from pyimagesearch import CentroidTracker
import os
import errno


def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords


video = "NET20070331_thlep_1_2"

training = 0

if training:
	camera = cv2.VideoCapture('../../gridnews/visual_cnn_training_set/' + video + '.mkv')
	file_path = "mouth_features_train/" + video + "/mouth_features_train.txt"
else:
	camera = cv2.VideoCapture('../../gridnews/visual_cnn_testing_set/' + video + '.mkv')
	file_path = "mouth_features_test/" + video + "/mouth_features_test.txt"

print("fps", camera.get(cv2.CAP_PROP_FPS))

if not os.path.exists(os.path.dirname(file_path)):
	try:
		os.makedirs(os.path.dirname(file_path))
	except OSError as exc:  # Guard against race condition
		if exc.errno != errno.EEXIST:
			raise

mouth_features = open(file_path, "w+")

predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

frame_number = 0

ct = CentroidTracker()

mag_reduced_dim = np.zeros((16, 16), dtype="float")

prvs = []
objectid_index_list_prvs = []

while True:
	objectid_num_in_frame = 0
	objectid_index_list_next = []
	rects = []
	ret, frame = camera.read()
	frame_number = frame_number + 1
	print(frame_number)

	if frame_number >= 3:
		prvs = next
		objectid_index_list_prvs = objectid_index_list_next

	next = []
	if ret is False:
		print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
		break
	number_of_mouths_in_frame = 0
	frame = imutils.resize(frame, width=500)
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	dets = detector(frame_gray, 1)

	if len(dets) > 0:
		for k, d in enumerate(dets):
			cor_list = []
			dct_image_gray_vector = []
			shape = predictor(frame_gray, d)
			shape_np = shape_to_np(shape)
			box = face_utils.rect_to_bb(d)
			(x, y, w, h) = face_utils.rect_to_bb(d)
			# cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

			startx = x
			starty = y
			endx = x + w
			endy = y + h
			rect = (startx, starty, endx, endy)

			face_region = frame[y: y + h, x: x + w]
			img_item = "faces/face_region" + str(frame_number) + "_" + str(k) + ".png"
			cv2.imwrite(img_item, face_region)

			image_gray = frame_gray[y: y + h, x: x + w]

			rects.append(rect)
			objects = ct.update(rects)

			# loop over the tracked objects
			for (objectID, centroid) in objects.items():
				# draw both the ID of the object and the centroid of the
				# object on the output frame
				text = "ID {}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
				if ((startx <= centroid[0] <= endx) and (starty <= centroid[1] <= endy)):
					for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
						if (name == "mouth"):
							objectid_num_in_frame = objectid_num_in_frame + 1
							cv2.putText(frame, "Face #{}".format(k + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
							# cv2.putText(frame,name,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
							cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
							img_item = "frames/FRAME" + str(frame_number) + ".png"
							cv2.imwrite(img_item, frame)

							x_max, y_max = shape_np[i:j].max(axis=0)
							x_min, y_min = shape_np[i:j].min(axis=0)

							mouth_center_x = int((x_min + x_max) / 2)
							mouth_center_y = int((y_min + y_max) / 2)
							x_min = int((mouth_center_x + x_min) / 2)
							x_max = int((mouth_center_x + x_max) / 2)

							x_min = int((mouth_center_x + x_min) / 2)
							x_max = int((mouth_center_x + x_max) / 2)

							mouth_region = frame[y: y + h, x:x + w]
							mouth_region = frame[y_min - 5: y_max + 5, x_min - 15: x_max + 15]
							mouth_region = cv2.resize(mouth_region, (32, 32))

							norm_mouth_region = np.zeros((32, 32), dtype="float64")
							norm_mouth_region = cv2.normalize(mouth_region, norm_mouth_region, 0, 255, norm_type=cv2.NORM_MINMAX)

							norm_mouth_region_gray = cv2.cvtColor(norm_mouth_region, cv2.COLOR_BGR2GRAY)
							mouth_region_gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
							cv2.imshow('flow', norm_mouth_region)

							# for NET20070326 VIDEO ONLY
							if video == "NET20070326_thlep_1_1":
								if frame_number >= 3501:
									mouth_features.write(str(frame_number - 3500) + ' ' + str(objectID) + '\n')
									for el in norm_mouth_region.flatten():
										k = k + 1
										mouth_features.write(str(el) + ' ')
									mouth_features.write('\n')
							else:
								mouth_features.write(str(frame_number) + ' ' + str(objectID) + '\n')
								for el in norm_mouth_region.flatten():
									k = k + 1
									mouth_features.write(str(el) + ' ')
								mouth_features.write('\n')

	cv2.imshow("image", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		camera.release()
		break

mouth_features.close()
