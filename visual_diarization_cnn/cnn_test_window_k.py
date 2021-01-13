import numpy as np
import os
from tensorflow.keras.models import load_model
import errno

new_model = load_model('model/30_cnn_model.h5')

# test videos: ['NET20070331_thlep_1_2', 'NET20070326_thlep_1_1', 'NET20070329_thlep_1_1', 'NET20070330_thlep_1_4', 'NET20070330_thlep_1_1']
dirname = ['NET20070330_thlep_1_1']

number_of_mouths_in_image = 30

file_path = "visual_hypothesis_file_cnn" + "_" + str(number_of_mouths_in_image) + "/" + dirname[0] + "/visual_hypothesis_file_cnn_temp" + dirname[0] + ".txt"
if not os.path.exists(os.path.dirname(file_path)):
	try:
		os.makedirs(os.path.dirname(file_path))
	except OSError as exc:  # Guard against race condition
		if exc.errno != errno.EEXIST:
			raise

hypothesis_file_temp = open(file_path, "w")

file_path = "visual_hypothesis_file_cnn" + "_" + str(number_of_mouths_in_image) + "/" + dirname[0] + "/visual_hypothesis_file_cnn_" + dirname[0] + ".txt"
if not os.path.exists(os.path.dirname(file_path)):
	try:
		os.makedirs(os.path.dirname(file_path))
	except OSError as exc:  # Guard against race condition
		if exc.errno != errno.EEXIST:
			raise

hypothesis_file = open(file_path, "w")

file_path = "visual_reference_file_cnn/" + dirname[0] + "/visual_reference_file_cnn_" + dirname[0] + ".txt"
if not os.path.exists(os.path.dirname(file_path)):
	try:
		os.makedirs(os.path.dirname(file_path))
	except OSError as exc:  # Guard against race condition
		if exc.errno != errno.EEXIST:
			raise

reference_file = open(file_path, "w")

x_test_0_frame_id = loaded_array = np.load('samples&targets_concatenate' + '_' + str(number_of_mouths_in_image) + '/' + dirname[0] + '/x_frame_id_concatenate.npy')

y_test_0 = loaded_array = np.load('samples&targets_concatenate' + '_' + str(number_of_mouths_in_image) + '/' + dirname[0] + '/y_concatenate.npy')

x_test_0 = loaded_array = np.load('samples&targets_concatenate' + '_' + str(number_of_mouths_in_image) + '/' + dirname[0] + '/x_concatenate.npy')

x_test_0 = x_test_0 / 255

x_test = x_test_0
y_test = y_test_0

x_test_frame_id = x_test_0_frame_id

if number_of_mouths_in_image == 20:
	x_test = x_test.reshape(x_test.shape[0], 128, 160, 3)
elif number_of_mouths_in_image == 30:
	x_test = x_test.reshape(x_test.shape[0], 160, 192, 3)
elif number_of_mouths_in_image == 49:
	x_test = x_test.reshape(x_test.shape[0], 224, 224, 3)

y_test = y_test.reshape(x_test.shape[0], 1)

predictions = new_model.predict([x_test])

hypothesis_file_temp_line_number = 0
error = 0
for pred, label, frame_id in zip(predictions, y_test, x_test_frame_id):

	start_frame = frame_id[0] - number_of_mouths_in_image + 1
	end_frame = frame_id[0]
	hypothesis_file_temp.write(str(start_frame) + ' ' + str(end_frame) + ' ' + str(frame_id[1]) + ' ' + str(pred[0]) + '\n')
	hypothesis_file_temp_line_number = hypothesis_file_temp_line_number + 1
	if label[0] == 1:
		if dirname[0] == 'NET20070326_thlep_1_1':
			if start_frame >= 120:
				reference_file.write(str(start_frame) + ' ' + str(end_frame) + ' ' + str(frame_id[1]) + '\n')
		else:
			reference_file.write(str(start_frame) + ' ' + str(end_frame) + ' ' + str(frame_id[1]) + '\n')
	if pred >= 0.5 and label == 0 or pred < 0.5 and label == 1:
		error = error + 1

print("error=", error / y_test.shape[0])
hypothesis_file_temp.close()
reference_file.close()

hypothesis_file_temp = open("visual_hypothesis_file_cnn" + "_" + str(number_of_mouths_in_image) + "/" + dirname[0] + "/visual_hypothesis_file_cnn_temp" + dirname[0] + ".txt", 'r')

line_number_hypothesis = 0
for line in hypothesis_file_temp:
	line_number_hypothesis = line_number_hypothesis + 1

hypothesis_file_temp.close()

reference_file = open("visual_reference_file_cnn/" + dirname[0] + "/visual_reference_file_cnn_" + dirname[0] + ".txt", 'r')

line_number_reference = 0
for line in reference_file:
	line_number_reference = line_number_reference + 1

reference_file.close()

# sort hypothesis file

bands = []

file_path = "visual_hypothesis_file_cnn" + "_" + str(number_of_mouths_in_image) + "/" + dirname[0] + "/visual_hypothesis_file_cnn_temp" + dirname[0] + ".txt"

with open(file_path) as fp:
	lines = fp.readlines()
	bands.append(sorted(lines, key=lambda x: float(x.split(' ')[0]), reverse=False))

fp.close()

str_list = []

for i in range(line_number_hypothesis):
	str_list.append(str(bands[0][i]))

file_path = "visual_hypothesis_file_cnn" + "_" + str(number_of_mouths_in_image) + "/" + dirname[0] + "/visual_hypothesis_file_cnn_temp" + dirname[0] + ".txt"

with open(file_path, 'w') as fp:
	for el in str_list:
		fp.write(el)

fp.close()

# sort reference file

bands = []

file_path = "visual_reference_file_cnn/" + dirname[0] + "/visual_reference_file_cnn_" + dirname[0] + ".txt"

with open(file_path) as fp:
	lines = fp.readlines()
	bands.append(sorted(lines, key=lambda x: float(x.split(' ')[0]), reverse=False))

fp.close()

str_list = []

for i in range(line_number_reference):
	str_list.append(str(bands[0][i]))

file_path = "visual_reference_file_cnn/" + dirname[0] + "/visual_reference_file_cnn_" + dirname[0] + ".txt"

with open(file_path, 'w') as fp:
	for el in str_list:
		fp.write(el)

fp.close()

hypothesis_file_temp = open("visual_hypothesis_file_cnn" + "_" + str(number_of_mouths_in_image) + "/" + dirname[0] + "/visual_hypothesis_file_cnn_temp" + dirname[0] + ".txt", 'r')

frames_list = []
pred_list = []
speaker_id_list = []
line_number = 0
for line in hypothesis_file_temp:
	words = line.rstrip().split(' ')
	start_frame = int(words[0])
	end_frame = int(words[1])
	speaker_id = int(words[2])
	pred = float(words[3])
	line_number = line_number + 1
	if not frames_list:
		frames_list.append([start_frame, end_frame])
		pred_list.append(pred)
		speaker_id_list.append(speaker_id)
	else:
		if start_frame == frames_list[0][0] or abs(int(start_frame) - int(frames_list[0][0])) <= (number_of_mouths_in_image - 1) and line_number < hypothesis_file_temp_line_number:
			frames_list.append([start_frame, end_frame])
			pred_list.append(pred)
			speaker_id_list.append(speaker_id)
		else:
			if line_number == hypothesis_file_temp_line_number:
				frames_list.append([start_frame, end_frame])
				pred_list.append(pred)
				speaker_id_list.append(speaker_id)
			start_frame_temp = start_frame
			end_frame_temp = end_frame
			speaker_id_temp = speaker_id
			max_pred = max(pred_list)
			max_pred = float(max_pred)
			index = pred_list.index(max_pred)
			start_frame = frames_list[index][0]
			end_frame = frames_list[index][1]
			speaker_id = speaker_id_list[index]

			if max_pred >= 0.01:
				if dirname[0] == 'NET20070326_thlep_1_1':
					if start_frame >= 120:
						hypothesis_file.write(str(start_frame) + ' ' + str(end_frame) + ' ' + str(speaker_id) + '\n')
				else:
					hypothesis_file.write(str(start_frame) + ' ' + str(end_frame) + ' ' + str(speaker_id) + '\n')
			frames_list = []
			pred_list = []
			speaker_id_list = []
			frames_list.append([start_frame_temp, end_frame_temp])
			pred_list.append(pred)
			speaker_id_list.append(speaker_id_temp)

hypothesis_file_temp.close()
hypothesis_file.close()
