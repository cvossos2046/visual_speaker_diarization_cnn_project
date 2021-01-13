import numpy as np
import os
import errno

video = "NET20070331_thlep_1_2"

training = 0

if training:
	file_path = "mouth_features_train/" + video
else:
	file_path = "mouth_features_test/" + video

mouth_features = open(file_path + "/mouth_features_clear.txt", "r")
y_frame_id_file = open("targets/" + video + "/y_frame_id_file.txt", "r")

file_path = "targets/" + video + "/y_file.txt"
if not os.path.exists(os.path.dirname(file_path)):
	try:
		os.makedirs(os.path.dirname(file_path))
	except OSError as exc:  # Guard against race condition
		if exc.errno != errno.EEXIST:
			raise

y_file = open(file_path, "w+")

x_frame_id_list = []
mouth_features_list = []
mouth_frame_id_list = []
y_frame_id_list = []
x_list = []
y_list = []
visual_ids = []
mouth_features_np = np.zeros((1024 * 3), dtype="float")

line_number = 0
for line in mouth_features:
	words = line.rstrip().split(' ')
	if len(words) == 2:
		mouth_frame_id_list.append([int(words[0]), int(words[1])])

		line_number = line_number + 1
		if not int(words[1]) in visual_ids:
			visual_ids.append(int(words[1]))
	else:
		for i, el in enumerate(words):
			mouth_features_np[i] = el
		mouth_features_list.append(tuple(mouth_features_np))
	print(line_number)
	if line_number == 22000:  # memory problem for more than 22000 lines
		break
print("line number", line_number)

for line in y_frame_id_file:
	words = line.rstrip().split(' ')
	frame = int(words[0])
	speakers_id = int(words[1])
	target = int(words[2])
	y_frame_id_list.append([frame, speakers_id, target])

mouth_features.close()
y_frame_id_file.close()
last_pos = 0
k = 0
for el_x1, el_x2 in zip(mouth_frame_id_list, mouth_features_list):
	i = last_pos
	while True:
		if el_x1[0] != y_frame_id_list[i][0]:
			i = i + 1
			if i == len(y_frame_id_list):
				break
		else:
			j = i
			while el_x1[0] == y_frame_id_list[j][0] and el_x1[1] != y_frame_id_list[j][1]:
				j = j + 1
				if j == len(y_frame_id_list):
					break
			if j < len(y_frame_id_list) and el_x1[1] == y_frame_id_list[j][1]:
				k = k + 1
				x_frame_id_list.append([el_x1[0], el_x1[1]])
				x_list.append(el_x2)
				y_list.append(y_frame_id_list[j][2])
			last_pos = i
			break

for el in y_list:
	y_file.write(str(el) + '\n')
x_frame_id = np.asarray(x_frame_id_list)
x = np.asarray(x_list)
y = np.asarray(y_list)
print("end", y.shape, x.shape)

y_file.close()

file_path = 'samples&targets/' + video + '/x.npy'
if not os.path.exists(os.path.dirname(file_path)):
	try:
		os.makedirs(os.path.dirname(file_path))
	except OSError as exc:  # Guard against race condition
		if exc.errno != errno.EEXIST:
			raise
np.save('samples&targets/' + video + '/x.npy', x)

file_path = 'samples&targets/' + video + '/x_frame_id.npy'
if not os.path.exists(os.path.dirname(file_path)):
	try:
		os.makedirs(os.path.dirname(file_path))
	except OSError as exc:  # Guard against race condition
		if exc.errno != errno.EEXIST:
			raise
np.save('samples&targets/' + video + '/x_frame_id.npy', x_frame_id)

file_path = 'samples&targets/' + video + '/y.npy'
if not os.path.exists(os.path.dirname(file_path)):
	try:
		os.makedirs(os.path.dirname(file_path))
	except OSError as exc:  # Guard against race condition
		if exc.errno != errno.EEXIST:
			raise
np.save('samples&targets/' + video + '/y.npy', y)
