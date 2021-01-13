import numpy as np
import cv2
import os
import glob
import re
import errno

videos = ['NET20070412_thlep_1_1', 'NET20070330_thlep_1_2', 'NET20070401_thlep_1_1', 'NET20070330_thlep_1_3', 'NET20070331_thlep_1_1', 'NET20070412_thlep_1_2', 'NET20070402_thlep_1_1', 'NET20070402_thlep_1_2', 'NET20070403_thlep_1_1']

# videos = ['NET20070330_thlep_1_4', 'NET20070326_thlep_1_1', 'NET20070329_thlep_1_1', 'NET20070330_thlep_1_1', 'NET20070331_thlep_1_2']

for video in videos:
	x = loaded_array = np.load('samples&targets/' + video + '/x.npy')
	y = loaded_array = np.load('samples&targets/' + video + '/y.npy')

	x = x.reshape(x.shape[0], 32, 32, 3)

	x_frame_id = loaded_array = np.load('samples&targets/' + video + '/x_frame_id.npy')

	print(x.shape, x_frame_id.shape)

	id_list = []
	line_number_list = []
	x_list_concatenate = []
	x_frame_id_list_concatenate = []
	y_list_concatenate = []

	for i in range(x.shape[0]):
		if x_frame_id[i][1] not in id_list:
			id_list.append(x_frame_id[i][1])
			line_number_list.append(i)
		else:
			pos_of_id = id_list.index(x_frame_id[i][1])
			line_number_prev = line_number_list[pos_of_id]
			line_number_list[pos_of_id] = i
			img_item = "mouths/" + video + "/spk" + str(x_frame_id[line_number_prev, 1]) + "/" + str(y[i]) + "/" + "frame_" + "{0:0=4d}".format(x_frame_id[line_number_prev, 0]) + ".png"
			if not os.path.exists(os.path.dirname(img_item)):
				try:
					os.makedirs(os.path.dirname(img_item))
				except OSError as exc:  # Guard against race condition
					if exc.errno != errno.EEXIST:
						raise
			cv2.imwrite(img_item, x[line_number_prev, :])

number_of_concatennated_images = 30
first_time = True
img_list = []
im_v_list = []
frame_number_list = []
for video in videos:
	x_list_concatenate = []
	x_frame_id_list_concatenate = []
	y_list_concatenate = []
	rootdir = "mouths/" + video
	for spk in os.listdir(rootdir):
		for target in os.listdir(rootdir + "/" + spk):
			path = rootdir + "/" + spk + "/" + target
			print(path)
			img_list = []
			filelist = glob.glob(os.path.join(path, "*.png"))
			for infile in sorted(filelist):
				words = infile.rstrip().split('/')
				frame_number = words[4]

				frame_number_int = int(re.search(r'\d+', frame_number).group(0))
				if not frame_number_list:
					frame_number_list.append(frame_number_int)
					img = cv2.imread(str(infile))
					img_list.append(img)
				else:
					frame_number_prev = frame_number_list[-1]
					if frame_number_int - frame_number_prev == 1:
						frame_number_list.append(frame_number_int)
						img = cv2.imread(str(infile))
						img_list.append(img)
					else:
						frame_number_list = []
						img_list = []
						frame_number_list.append(frame_number_int)
						img = cv2.imread(str(infile))
						img_list.append(img)

				if len(img_list) == number_of_concatennated_images:
					for k in range(len(img_list)):
						if first_time:
							im1 = img_list[k]
							first_time = False
						else:
							im2 = img_list[k]
							im_v = cv2.hconcat([im1, im2])
							if number_of_concatennated_images == 30:
								if k == 5 or k == 11 or k == 17 or k == 23 or k == 29:
									im_v_list.append(im_v)
									im_v = 0
									first_time = True
								im1 = im_v
							elif number_of_concatennated_images == 49:
								if k == 6 or k == 13 or k == 20 or k == 27 or k == 34 or k == 41 or k == 48:
									im_v_list.append(im_v)
									im_v = 0
									first_time = True
								im1 = im_v
							elif number_of_concatennated_images == 20:
								if k == 4 or k == 9 or k == 14 or k == 19:
									im_v_list.append(im_v)
									im_v = 0
									first_time = True
								im1 = im_v

					img_list = []
					frame_number_list = []
					for i in range(len(im_v_list)):
						if i == 0:
							im_v1 = im_v_list[i]
						else:
							im_v2 = im_v_list[i]
							im_v_concatenate = cv2.vconcat([im_v1, im_v2])
							im_v1 = im_v_concatenate
					im_v_list = []
					img_item = "mouths_cocatenate" + "_" + str(number_of_concatennated_images) + "/" + video + "/" + spk + "/" + target + "/" + frame_number + ".png"
					if not os.path.exists(os.path.dirname(img_item)):
						try:
							os.makedirs(os.path.dirname(img_item))
						except OSError as exc:  # Guard against race condition
							if exc.errno != errno.EEXIST:
								raise
					cv2.imwrite(img_item, im_v_concatenate)

					image = cv2.imread(img_item)
					speaker_id = int(re.search(r'\d+', spk).group(0))
					frame = int(re.search(r'\d+', frame_number).group(0))

					x_list_concatenate.append(image)
					x_frame_id_list_concatenate.append([frame, speaker_id])
					y_list_concatenate.append(int(target))

	x_concatenate = np.asarray(x_list_concatenate)
	x_frame_id_concatenate = np.asarray(x_frame_id_list_concatenate)
	y_concatenate = np.asarray(y_list_concatenate)
	print(x_concatenate.shape, x_frame_id_concatenate.shape, y_concatenate.shape)

	file_path = 'samples&targets_concatenate' + '_' + str(number_of_concatennated_images) + '/' + video + '/x_concatenate.npy'
	if not os.path.exists(os.path.dirname(file_path)):
		try:
			os.makedirs(os.path.dirname(file_path))
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	np.save('samples&targets_concatenate' + '_' + str(number_of_concatennated_images) + '/' + video + '/x_concatenate.npy', x_concatenate)

	file_path = 'samples&targets_concatenate' + '_' + str(number_of_concatennated_images) + '/' + video + '/x_frame_id_concatenate.npy'
	if not os.path.exists(os.path.dirname(file_path)):
		try:
			os.makedirs(os.path.dirname(file_path))
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	np.save('samples&targets_concatenate' + '_' + str(number_of_concatennated_images) + '/' + video + '/x_frame_id_concatenate.npy', x_frame_id_concatenate)

	file_path = 'samples&targets_concatenate' + '_' + str(number_of_concatennated_images) + '/' + video + '/y_concatenate.npy'
	if not os.path.exists(os.path.dirname(file_path)):
		try:
			os.makedirs(os.path.dirname(file_path))
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	np.save('samples&targets_concatenate' + '_' + str(number_of_concatennated_images) + '/' + video + '/y_concatenate.npy', y_concatenate)
