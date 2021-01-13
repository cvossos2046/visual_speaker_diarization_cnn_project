video = "NET20070331_thlep_1_2"

training = 0

if training:
	file_path = "mouth_features_train/" + video
	mouth_features = open(file_path + "/mouth_features_train.txt", "r")
	mouth_features_clear = open(file_path + "/mouth_features_clear.txt", "w")
else:
	file_path = "mouth_features_test/" + video
	mouth_features = open(file_path + "/mouth_features_test.txt", "r")
	mouth_features_clear = open(file_path + "/mouth_features_clear.txt", "w")

visual_ids = []

for line in mouth_features:
	words = line.rstrip().split(' ')
	if len(words) == 2:
		if not int(words[1]) in visual_ids:
			visual_ids.append(int(words[1]))

print("visual_ids", visual_ids)
mouth_features.close()

if training:
	mouth_features = open(file_path + "/mouth_features_train.txt", "r")
else:
	mouth_features = open(file_path + "/mouth_features_test.txt", "r")


for line in mouth_features:
	words = line.rstrip().split(' ')
	if len(words) == 2:
		frame = int(words[0])
		speaker_id = int(words[1])
	else:
		if speaker_id == 0 or speaker_id == 1 or speaker_id == 2 or speaker_id == 3 or speaker_id == 4:
			mouth_features_clear.write(str(frame) + ' ' + str(speaker_id) + '\n')
			for features in words:
				mouth_features_clear.write(str(features) + ' ')

			mouth_features_clear.write('\n')

mouth_features.close()
mouth_features_clear.close()
