#!/usr/bin/env python3
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2015 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

from __future__ import print_function
from diarization import DiarizationErrorRate
from pyannote.core import Annotation
from pyannote.core import Segment as PyannSegment

video = 'NET20070330_thlep_1_1'
number_of_mouths_in_image = 30

hypothesis_visual_cnn = Annotation()
reference_visual_cnn = Annotation()

hypothesis_file = open("../../visual_diarization_cnn/visual_hypothesis_file_cnn" + "_" + str(number_of_mouths_in_image) + '/' + video + "/visual_hypothesis_file_cnn_" + video + ".txt", "r")
reference_file = open("../../visual_diarization_cnn/visual_reference_file_cnn/" + video + "/visual_reference_file_cnn_" + video + ".txt", "r")

for line in hypothesis_file:
	words = line.rstrip().split(' ')
	start = int(words[0])
	end = int(words[1])
	speaker_id = int(words[2])
	hypothesis_visual_cnn[PyannSegment(start=start * 0.04, end=end * 0.04)] = speaker_id

hypothesis_file.close()

for line in reference_file:
	words = line.rstrip().split(' ')
	start = int(words[0])
	end = int(words[1])
	speaker_id = int(words[2])
	reference_visual_cnn[PyannSegment(start=start * 0.04, end=end * 0.04)] = speaker_id

reference_file.close()
metric = DiarizationErrorRate()

value = metric(reference_visual_cnn, hypothesis_visual_cnn)
print("cnn der =", value)
