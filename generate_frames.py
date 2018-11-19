import cv2
import os
import shutil

def generate_frames(video_path):
	shutil.rmtree("./images", ignore_errors=True)
	os.makedirs("./images")
	print('Generating frames from the video into ./images directory...')
	vidcap = cv2.VideoCapture(video_path)
	success,image = vidcap.read()
	count = 1
	while success:
		cv2.imwrite("./images/"+("000000000000000000000"+str(count))[-6:]+".jpg", image)     # save frame as JPG file      
		success,image = vidcap.read()
		print('Read a new frame: ', success)
		count += 1
