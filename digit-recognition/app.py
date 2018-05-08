from flask import Flask, render_template
import os
import cv2  
import matplotlib.pyplot as plt
import io
import pyaudio
import wave
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import time


app = Flask(__name__)


 

@app.route('/')

def img_prc():



    return render_template('index.html')


@app.route('/digit')

def image_main_page():

	#setwd = '/Users/avikmoulik/Desktop/digit-recognition/static'
	#os.chdir(setwd)
	import my_methods

	url='http://192.168.43.1:8080/shot.jpg'

	weights_to_use= '/Users/avikmoulik/Documents/Work/Digit Recognition/Dig Recg/mnistneuralnet_new.h5'

	img_width, img_height = 28, 28


	orig,myimg = my_methods.read_img_url(url)

	thresh = cv2.threshold(myimg, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	thresh = cv2.resize(thresh, (500, 300))

	#cv2.imwrite('/Users/avikmoulik/Desktop/digit-recognition/static/im1.jpeg',myimg)
	#cv2.imwrite('/Users/avikmoulik/Desktop/digit-recognition/static/im2.jpeg',orig)
	cv2.imwrite('/Users/avikmoulik/Desktop/digit-recognition/static/im3.jpeg',thresh)


	strng = my_methods.img_segmentation(myimg,img_width,img_height,20,weights_to_use)


	return render_template('img-prc.html', strng=strng)
 
@app.route('/audio')
def audio_main_page():

	filename = "/Users/avikmoulik/Desktop/digit-recognition/static//a1.wav"

	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 16000
	RECORD_SECONDS =6
	WAVE_OUTPUT_FILENAME = filename

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
	                channels=CHANNELS,
	                rate=RATE,
	                input=True,
	                frames_per_buffer=CHUNK)

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	    data = stream.read(CHUNK)
	    frames.append(data)

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

	client = speech.SpeechClient()

	with io.open(filename, 'rb') as audio_file:
	    content = audio_file.read()

	audio = types.RecognitionAudio(content=content)
	config = types.RecognitionConfig(
	    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
	    sample_rate_hertz=RATE,
	    language_code='en-US')

	response = client.recognize(config, audio)

	for result in response.results:
	    a = '{}'.format(result.alternatives[0].transcript)
	return render_template('audio-prc.html',a=a)
 

if __name__ == '__main__':

    app.run(debug=True)

