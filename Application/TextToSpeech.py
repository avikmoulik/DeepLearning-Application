import kivy
from kivy.app import App
import my_methods
import cv2  
import os
import io
import pyaudio
import wave
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types



#from kivy.uix.scatter import Scatter
#from kivy.uix.label import Label
#from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.lang import Builder


Builder.load_string('''

<MyLayout>:
    canvas.before:
        Rectangle:
            pos: self.pos
            size: self.size
            source: 'background-1.jpeg'
    orientation: 'vertical'
    Label:
        text: 'Deep Learning App'
        font_size: 70


    Button : 
        text : 'Speech To Text'
        on_release: root.clk2()
        pos: (.5*root.width,.5*root.height)
        size_hint: (.3,.5)
        background_color: (1,1,0,1)


    Label:
        id: result
        font_size: 40
        text: ''




''')

Dir = "/Users/avikmoulik/Documents/Work/GIT_REPOS/DeepLearning-Application/Application/"

class MyLayout(BoxLayout):
	

	def clk2(self):
		

		filename = Dir+"a1.wav"

		CHUNK = 1024
		FORMAT = pyaudio.paInt16
		CHANNELS = 1
		RATE = 16000
		RECORD_SECONDS =6
		WAVE_OUTPUT_FILENAME = filename

		print ('Start Speaking')

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

		print ('Recording Done, We are processing the audio')

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
		self.ids.result.text = a

class DeepLearningApp(App):
	def build(self):
		return MyLayout()


if __name__ == "__main__":
	DeepLearningApp().run()
