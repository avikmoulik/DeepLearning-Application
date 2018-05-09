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
        text : 'Image Prpcessing'
        on_release: root.clk1()
        pos: (.5*root.width,.5*root.height)
        size_hint: (.3,.5)
        background_color: (1,1,0,1)


    Label:
        id: result
        font_size: 40
        text: ''




''')
Dir = "/Users/avikmoulik/Documents/Work/GIT_REPOS/DeepLearning-Application/Application/"
url='http://192.168.1.118:8080/shot.jpg'
weights_to_use = Dir+'mnistneuralnet_new.h5'

class MyLayout(BoxLayout):
    
    def clk1(self):
        

        img_width, img_height = 28, 28


        orig,myimg = my_methods.read_img_url(url)

        thresh = cv2.threshold(myimg, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (500, 300))

        cv2.imwrite(Dir+'im.jpeg',thresh)


        strng = my_methods.img_segmentation(myimg,img_width,img_height,20,weights_to_use)
        self.ids.result.text = strng


class DeepLearningApp(App):
    def build(self):
        return MyLayout()


if __name__ == "__main__":
    DeepLearningApp().run()
