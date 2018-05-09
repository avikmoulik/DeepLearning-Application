# DeepLearning-Application
Recognise Hand written digit and convert Speech to text. We have 2 variant:
* Web based Application (Recommended)
* Window based Application


# Requirements
* IP Webcam Application installed in Android mobile
* Google Chrome Installed in the System
* Python 3.5 and the following packages installed
	+ Pygame
	+ kivy
	+ Flask
	+ cv2  
	+ os
	+ io
	+ pyaudio
	+ wave
	+ google.cloud 
	+ urllib
	+ numpy
	+ imutils
	+ keras
	+ pandas
	+ math
	+ matplotlib
	+ keras

# How to Use the applications
 Go to Terminal/Shell and call the '.PY' file. 
 
 eg. type this:
 python /Users/avikmoulik/Documents/Work/GIT_REPOS/Deeparning-Application/WebPage/app.py

 and then go to the local host(Shown in terminal) in chrome
 Make sure in chrome, inspect mode is on and 'Disable Cache' is ticked under network

 PS: Dont forget to edit the 'app.py' file with proper file path and url
 refer to this 2 lines in code :

 * Dir = "/Users/avikmoulik/Documents/Work/GIT_REPOS/DeepLearning-Application/WebPage/"
 * url='http://192.168.1.118:8080/shot.jpg'

Change directory to where the 'app.py' is placed
Change url to what is shown on IP webcam mobile application. System and mobile should be connected in same network. Otherwise Digit recognition will not work.

For handwritten digit, try to use black marker and small piece of paper (1.5inch * 0.5inch). And While capturing the image place the paper on contrasting surface(Black/Brown surface)


Error Handelling is not done properly, incase there is any error just close the running app and re run