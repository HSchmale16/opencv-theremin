# Theremin with OpenCV and Mediapipe

The theremin is a super cool kind of ghostly musical instrument. 
I wanted one, but didn't want to buy one cause it would sit around. So I decided to make one with
OpenCV. Right now it just does the pitch stuff, without much feedback like a proper theremin would
do, but that can be fixed in phase2.

This uses mediapipe's hand detector to find all your hands. It will only make a sound if it finds
exactly 2 hands.



# Setup

I recommend that you use a virtualenv to install this things dependencies.

So follow these steps:


```
cd opencv-theremin
virtualenv .
source bin/activate
pip3 install -r requirements.txt
python3 main.py
```


# Sources
* https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/
    * FYI doesn't use proper kwargs for hand detector in part 2
* https://magpi.raspberrypi.com/articles/raspberry-pi-theremin