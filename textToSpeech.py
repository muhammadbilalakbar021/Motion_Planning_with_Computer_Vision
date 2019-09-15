from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play

# fh=open("story.txt","r")
# myText=fh.read().replace("\n"," ")
myText="My name is Bilal Akbar"
print("Loading Text !!!!!!!!!")
language='en'
print("Enabling Language done !!!!!!!!!")

output=gTTS(text=myText , lang=language, slow=False)
output.save("voice.mp3")
print("Converted text to Speech")
print("Loading !!!!!!!!!")
song =AudioSegment.from_mp3("/media/bilal/Work_Space/Robotics_Path_Planning/openCV/voice.mp3")
play(song)
#os.system("start voice.pm3")


