from gtts import gTTS
import os

# fh=open("story.txt","r")
# myText=fh.read().replace("\n"," ")
myText="ok done see you there"
language='urdu'

output=gTTS(text=myText , lang='en', slow=False)
output.save("voice.mp3")
os.system("start voice.pm3")


