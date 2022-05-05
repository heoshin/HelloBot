import string
from gtts import gTTS
import os

def tts(text = ""):
    filePath = "./tts/hello.mp3"
    if not os.path.isfile(filePath):
        tts = gTTS(text=text, lang='ko')
        tts.save(filePath)
        print("tts generate:", filePath)


filePath = "./tts/hello.mp3"
text = ""
while len(text) == 0:
    text = input("Please input tts Text: ")

if os.path.isfile(filePath):
    isOverwrite = "n"
    isOverwrite = input("./tts/hello.mp3 is already generated. overwrite?(y/N): ")
    if isOverwrite == "y":
        print("remove:", filePath)
        os.remove(filePath)
    elif isOverwrite == "n":
        print("exit")
        exit()

tts(text)