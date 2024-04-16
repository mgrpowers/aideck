import speech_recognition as sr
import json
import cv2
import time
from imageai.Detection import ObjectDetection
import os
from torchvision import models, transforms
from PIL import Image
import torch
from gtts import gTTS

language = 'en'

recognizer = sr.Recognizer()

with open('imagenet-simple-labels.json') as f:
    class_index = json.load(f)

model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def listen_and_process():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said: " + text)

        if "take a picture and tell me what you see" in text.lower():
            print("you get a picture")
            import cv2

            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                raise IOError("Cannot open webcam")

            ret, frame = cap.read()

            if not ret:
                raise IOError("Cannot read frame")

            cv2.imwrite('pin.jpg', frame)

            cap.release()


            image = Image.open('pin.jpg')
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image)

            _, top5_preds = torch.topk(output, 5)
            print(top5_preds)


            class_names = [class_index[i] for i in top5_preds[0]]

            print(class_names)

            highprob = class_names[0]
            print(highprob)
            myobj = gTTS(text=highprob, lang=language, slow=False)

            myobj.save("sayit.mp3")

            os.system("mpg321 sayit.mp3")

        else:
            print("not it")


        listen_and_process()
        time.sleep(3)

    except sr.UnknownValueError:
        print("Speak Up")
        listen_and_process()
    except sr.RequestError as e:
        print("Could not understand {0}".format(e))

listen_and_process()





