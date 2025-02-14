import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
from transformers import pipeline
import pyttsx3
import threading

model = whisper.load_model("base")

classifier = pipeline("text-classification", model="Panda0116/emotion-classification-model")

engine = pyttsx3.init()

label_map = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}

recording = False

def record_audio(fs):
    global recording
    print("Recording... Press Enter to stop.")
    audio = []
    stream = sd.InputStream(samplerate=fs, channels=1, dtype='float32')
    with stream:
        while recording:
            frame, _ = stream.read(1024)
            audio.append(frame)
    audio = np.concatenate(audio, axis=0)
    print("Recording complete.")
    return audio

def wait_for_enter():
    global recording
    input()
    recording = False

def main():
    global recording
    fs = 16000  

    recording = True
    input_thread = threading.Thread(target=wait_for_enter)
    input_thread.start()

    audio = record_audio(fs)

    sf.write("input.wav", audio, fs)

    result = model.transcribe("input.wav")
    text = result["text"]
    print(f"Transcribed Text: {text}")

    if text:
        result = classifier(text)[0]
        label = int(result['label'].split('_')[-1])
        emotion = label_map[label]
        score = result['score']
        print(f"Emotion: {emotion}, Score: {score:.2f}")

        engine.say(f"The detected emotion is {emotion} with a confidence score of {score:.2f}")
        engine.runAndWait()

if __name__ == "__main__":
    main()




