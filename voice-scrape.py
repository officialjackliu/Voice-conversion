import os
import csv
import requests
import whisper
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import subprocess

def scrape_voice_lines(character_url, output_dir, character_name):
    response = requests.get(character_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    os.makedirs(f"{output_dir}/wavs", exist_ok=True)
    metadata = []

    voice_entries = soup.select("table.wikitable tr")
    print(f"Found {len(voice_entries)} voice lines. Starting download...")

    model = whisper.load_model("base")  

    for i, row in enumerate(voice_entries, start=1):
        audio_elem = row.select_one("audio[src]")
        text_elem = row.find_previous("th")

        if not audio_elem or not text_elem:
            continue  

        text = text_elem.text.strip()
        audio_url = urljoin(character_url, audio_elem["src"])  
        audio_filename = f"{character_name}{i:04d}.ogg"
        audio_path = os.path.join(output_dir, "wavs", audio_filename)

        with requests.get(audio_url, stream=True) as r:
            with open(audio_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)

        wav_filename = audio_filename.replace(".ogg", ".wav")
        wav_path = os.path.join(output_dir, "wavs", wav_filename)
        subprocess.run(["ffmpeg", "-i", audio_path, "-ar", "22050", wav_path, "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(audio_path) 

        transcription = model.transcribe(wav_path)["text"].strip()

        metadata.append([wav_filename, transcription, transcription])

        print(f"Processed {i}/{len(voice_entries)}: {wav_filename}") 

    metadata_file = os.path.join(output_dir, "metadata.csv")
    with open(metadata_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows(metadata)

    print(f"Dataset saved at {output_dir}")

character_name = "MARCH_7TH"
character_url = "https://honkai-star-rail.fandom.com/wiki/March_7th/Voice-Overs"
output_dir = f"XTTS/notebooks/tts_train_dir/{character_name}"

scrape_voice_lines(character_url, output_dir, character_name)













# import os
# from TTS.api import TTS
# import subprocess

# # Function to download and convert YouTube video to WAV
# def download_audio(youtube_url, output_file="speaker.wav"):
#     print("Downloading audio...")
#     command = [
#         "yt-dlp",
#         "-x",  # Extract audio
#         "--audio-format", "wav",  # Convert to WAV
#         "-o", output_file,  # Save as specified file
#         youtube_url
#     ]
    
#     subprocess.run(command, check=True)
#     print(f"Audio downloaded and saved as {output_file}.")

# def main():

#     youtube_url = input("Enter YouTube video URL: ").strip()
#     output_file = "speaker.wav"
#     download_audio(youtube_url, output_file)

#     tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

#     text = input("Enter the text you want to convert to speech: ").strip()

#     tts.tts_to_file(
#         text=text,
#         speaker_wav=output_file,
#         language="en",
#         file_path="output.wav"
#     )
#     print("TTS conversion complete. Output saved as 'output.wav'.")
