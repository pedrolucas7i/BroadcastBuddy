import os
import time
import tempfile
import subprocess
import json
import smtplib
from email.mime.text import MIMEText
from pydub import AudioSegment, silence
import speech_recognition as sr
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file


# -------- CONFIG --------

IPTV_URL = "https://d277k9d1h9dro4.cloudfront.net/out/v1/293e7c3464824cbd8818ab8e49dc5fe9/index.m3u8"  # Your IPTV stream URL here

EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

# Ollama model name (make sure you installed it)
OLLAMA_MODEL = "llama3.2:1b"

SEGMENT_LENGTH_SEC = 60  # how many seconds per audio chunk to process

# Silence detection params (adjust if needed)
MIN_SILENCE_LEN_MS = 1500
SILENCE_THRESH_DB = -40
AD_SILENCE_COUNT_THRESHOLD = 3  # number of silence chunks to assume ads started

# ------------------------


def download_audio_segment(url, start_sec, duration_sec, output_path):
    # Grab audio segment from IPTV stream with ffmpeg
    cmd = (
        f"ffmpeg -y -ss {start_sec} -i \"{url}\" -t {duration_sec} "
        f"-vn -acodec pcm_s16le -ar 16000 -ac 1 \"{output_path}\""
    )
    os.system(cmd)


def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Speech recognition API error: {e}")
        return ""


def ollama_summarize(text):
    prompt = f"Summarize this content briefly:\n{text}"
    try:
        result = subprocess.run(
            ['ollama', 'run', OLLAMA_MODEL, '--json'],
            input=json.dumps({"prompt": prompt}),
            text=True,
            capture_output=True,
            check=True
        )
        response = json.loads(result.stdout)
        return response.get('completion', '').strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return "Error generating summary."


def send_email(subject, body):
    msg = MIMEText(body)
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


def detect_ads(audio_segment):
    silent_chunks = silence.detect_silence(
        audio_segment,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=SILENCE_THRESH_DB
    )
    return len(silent_chunks) >= AD_SILENCE_COUNT_THRESHOLD


def main():
    print("Starting IPTV monitoring...")
    summary_text = ""
    start_time = 0
    ads_started = False

    while not ads_started:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_path = temp_audio_file.name

        print(f"Downloading audio segment: {start_time}s to {start_time + SEGMENT_LENGTH_SEC}s")
        download_audio_segment(IPTV_URL, start_time, SEGMENT_LENGTH_SEC, temp_path)

        audio = AudioSegment.from_wav(temp_path)
        ads_started = detect_ads(audio)

        if not ads_started:
            transcript = transcribe_audio(temp_path)
            print(f"Transcript: {transcript[:150]}...")
            summary_text += " " + transcript
            start_time += SEGMENT_LENGTH_SEC
        else:
            print("Ads detected! Ending program monitoring.")

        os.remove(temp_path)
        time.sleep(1)  # avoid tight loop

    print("Generating summary with Ollama...")
    short_summary = ollama_summarize(summary_text)

    print("Summary:\n", short_summary)

    print("Sending summary email...")
    send_email("IPTV Program Summary", short_summary)


if __name__ == "__main__":
    main()
