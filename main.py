import os
import cv2
import time
import json
import tempfile
import subprocess
import smtplib
import logging
from threading import Thread
from email.mime.text import MIMEText
from dotenv import load_dotenv

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QHBoxLayout, QVBoxLayout, QPushButton
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject

import streamlink
import whisper

# === Configuração ===
load_dotenv()
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

IPTV_URL = os.getenv("IPTV_URL", "https://d277k9d1h9dro4.cloudfront.net/out/v1/293e7c3464824cbd8818ab8e49dc5fe9/index.m3u8")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
SEGMENT_LENGTH_SEC = 5
OLLAMA_CHUNK_SIZE_CHARS = 500

# === Whisper ===
whisper_model = whisper.load_model("base")

# === Transcription Worker ===
class TranscriptionWorker(QObject):
    update_summary_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = True
        self.summary = ""
        self.transcript_accum = ""

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as temp_stream, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                stream_path = temp_stream.name
                audio_path = temp_audio.name

            if not self.download_segment(IPTV_URL, SEGMENT_LENGTH_SEC, stream_path):
                time.sleep(5)
                continue

            try:
                self.extract_audio(stream_path, audio_path)
                chunk = whisper_model.transcribe(audio_path).get("text", "")
            except Exception as e:
                logging.error(f"Erro: {e}")
                continue
            finally:
                os.remove(stream_path)
                os.remove(audio_path)

            if chunk:
                self.transcript_accum += " " + chunk
                if len(self.transcript_accum) >= OLLAMA_CHUNK_SIZE_CHARS:
                    self.summary = self.update_summary(self.summary, self.transcript_accum)
                    self.transcript_accum = ""
                    self.update_summary_signal.emit(self.summary)

            time.sleep(1)

    def download_segment(self, url, duration, output_path):
        try:
            streams = streamlink.streams(url)
            stream = streams.get("best")
            if not stream:
                return False

            with stream.open() as fd, open(output_path, "wb") as out_file:
                start = time.time()
                while time.time() - start < duration:
                    data = fd.read(1024)
                    if not data:
                        break
                    out_file.write(data)
            return True
        except Exception as e:
            logging.error(f"Download falhou: {e}")
            return False

    def extract_audio(self, input_path, output_path):
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def update_summary(self, current, new_chunk):
        prompt = f"""
Você é um assistente de IA que resume transcrições de vídeo de um canal de IPTV legalmente acessado.

RESUMO ATUAL:
{current}

NOVO TRECHO DE TRANSCRIÇÃO:
{new_chunk}

RESUMO ATUALIZADO:
"""

        try:
            result = subprocess.run(
                ['ollama', 'run', OLLAMA_MODEL, prompt],
                capture_output=True, text=True, check=True
            )
            try:
                parsed = json.loads(result.stdout)
                return parsed.get('completion', current).strip()
            except json.JSONDecodeError:
                return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logging.error(f"Ollama erro: {e}")
            return current

# === Interface PyQt5 ===
class IPTVApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BroadcastBuddy - IPTV com resumo")
        self.setGeometry(200, 100, 1200, 700)

        # Layouts
        h_layout = QHBoxLayout()
        v_layout = QVBoxLayout()

        # Vídeo
        self.video_label = QLabel("Carregando stream...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(800, 600)
        h_layout.addWidget(self.video_label)

        # Resumo
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setFixedWidth(380)
        h_layout.addWidget(self.summary_box)

        # Botão sair
        self.quit_btn = QPushButton("Sair")
        self.quit_btn.clicked.connect(self.close)
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.quit_btn)
        self.setLayout(v_layout)

        # Worker de transcrição
        self.worker = TranscriptionWorker()
        self.worker.update_summary_signal.connect(self.update_summary)
        self.thread = Thread(target=self.worker.run, daemon=True)
        self.thread.start()

        # Timer de vídeo
        self.cap = self.init_stream_capture()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_stream_capture(self):
        try:
            streams = streamlink.streams(IPTV_URL)
            stream_url = streams["best"].url
            cap = cv2.VideoCapture(stream_url)
            return cap
        except Exception as e:
            logging.error(f"Erro ao capturar stream: {e}")
            return None

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qt_image)
                self.video_label.setPixmap(pix.scaled(
                    self.video_label.width(),
                    self.video_label.height(),
                    Qt.KeepAspectRatio
                ))

    def update_summary(self, summary_text):
        self.summary_box.setPlainText(summary_text)

    def closeEvent(self, event):
        self.worker.stop()
        if self.cap:
            self.cap.release()
        event.accept()

# === Execução ===
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = IPTVApp()
    window.show()
    sys.exit(app.exec_())
