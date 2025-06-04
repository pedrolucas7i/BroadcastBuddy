import os
import tempfile
import time
import subprocess
import json
import smtplib
import logging
from email.mime.text import MIMEText
from threading import Thread

import streamlink
import whisper
from dotenv import load_dotenv

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject

# === Configuração ===
load_dotenv()
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

IPTV_URL = os.getenv("IPTV_URL", "https://d277k9d1h9dro4.cloudfront.net/out/v1/293e7c3464824cbd8818ab8e49dc5fe9/index.m3u8")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
SEGMENT_LENGTH_SEC = 5
OLLAMA_CHUNK_SIZE_CHARS = 500

# === Inicialização Whisper ===
try:
    whisper_model = whisper.load_model("base")
    logging.info("Whisper carregado.")
except Exception as e:
    logging.error(f"Erro ao carregar Whisper: {e}")
    exit(1)

# === Backend Worker (transcrição e resumo) ===
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

            if not self.download_stream_segment(IPTV_URL, SEGMENT_LENGTH_SEC, stream_path):
                time.sleep(5)
                continue

            try:
                self.extract_audio(stream_path, audio_path)
                chunk = whisper_model.transcribe(audio_path).get("text", "")
            except Exception as e:
                logging.error(f"Erro no processamento de áudio: {e}")
                continue
            finally:
                os.remove(stream_path)
                os.remove(audio_path)

            if chunk:
                self.transcript_accum += " " + chunk
                if len(self.transcript_accum) >= OLLAMA_CHUNK_SIZE_CHARS:
                    self.summary = self.update_summary(self.summary, self.transcript_accum)
                    self.update_summary_signal.emit(self.summary)
                    self.transcript_accum = ""

            time.sleep(1)

    def download_stream_segment(self, url, duration, output_path):
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
            logging.error(f"Erro ao baixar segmento: {e}")
            return False

    def extract_audio(self, input_path, output_path):
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def update_summary(self, current, new_chunk):
        prompt = f"""Você é um assistente de IA que resume transcrições de vídeo de um canal de IPTV legalmente acessado. 
Seu único trabalho é atualizar um resumo contínuo do conteúdo.

RESUMO ATUAL:
{current}

NOVO TRECHO DE TRANSCRIÇÃO:
{new_chunk}

RESUMO ATUALIZADO:"""

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
            logging.error(f"Ollama error: {e}")
            return current

# === Interface Gráfica ===
class IPTVApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BroadcastBuddy - IPTV ao vivo com resumo")
        self.setGeometry(100, 100, 900, 600)

        layout = QVBoxLayout()

        self.video_label = QLabel("Abrindo player externo...")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        layout.addWidget(self.summary_box)

        self.quit_button = QPushButton("Encerrar")
        self.quit_button.clicked.connect(self.close)
        layout.addWidget(self.quit_button)

        self.setLayout(layout)

        # Iniciar player externo
        self.play_live_stream()

        # Iniciar transcrição em thread separada
        self.worker = TranscriptionWorker()
        self.worker.update_summary_signal.connect(self.update_summary)
        self.thread = Thread(target=self.worker.run, daemon=True)
        self.thread.start()

    def play_live_stream(self):
        # Abre com ffplay em processo separado
        subprocess.Popen([
            "ffplay", "-loglevel", "quiet", "-i", IPTV_URL, "-x", "480", "-y", "270", "-noborder"
        ])

    def update_summary(self, new_summary):
        self.summary_box.setPlainText(new_summary)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

# === Execução ===
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = IPTVApp()
    window.show()
    sys.exit(app.exec_())
