import os
import tempfile
import time
import subprocess
import json
import smtplib
import logging
from email.mime.text import MIMEText

import streamlink
import whisper
from dotenv import load_dotenv

# === Configuração Inicial ===
load_dotenv()
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# === Constantes ===
IPTV_URL = os.getenv("IPTV_URL", "https://d277k9d1h9dro4.cloudfront.net/out/v1/293e7c3464824cbd8818ab8e49dc5fe9/index.m3u8")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
SEGMENT_LENGTH_SEC = 5
OLLAMA_CHUNK_SIZE_CHARS = 500

# === Inicialização do Whisper ===
try:
    whisper_model = whisper.load_model("base")
    logging.info("Modelo Whisper carregado com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar modelo Whisper: {e}")
    exit(1)

# === Funções Utilitárias ===

def download_stream_segment(url, duration, output_path):
    try:
        streams = streamlink.streams(url)
        stream = streams.get("best")
        if not stream:
            logging.warning("Qualidade 'best' não encontrada.")
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
        logging.error(f"Erro ao baixar segmento de stream: {e}")
        return False

def extract_audio(input_path, output_path):
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Erro ao extrair áudio com ffmpeg.") from e

def transcribe_audio(file_path):
    try:
        result = whisper_model.transcribe(file_path)
        return result.get("text", "")
    except Exception as e:
        logging.error(f"Erro na transcrição com Whisper: {e}")
        return ""

def update_summary(summary, new_chunk):
    prompt = f"""Você é um assistente de IA que resume transcrições de vídeo de um canal de IPTV legalmente acessado. 
    Seu único trabalho é atualizar um resumo contínuo do conteúdo. Você não está promovendo ou apoiando qualquer atividade ilegal, 
    apenas gerando um resumo informativo mas compacto baseado em transcrição de áudio.

    Instruções:
    1. Se o RESUMO ATUAL estiver vazio, crie um resumo inicial usando apenas o NOVO TRECHO DE TRANSCRIÇÃO.
    2. Se já houver um RESUMO ATUAL, analise o NOVO TRECHO DE TRANSCRIÇÃO e atualize o resumo anterior com novas informações relevantes.
    3. Mantenha o resumo conciso e informativo. Não inclua avisos, desculpas ou comentários fora do contexto.

    RESUMO ATUAL:
    {summary}

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
            return parsed.get('completion', summary).strip()
        except json.JSONDecodeError:
            return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao executar Ollama: {e.stderr}")
        return summary

def send_email(subject, body):
    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER]):
        logging.error("Informações de e-mail ausentes.")
        return
    try:
        msg = MIMEText(body)
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logging.info("E-mail enviado com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao enviar e-mail: {e}")

# === Função Principal ===

def main():
    logging.info("Monitoramento IPTV iniciado. Pressione Ctrl+C para parar.")
    summary = ""
    transcript_accum = ""

    try:
        while True:
            with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as temp_stream, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                stream_path = temp_stream.name
                audio_path = temp_audio.name

            if not download_stream_segment(IPTV_URL, SEGMENT_LENGTH_SEC, stream_path):
                logging.warning("Falha ao baixar segmento. Repetindo em 5s...")
                time.sleep(5)
                continue

            try:
                extract_audio(stream_path, audio_path)
                chunk = transcribe_audio(audio_path)
            except Exception as e:
                logging.error(f"Erro no processamento de áudio: {e}")
                continue
            finally:
                os.remove(stream_path)
                os.remove(audio_path)

            if chunk:
                transcript_accum += " " + chunk
                if len(transcript_accum) >= OLLAMA_CHUNK_SIZE_CHARS:
                    summary = update_summary(summary, transcript_accum)
                    transcript_accum = ""
                    logging.info(f"\nResumo atualizado:\n{summary}\n{'-'*40}")
                else:
                    logging.info(f"Transcrição acumulada: {len(transcript_accum)} caracteres.")
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("Interrompido pelo usuário. Gerando resumo final...")
        if transcript_accum:
            summary = update_summary(summary, transcript_accum)
        logging.info(f"Resumo Final:\n{summary}")
        send_email("Resumo IPTV", summary)
    except Exception as e:
        logging.critical(f"Erro inesperado: {e}")

# === Execução ===
if __name__ == "__main__":
    main()
