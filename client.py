import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask server URL
SERVER_URL = "http://localhost:5000"

def test_server():
    try:
        response = requests.get(f"{SERVER_URL}/test")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Server Test Successful: {data}")
            return True
        else:
            logger.error(f"❌ Server Test Failed: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Error connecting to server: {e}")
        return False

def check_health():
    try:
        response = requests.get(f"{SERVER_URL}/health")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Health Check Successful: {data}")
            return True
        else:
            logger.error(f"❌ Health Check Failed: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking health: {e}")
        return False

def diarize_audio(file_path):
    try:
        with open(file_path, 'rb') as audio_file:
            files = {'file': audio_file}
            response = requests.post(f"{SERVER_URL}/diarize", files=files)

        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Diarization completed at {result['timestamp']}")
            
            print("\n🗣️ Diarization Result:\n")
            for segment in result['speaker_segments']:
                print(f"{segment['speaker']}: {segment['start_time']}s → {segment['end_time']}s")
        else:
            logger.error(f"❌ Error: {response.text}")
    except Exception as e:
        logger.error(f"❌ Error sending audio file: {e}")

if __name__ == "__main__":
    # 1. Test if server is alive
    if not test_server():
        exit(1)

    # 2. Check server health
    if not check_health():
        exit(1)

    # 3. Path to your test audio file
    audio_path = "Scala.wav"  # 👈 replace this with your local file

    # 4. Send audio for diarization
    diarize_audio(audio_path)
