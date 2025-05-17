import subprocess
import time
import threading

def run_flask():
    subprocess.run(["python", "app/flask_api.py"])

def run_streamlit():
    time.sleep(3)  # Give Flask time to start
    subprocess.run(["streamlit", "run", "streamlit_app.py"])

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask)
    streamlit_thread = threading.Thread(target=run_streamlit)

    flask_thread.start()
    streamlit_thread.start()

    flask_thread.join()
    streamlit_thread.join()
