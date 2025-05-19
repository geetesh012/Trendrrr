import subprocess
import time
import threading

def run_flask():
    try:
        # Run Flask API in a separate process (non-blocking)
        flask_proc = subprocess.Popen(["python", "app/flask_api.py"])
        # Wait for Flask process to finish (if ever)
        flask_proc.wait()
    except Exception as e:
        print(f"Error running Flask: {e}")

def run_streamlit():
    try:
        # Wait a few seconds for Flask to start before running Streamlit
        time.sleep(5)
        streamlit_proc = subprocess.Popen(["streamlit", "run", "streamlit_app.py"])
        streamlit_proc.wait()
    except Exception as e:
        print(f"Error running Streamlit: {e}")

if __name__ == "_main_":
    flask_thread = threading.Thread(target=run_flask)
    streamlit_thread = threading.Thread(target=run_streamlit)

    flask_thread.start()
    streamlit_thread.start()

    flask_thread.join()
    streamlit_thread.join()