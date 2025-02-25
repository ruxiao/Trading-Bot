import webbrowser
import subprocess
import time
import os

# Change to the project directory
os.chdir('/Users/ruxiaoqian/tradebot')

# Start streamlit in a subprocess
proc = subprocess.Popen(['streamlit', 'run', 'app.py'], 
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)

# Wait for the server to start
time.sleep(3)

# Open the browser
webbrowser.open('http://localhost:8501')

print("Trading app launched! If browser doesn't open automatically, go to: http://localhost:8501")