# Setup Instructions
1. Run `pip install -r requirements.txt` to install requisite packages.
2. Ensure you have [OBS](https://obsproject.com/) installed.
3. Ensure you have [VB-Audio](https://vb-audio.com/Cable/) installed (for piping audio output to microphone for Text2Speech functionality). If on Mac, click on the Mac Download, then install the Pack108 dmg file.
4. Run `python3 camera.py` to begin the main script.
5. Add a new "Video Capture Device" using FaceTime HD Camera (if using Mac).
6. Open OBS and click on "Start Virtual Camera" in the bottom right set of options. This only needs to be done once — in the future, there's no need to open OBS while opening the Python script.
7. Open Zoom and join a meeting.
8. Go into Zoom Settings (Video) and make sure "Mirror my video" is not checked.
9. Click on the arrow adjacent to the "Start Video" button. Select "OBS Virtual Camera". The video should begin streaming.
10. Click on the arrow adjacent to the "Unmute" button. Select "VB-Cable" under "Select a Microphone". Then, in your sound settings (i.e. at the Mac status bar at the very top, speaker icon), select "VB-Cable". This routes all emitted sound from your device to play through VB-Cable. Note that you must be unmuted in Zoom in order for the sound piping to actually work.
