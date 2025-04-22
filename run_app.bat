@echo off
echo =======================================
echo    IPL Score Predictor - Start App
echo =======================================
echo.
echo Starting the app...
echo The app will open in your default browser.
echo.
echo To stop the app:
echo 1. Press Ctrl + C in this window
echo 2. OR close this window
echo.
echo Press any key to start the app...
pause > nul

"C:\Users\ad602\AppData\Roaming\Python\Python313\Scripts\streamlit.exe" run app.py

echo.
echo App has been stopped.
echo Press any key to exit...
pause > nul 
