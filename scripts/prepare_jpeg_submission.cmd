@echo off
setlocal

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
if not defined PYTHON_BIN set "PYTHON_BIN=%REPO_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON_BIN%" set "PYTHON_BIN=python"
if not defined BTS_RENDER_ROOT set "BTS_RENDER_ROOT=%REPO_ROOT%\outputs"
if not defined BTS_SUBMISSION_ROOT set "BTS_SUBMISSION_ROOT=%REPO_ROOT%\submission_outputs"
if not defined BTS_SCENES_ROOT set "BTS_SCENES_ROOT=%REPO_ROOT%\data\bts_scenes"
if not defined BTS_MANIFESTS_ROOT set "BTS_MANIFESTS_ROOT=%REPO_ROOT%\runs\manifests"
if not defined BTS_JPEG_REPORT set "BTS_JPEG_REPORT=%REPO_ROOT%\runs\submission\jpeg_report.json"
if not defined BTS_JPEG_QUALITY set "BTS_JPEG_QUALITY=99"
if not defined BTS_SUBMISSION_MAX_BYTES set "BTS_SUBMISSION_MAX_BYTES=350000000"

cd /d "%REPO_ROOT%"
set "PYTHONPATH=%REPO_ROOT%\src;%PYTHONPATH%"
"%PYTHON_BIN%" -m bts_nvs.submission.prepare_jpeg ^
  --source_root "%BTS_RENDER_ROOT%" ^
  --output_root "%BTS_SUBMISSION_ROOT%" ^
  --scenes_root "%BTS_SCENES_ROOT%" ^
  --manifests_root "%BTS_MANIFESTS_ROOT%" ^
  --report_path "%BTS_JPEG_REPORT%" ^
  --quality "%BTS_JPEG_QUALITY%" ^
  --max_bytes "%BTS_SUBMISSION_MAX_BYTES%" ^
  %*

exit /b %ERRORLEVEL%
