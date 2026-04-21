@echo off
REM Stock Price Linear Regression - Windows Runner
REM Usage: run.bat [command]
REM Commands: quickstart, train, report, install, clean

if "%1"=="" goto help
if "%1"=="quickstart" goto quickstart
if "%1"=="train" goto train
if "%1"=="report" goto report
if "%1"=="install" goto install
if "%1"=="clean" goto clean
goto help

:quickstart
    echo Running quickstart...
    python quickstart.py
    goto end

:train
    echo Training models...
    mkdir output 2>nul
    python scripts/train.py
    goto end

:report
    echo Generating diagnostic report...
    python scripts/report.py
    goto end

:install
    echo Installing dependencies...
    pip install -r requirements.txt
    goto end

:clean
    echo Cleaning output files...
    if exist output rmdir /s /q output
    goto end

:help
    echo Stock Price Linear Regression - Available commands:
    echo.
    echo   run.bat quickstart  - Run quickstart (generates prediction in ^<60s)
    echo   run.bat train       - Train all models using configs/
    echo   run.bat report      - Generate full diagnostic report
    echo   run.bat install     - Install dependencies
    echo   run.bat clean       - Remove output files
    goto end

:end
