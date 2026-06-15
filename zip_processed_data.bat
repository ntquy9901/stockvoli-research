@echo off
REM Script to package processed data for Google Colab upload
REM This creates a zip file of all processed VN30 stock data

echo 📦 Packaging processed VN30 data for Google Colab upload...
echo.

REM Check if processed data exists
if not exist "data\processed" (
    echo ❌ Error: data/processed directory not found!
    exit /b 1
)

REM Count processed files
for /f %%i in ('dir /b data\processed\*.csv 2^>nul ^| find /c /v ""') do set file_count=%%i
echo ✅ Found %file_count% processed stock files

if %file_count% LSS 30 (
    echo ⚠️  Warning: Expected 30 files, found %file_count%
)

REM Create zip file using PowerShell
echo 📦 Creating zip file: vn30_processed_data.zip
powershell -Command "Compress-Archive -Path 'data\processed\*.csv' -DestinationPath 'vn30_processed_data.zip' -Force"

if %ERRORLEVEL% EQU 0 (
    echo ✅ Zip file created: vn30_processed_data.zip

    REM Show file size
    for %%F in (vn30_processed_data.zip) do echo Size: %%~zF bytes

    echo.
    echo 📤 Upload Instructions:
    echo 1. Download vn30_processed_data.zip
    echo 2. Extract to Google Drive: My Drive\TimesFM_VN30\
    echo 3. Ensure folder structure: TimesFM_VN30\{TICKER}_processed.csv
    echo.
    echo 🚀 Ready for Google Colab testing!
) else (
    echo ❌ Error creating zip file
    exit /b 1
)

pause