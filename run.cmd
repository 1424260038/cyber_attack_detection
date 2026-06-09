@echo off
setlocal EnableExtensions
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set "APP_NAME=CyberDD Cyber Attack Detection System"

:menu
cls
echo ========================================
echo   %APP_NAME%
echo ========================================
echo.

call :check_python
if errorlevel 1 (
    echo [ERROR] Python was not found or cannot run.
    echo Install Python 3.8+ and make sure python.exe is available in PATH.
    echo.
    pause
    exit /b 1
)

echo Select a run mode:
echo   [1] Train model (CNN-LSTM)
echo   [2] Test model
echo   [3] Start API backend
echo   [4] Start Web frontend
echo   [5] Evaluate knowledge graph
echo   [6] Run quick demo test
echo   [7] Run full quality checks
echo   [8] Build or refresh demo system
echo   [9] Generate demo data and preprocessor
echo   [10] Export TorchScript model
echo   [11] Generate project report
echo   [12] Package release zip
echo   [0] Exit
echo.

set "choice="
set /p "choice=Enter option (0-12): "
if errorlevel 1 exit /b 0

if "%choice%"=="1" goto train
if "%choice%"=="2" goto test
if "%choice%"=="3" goto api
if "%choice%"=="4" goto web
if "%choice%"=="5" goto kg_eval
if "%choice%"=="6" goto demo
if "%choice%"=="7" goto system_check
if "%choice%"=="8" goto build_demo_system
if "%choice%"=="9" goto prepare_demo
if "%choice%"=="10" goto export_model
if "%choice%"=="11" goto project_report
if "%choice%"=="12" goto package_release
if "%choice%"=="0" exit /b 0

if "%choice%"=="" goto menu
echo Invalid option. Press any key to try again...
pause >nul
goto menu

:train
cls
echo ========================================
echo Training model...
echo ========================================
python main.py --mode train --model cnn_lstm --epochs 100 --device cpu --output_dir checkpoints
set "STATUS=%errorlevel%"
goto finish

:test
cls
echo ========================================
echo Testing model...
echo ========================================
python main.py --mode test --model cnn_lstm --checkpoint checkpoints/best_model.pth --device cpu
set "STATUS=%errorlevel%"
goto finish

:api
cls
echo ========================================
echo Starting API backend...
echo ========================================
echo Service: http://localhost:8000
echo Docs:    http://localhost:8000/docs
echo.
echo Tip: start this before the Web frontend.
echo Press Ctrl+C to stop the service, then press any key to return to the menu.
echo.
python api.py
set "STATUS=%errorlevel%"
goto finish

:web
cls
echo ========================================
echo Starting Web frontend...
echo ========================================
if not exist "web\package.json" (
    echo [ERROR] web\package.json was not found.
    echo.
    pause
    goto menu
)

call :check_pnpm
if errorlevel 1 (
    echo [ERROR] pnpm was not found.
    echo Install it with: npm install -g pnpm
    echo.
    pause
    goto menu
)

echo Web directory: web
echo URL after startup: http://localhost:5173
echo.
echo Tip: make sure the API backend is already running with option 3.
echo Press Ctrl+C to stop the frontend, then press any key to return to the menu.
echo.
pushd web
call pnpm run dev
set "WEB_STATUS=%errorlevel%"
popd
set "STATUS=%WEB_STATUS%"
goto finish

:kg_eval
cls
echo ========================================
echo Evaluating knowledge graph...
echo ========================================
python main.py --mode evaluate_kg --device cpu
set "STATUS=%errorlevel%"
goto finish

:demo
cls
echo ========================================
echo Running quick demo test...
echo ========================================
python test_demo.py
set "STATUS=%errorlevel%"
goto finish

:system_check
cls
echo ========================================
echo Running full quality checks...
echo ========================================
echo This may take a few minutes. It runs backend tests, smoke tests, and frontend checks.
echo.
python tools\run_all_checks.py
set "STATUS=%errorlevel%"
goto finish

:prepare_demo
cls
echo ========================================
echo Generating demo data and preprocessor...
echo ========================================
python tools\generate_demo_dataset.py --output data\demo_traffic.csv --samples-per-class 120 --input-dim 64
if errorlevel 1 (
    set "STATUS=%errorlevel%"
    goto finish
)
python tools\fit_preprocessor.py --input data\demo_traffic.csv --output artifacts\preprocessor.json --input-dim 64
set "STATUS=%errorlevel%"
goto finish

:build_demo_system
cls
echo ========================================
echo Building or refreshing demo system...
echo ========================================
echo This may take several minutes. Use Ctrl+C only if you want to cancel.
echo.
python tools\build_demo_system.py
set "STATUS=%errorlevel%"
goto finish

:export_model
cls
echo ========================================
echo Exporting TorchScript model...
echo ========================================
python tools\export_model.py --checkpoint checkpoints\best_model.pth --output artifacts\model.pt
set "STATUS=%errorlevel%"
goto finish

:project_report
cls
echo ========================================
echo Generating project report...
echo ========================================
python tools\generate_model_manifest.py
if errorlevel 1 (
    set "STATUS=%errorlevel%"
    goto finish
)
python tools\generate_project_report.py
if errorlevel 1 (
    set "STATUS=%errorlevel%"
    goto finish
)
echo Report: outputs\project_report.md
set "STATUS=0"
goto finish

:package_release
cls
echo ========================================
echo Packaging release zip...
echo ========================================
python tools\package_release.py
if errorlevel 1 (
    set "STATUS=%errorlevel%"
    goto finish
)
echo Release package: release\cyberdd_release.zip
set "STATUS=0"
goto finish

:check_python
where python >nul 2>&1
if errorlevel 1 exit /b 1
python --version >nul 2>&1
exit /b %errorlevel%

:check_pnpm
where pnpm.cmd >nul 2>&1
if not errorlevel 1 exit /b 0
where pnpm >nul 2>&1
exit /b %errorlevel%

:finish
echo.
if "%STATUS%"=="0" (
    echo [OK] Finished successfully.
) else (
    echo [ERROR] Failed with exit code %STATUS%.
)
echo.
pause
goto menu
