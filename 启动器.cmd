@echo off
setlocal
cd /d "%~dp0"
call run.cmd
exit /b %errorlevel%
