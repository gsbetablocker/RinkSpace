@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_POINT_EDITOR.ps1" -Dataset "%~dp0events\ankara_2026\programmes\okada_mei_sp\source_datasets\shot8_20261014" -Media "D:\CodexData\RinkSpace_Data\events\ankara_2026\programmes\okada_mei_sp\results\shot8_review_20261014" -SourceMedia "D:\CodexData\RinkSpace_Data\events\ankara_2026\programmes\okada_mei_sp\results\shot8_review_20261014" -Model "%~dp0events\ankara_2026\models\shot8_8a28_20261014" -Port 48181 -NoOpen
exit /b %errorlevel%
