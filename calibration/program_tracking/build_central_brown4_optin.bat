@echo off
call "D:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cl /nologo /O2 /fp:strict /LD /TC "D:\OneDrive\codex_projs\RinkSpace\three_frame_refit\program_tracking\central_brown4_axis_domain.c" /Fo"D:\CodexData\RinkSpace_program_tracking_tmp\solver_acceleration\central_brown4_optin\central_brown4.obj" /link /OUT:"D:\CodexData\RinkSpace_program_tracking_tmp\solver_acceleration\central_brown4_optin\central_brown4.dll" /IMPLIB:"D:\CodexData\RinkSpace_program_tracking_tmp\solver_acceleration\central_brown4_optin\central_brown4.lib"
exit /b %errorlevel%
