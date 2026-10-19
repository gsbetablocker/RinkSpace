@echo off
call "D:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cl /nologo /O2 /std:c++17 /EHsc /MD /LD /Iinclude tbb_control.cpp /Fotbb_control.obj /link /OUT:tbb_control.dll /LIBPATH:lib tbb12.lib
