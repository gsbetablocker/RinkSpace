@echo off
setlocal
call "D:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "RS_BASE=D:\CodexData\RinkSpace_program_tracking_tmp\solver_acceleration"
set "RS_ENV=D:\CodexData\RinkSpace_program_tracking_tmp\gtsam-env"
set "RS_VENDOR=%RS_BASE%\round3\native\reference_callback\vendor"
set "RS_TBB=%RS_BASE%\round3\native\tbb_control"
set "RS_BIN=%~dp0bin"
if not exist "%RS_BIN%" mkdir "%RS_BIN%"
if errorlevel 1 exit /b 1
cd /d "%~dp0"
cl /nologo /O2 /fp:strict /std:c++17 /EHsc /MD /LD /DRS_EDGE_BUILD ^
  source_edge_native_prod.cpp /Fo"%RS_BIN%\source_edge_native.obj" ^
  /link /OUT:"%RS_BIN%\source_edge_native.dll" /IMPLIB:"%RS_BIN%\source_edge_native.lib"
if errorlevel 1 exit /b 1
cl /nologo /O2 /fp:strict /std:c++17 /EHsc /MD /bigobj /LD ^
  /I"%RS_ENV%\Library\include" ^
  /I"%RS_ENV%\include" ^
  /I"%RS_VENDOR%\eigen-5.0.1-h5112557_0\Library\include\eigen3" ^
  /I"%RS_VENDOR%\libboost-headers-1.90.0-h57928b3_1\Library\include" ^
  /I"%RS_VENDOR%\pybind11-global-3.0.3-pyhc8003f9_0\Library\include" ^
  /I"%RS_TBB%\include" ^
  native_source_edge_prod.cpp /Fo"%RS_BIN%\native_source_edge_quadratic_prod.obj" ^
  /link /OUT:"%RS_BIN%\native_source_edge_quadratic_prod.pyd" ^
  /LIBPATH:"%RS_ENV%\Library\lib" ^
  /LIBPATH:"%RS_ENV%\libs" ^
  /LIBPATH:"%RS_TBB%\lib" ^
  /LIBPATH:"%RS_BIN%" ^
  gtsam.lib python311.lib tbb12.lib source_edge_native.lib
if errorlevel 1 exit /b 1
exit /b 0
