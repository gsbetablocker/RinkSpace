@echo off
call "D:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cl /nologo /O2 /std:c++17 /EHsc /MD /bigobj /LD ^
  /I"D:\CodexData\RinkSpace_program_tracking_tmp\gtsam-env\Library\include" ^
  /I"D:\CodexData\RinkSpace_program_tracking_tmp\gtsam-env\include" ^
  /I"vendor\eigen-5.0.1-h5112557_0\Library\include\eigen3" ^
  /I"vendor\libboost-headers-1.90.0-h57928b3_1\Library\include" ^
  /I"vendor\pybind11-global-3.0.3-pyhc8003f9_0\Library\include" ^
  /I"..\tbb_control\include" ^
  reference_callback.cpp /Foreference_callback.obj ^
  /link /OUT:reference_callback.pyd ^
  /LIBPATH:"D:\CodexData\RinkSpace_program_tracking_tmp\gtsam-env\Library\lib" ^
  /LIBPATH:"D:\CodexData\RinkSpace_program_tracking_tmp\gtsam-env\libs" ^
  /LIBPATH:"..\tbb_control\lib" ^
  gtsam.lib python311.lib tbb12.lib


