# Host-local MSVC 2052 output; keep VSLANG=2052 in configure and build processes.
# Pass with -DCMAKE_PROJECT_INCLUDE=<this file>; no source or DLL change.
# Build rules must contain these CP936 bytes, not UTF-8 text or mojibake.
string(ASCII 215 162 210 226 58 32 176 252 186 172 206 196 188 254 58 32 32 CMAKE_CL_SHOWINCLUDES_PREFIX)
set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "${CMAKE_CL_SHOWINCLUDES_PREFIX}")
set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "${CMAKE_CL_SHOWINCLUDES_PREFIX}")
