#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Ceres::ceres_cuda_kernels" for configuration "Release"
set_property(TARGET Ceres::ceres_cuda_kernels APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Ceres::ceres_cuda_kernels PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/ceres_cuda_kernels.lib"
  )

list(APPEND _cmake_import_check_targets Ceres::ceres_cuda_kernels )
list(APPEND _cmake_import_check_files_for_Ceres::ceres_cuda_kernels "${_IMPORT_PREFIX}/lib/ceres_cuda_kernels.lib" )

# Import target "Ceres::ceres" for configuration "Release"
set_property(TARGET Ceres::ceres APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Ceres::ceres PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/ceres.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/ceres.dll"
  )

list(APPEND _cmake_import_check_targets Ceres::ceres )
list(APPEND _cmake_import_check_files_for_Ceres::ceres "${_IMPORT_PREFIX}/lib/ceres.lib" "${_IMPORT_PREFIX}/bin/ceres.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
