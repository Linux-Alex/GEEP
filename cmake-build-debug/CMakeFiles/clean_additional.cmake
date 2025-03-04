# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles/GEEP_autogen.dir/AutogenUsed.txt"
  "CMakeFiles/GEEP_autogen.dir/ParseCache.txt"
  "GEEP_autogen"
  )
endif()
