execute_process(COMMAND "/home/torobo/test/src/test/hack19/build/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/torobo/test/src/test/hack19/build/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
