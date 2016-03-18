project(expofx)
cmake_minimum_required(VERSION 2.6)

set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-Wall -O2")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -O2")
set(expofx_SRC
integer.c
main.c
)

set(expofx_HDR 
integer.h
pefactorial.h
utility.h
power.h
)

add_executable(expofx ${expofx_SRC} ${expofx_HDR})

target_link_libraries(expofx -lm)

install(TARGETS expofx RUNTIME DESTINATION bin)
