# Host stuff
HOST_SRCDIR := ${HOME_DIR}/host
HOST_O_DIR := build/host
HOST_C_TARGET := host.exe

HOST_HEADERS = $(wildcard ${HOST_SRCDIR}/*.hpp)
HOST_SRCS = $(wildcard ${HOST_SRCDIR}/*.cpp)

NPU_UTILS_SRCS = ${HOME_DIR}/common/npu_utils.cpp
NPU_UTILS_HEADERS = ${HOME_DIR}/common/npu_utils.hpp
NPU_UTILS_HEADERS = ${HOME_DIR}/common/vector_view.hpp

NPU_UTILS_OBJS = ${HOST_O_DIR}/npu_utils.o

HOST_OBJS = $(patsubst $(HOST_SRCDIR)/%.cpp,$(HOST_O_DIR)/%.o,$(HOST_SRCS))

CXX := g++-13
ifeq ($(DEVICE),npu1)
	CXXFLAGS += -c
	CXXFLAGS += -std=c++23
	CXXFLAGS += -ggdb
	CXXFLAGS += -I/opt/xilinx/xrt/include
	CXXFLAGS += -I/usr/include/boost
	CXXFLAGS += -I${HOME_DIR}/host
	CXXFLAGS += -I${HOME_DIR}/common

	LDFLAGS += -lm
	LDFLAGS += -L/opt/xilinx/xrt/lib
	LDFLAGS += -Wl,-rpath,/opt/xilinx/xrt/lib
	LDFLAGS += -lxrt_coreutil
	LDFLAGS += -lboost_program_options -lboost_filesystem
else ifeq ($(DEVICE),npu2)
	CXXFLAGS += -c
	CXXFLAGS += -std=c++23
	CXXFLAGS += -ggdb
	CXXFLAGS += -I/opt/xilinx/xrt/include
	CXXFLAGS += -I/usr/include/boost
	CXXFLAGS += -I${HOME_DIR}/host
	CXXFLAGS += -I${HOME_DIR}/common

	LDFLAGS += -lm
	LDFLAGS += -L/opt/xilinx/xrt/lib
	LDFLAGS += -Wl,-rpath,/opt/xilinx/xrt/lib
	LDFLAGS += -lxrt_coreutil
	LDFLAGS += -lboost_program_options -lboost_filesystem
endif



${HOST_C_TARGET}: ${HOST_OBJS} ${NPU_UTILS_OBJS}
	mkdir -p ${HOST_O_DIR}
	echo ${HOST_OBJS}
	$(CXX) -o "$@" $(+) $(LDFLAGS)

$(HOST_O_DIR)/%.o: $(HOST_SRCDIR)/%.cpp $(HOST_HEADERS)
	-@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o "$@" "$<"

$(HOST_O_DIR)/npu_utils.o: $(NPU_UTILS_SRCS) $(NPU_UTILS_HEADERS)
	-@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o "$@" "$<"

