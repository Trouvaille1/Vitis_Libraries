# Copyright (C) 2019-2022, Xilinx, Inc.
# Copyright (C) 2022-2023, Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# vitis hls makefile-generator v2.0.0

set CSIM 1
set CSYNTH 1
set COSIM 1
set VIVADO_SYN 1
set VIVADO_IMPL 1
set CUR_DIR [pwd]
set XF_PROJ_ROOT $CUR_DIR/../../../..
set XPART xcu280-fsvh2892-2L-e

set PROJ "join_dut.prj"
set SOLN "sol"

if {![info exists CLKP]} {
  set CLKP 300MHz
}

open_project -reset $PROJ

add_files "join_dut.cpp" -cflags "-I${XF_PROJ_ROOT}/L1/include/hw"
add_files -tb "join_test.cpp" -cflags "-I${XF_PROJ_ROOT}/L1/include/hw -std=c++14"
set_top join_dut

open_solution -reset $SOLN



set_part $XPART
create_clock -period $CLKP

if {$CSIM == 1} {
  csim_design
}

if {$CSYNTH == 1} {
  csynth_design
}

if {$COSIM == 1} {
  cosim_design
}

if {$VIVADO_SYN == 1} {
  export_design -flow syn -rtl verilog
}

if {$VIVADO_IMPL == 1} {
  export_design -flow impl -rtl verilog
}

exit