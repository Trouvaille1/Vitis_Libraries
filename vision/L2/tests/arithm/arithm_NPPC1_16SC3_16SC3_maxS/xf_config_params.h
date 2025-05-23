/*
 * Copyright 2022 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _XF_ARITHM_CONFIG_H_
#define _XF_ARITHM_CONFIG_H_

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "hls_stream.h"

#include <ap_int.h>

#include "core/xf_arithm.hpp"

#define HEIGHT 1080
#define WIDTH 1920

#define XF_CV_DEPTH_IN_1 2
#define XF_CV_DEPTH_IN_2 2

#define XF_CV_DEPTH_OUT_1 2

// Resolve pixel precision:

#define ARRAY 0
#define SCALAR 1
#define ENABLE_ARG_EXTRA 0
#define ENABLE_PARM_EXTRA 0
#define FUNCT_NUM 20
// macros for accel
#define FUNCT_NAME max
// OpenCV reference macros
#define CV_FUNCT_NAME max

#if ENABLE_ARG_EXTRA
#define EXTRA_ARG 0.5
#define CV_EXTRA_ARG 0.5
#endif

#if ENABLE_PARM_EXTRA
#define EXTRA_PARM XF_CONVERT_POLICY_SATURATE
#endif

#if FUNCT_NUM == 0
// FUNCTION_NAME absdiff
#elif FUNCT_NUM == 1
// FUNCTION_NAME add
#elif FUNCT_NUM == 2
// FUNCTION_NAME addS
#elif FUNCT_NUM == 3
// FUNCTION_NAME bitwise_and
#elif FUNCT_NUM == 4
// FUNCTION_NAME bitwise_not
#define FUNCT_BITWISENOT
#elif FUNCT_NUM == 5
// FUNCTION_NAME bitwise_or
#elif FUNCT_NUM == 6
// FUNCTION_NAME bitwise_xor
#elif FUNCT_NUM == 7
// FUNCTION_NAME compare_EQ
#define FUNCT_COMPARE
#elif FUNCT_NUM == 8
// FUNCTION_NAME compare_GE
#define FUNCT_COMPARE
#elif FUNCT_NUM == 9
// FUNCTION_NAME compare_GT
#define FUNCT_COMPARE
#elif FUNCT_NUM == 10
// FUNCTION_NAME compare_LE
#define FUNCT_COMPARE
#elif FUNCT_NUM == 11
// FUNCTION_NAME compare_LT
#define FUNCT_COMPARE
#elif FUNCT_NUM == 12
// FUNCTION_NAME compare_NE
#define FUNCT_COMPARE
#elif FUNCT_NUM == 13
// FUNCTION_NAME compareS_EQ
#define FUNCT_COMPARE
#elif FUNCT_NUM == 14
// FUNCTION_NAME compareS_GE
#define FUNCT_COMPARE
#elif FUNCT_NUM == 15
// FUNCTION_NAME compareS_GT
#define FUNCT_COMPARE
#elif FUNCT_NUM == 16
// FUNCTION_NAME compareS_LE
#define FUNCT_COMPARE
#elif FUNCT_NUM == 17
// FUNCTION_NAME compareS_LT
#define FUNCT_COMPARE
#elif FUNCT_NUM == 18
// FUNCTION_NAME compareS_NE
#define FUNCT_COMPARE
#elif FUNCT_NUM == 19
// FUNCTION_NAME max
#elif FUNCT_NUM == 20
// FUNCTION_NAME maxS
#elif FUNCT_NUM == 21
// FUNCTION_NAME minS
#elif FUNCT_NUM == 22
// FUNCTION_NAME min
#elif FUNCT_NUM == 23
// FUNCTION_NAME multiply
#define FUNCT_MULTIPLY
#elif FUNCT_NUM == 24
// FUNCTION_NAME sub
#elif FUNCT_NUM == 25
// FUNCTION_NAME subS
#endif

#define GRAY 0
#define RGB 1

#define T_8U 0
#define T_16S 1

#define NPPCX XF_NPPC1

#define IN_TYPE XF_16SC3
#define OUT_TYPE XF_16SC3

#define CV_IN_TYPE CV_16SC3
#define CV_OUT_TYPE CV_16SC3

#define INPUT_PTR_WIDTH 64
#define OUTPUT_PTR_WIDTH 64

#endif
// end of _XF_ARITHM_CONFIG_H_
