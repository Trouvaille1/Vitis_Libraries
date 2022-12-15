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

#include "kernels.hpp"

namespace us {
namespace L1 {

template <typename T, const unsigned int LEN, const unsigned int INCREMENT, const unsigned VECDIM>
void sumVVStreamIn1(input_stream<T>* in1, input_window<T>* in2, output_window<T>* out) {
    aie::vector<T, VECDIM> op1 = aie::zeros<T, VECDIM>();
    aie::vector<T, VECDIM> op2 = aie::zeros<T, VECDIM>();
    aie::vector<T, VECDIM> res = aie::zeros<T, VECDIM>();

    for (unsigned i = 0; i < LEN; i += INCREMENT) {
        op1 = readincr_v<SIMD_DEPTH>(in1);
        window_readincr_v(in2, op2);

        res = aie::add(op1, op2);

        window_writeincr(out, res);
    }
}

template <typename T, const unsigned int LEN, const unsigned int INCREMENT, const unsigned VECDIM>
void sumVV(input_window<T>* in1, input_window<T>* in2, output_window<T>* out) {
    aie::vector<T, VECDIM> op1 = aie::zeros<T, VECDIM>();
    aie::vector<T, VECDIM> op2 = aie::zeros<T, VECDIM>();
    aie::vector<T, VECDIM> res = aie::zeros<T, VECDIM>();

    for (unsigned i = 0; i < LEN; i += INCREMENT) {
        window_readincr_v(in1, op1);
        window_readincr_v(in2, op2);

        res = aie::add(op1, op2);

        window_writeincr(out, res);
    }
}

template <typename T, const unsigned int LEN, const unsigned VECDIM>
void sumVVStreamOut(input_window<T>* in1, input_window<T>* in2, output_stream<T>* out) {
    aie::vector<T, VECDIM> op1 = aie::zeros<T, VECDIM>();
    aie::vector<T, VECDIM> op2 = aie::zeros<T, VECDIM>();
    aie::vector<T, VECDIM> res = aie::zeros<T, VECDIM>();

    for (unsigned i = 0; i < LEN; i++) {
        window_readincr_v(in1, op1);
        window_readincr_v(in2, op2);

        res = aie::add(op1, op2);

        writeincr(out, res);
    }
}

} // namespace L1
} // namespace us

void sumVV_foc(input_window<float>* in1, input_window<float>* in2, output_window<float>* out) {
    aie::vector<float, SIMD_DEPTH> op1 = aie::zeros<float, SIMD_DEPTH>();
    aie::vector<float, SIMD_DEPTH> op2 = aie::zeros<float, SIMD_DEPTH>();
    aie::vector<float, SIMD_DEPTH> res = aie::zeros<float, SIMD_DEPTH>();

    for (unsigned i = 0; i < LENGTH_FOC; i += SIMD_DEPTH) {
        window_readincr_v(in1, op1);
        window_readincr_v(in2, op2);

        res = aie::add(op1, op2);

        window_writeincr(out, res);
    }
}

// void sumVVStreamIn1(input_stream<float>* in1, input_window<float>* in2, output_window<float>* out){
//
//	aie::vector<float, SIMD_DEPTH> op1 = aie::zeros<float, SIMD_DEPTH>();
//	aie::vector<float, SIMD_DEPTH> op2 = aie::zeros<float, SIMD_DEPTH>();
//	aie::vector<float, SIMD_DEPTH> res = aie::zeros<float, SIMD_DEPTH>();
//
//	for(unsigned i = 0; i < LENGTH; i+=SIMD_DEPTH){
//
//		op1 = readincr_v<SIMD_DEPTH>(in1);
//		window_readincr_v(in2, op2);
//
//		res = aie::add(op1, op2);
//
//		window_writeincr(out, res);
//	}
//
// }

void sumVVStreamIn2(input_window<float>* in1, input_stream<float>* in2, output_window<float>* out) {
    aie::vector<float, SIMD_DEPTH> op1 = aie::zeros<float, SIMD_DEPTH>();
    aie::vector<float, SIMD_DEPTH> op2 = aie::zeros<float, SIMD_DEPTH>();
    aie::vector<float, SIMD_DEPTH> res = aie::zeros<float, SIMD_DEPTH>();

    for (unsigned i = 0; i < LENGTH; i += INCREMENT_VECTOR) {
        window_readincr_v(in1, op1);
        op2 = readincr_v<SIMD_DEPTH>(in2);

        res = aie::add(op1, op2);

        window_writeincr(out, res);
    }
}

void sumVV(input_window<float>* in1, input_window<float>* in2, output_window<float>* out) {
    aie::vector<float, SIMD_DEPTH> op1 = aie::zeros<float, SIMD_DEPTH>();
    aie::vector<float, SIMD_DEPTH> op2 = aie::zeros<float, SIMD_DEPTH>();
    aie::vector<float, SIMD_DEPTH> res = aie::zeros<float, SIMD_DEPTH>();

    for (unsigned i = 0; i < POINTS_PER_ITERATION; i++) {
        window_readincr_v(in1, op1);
        window_readincr_v(in2, op2);

        res = aie::add(op1, op2);

        window_writeincr(out, res);
    }
}

void sumVVStream(input_window<float>* in1, input_window<float>* in2, output_stream<float>* out) {
    aie::vector<float, SIMD_DEPTH> op1 = aie::zeros<float, SIMD_DEPTH>();
    aie::vector<float, SIMD_DEPTH> op2 = aie::zeros<float, SIMD_DEPTH>();
    aie::vector<float, SIMD_DEPTH> res = aie::zeros<float, SIMD_DEPTH>();

    for (unsigned i = 0; i < POINTS_PER_ITERATION; i++) {
        window_readincr_v(in1, op1);
        window_readincr_v(in2, op2);

        res = aie::add(op1, op2);

        writeincr(out, res);
    }
}
