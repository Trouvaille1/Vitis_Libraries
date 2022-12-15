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
#include "cos_lut.hpp"

namespace us {
namespace L1 {

template <typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
void cosV(input_window<T>* in1, output_window<T>* out) {
    aie::vector<T, VECDIM> op1 = aie::zeros<T, VECDIM>();
    aie::vector<T, VECDIM> op2 = aie::broadcast<T, VECDIM>(100);
    aie::vector<T, VECDIM> res = aie::zeros<T, VECDIM>();
    //	int32 value_int;
    int32 angle_index;

    for (unsigned i = 0; i < LEN; i += INCREMENT) {
        window_readincr_v(in1, op1);
        op1 = aie::mul(op1, op2);

        for (unsigned int j = 0; j < VECDIM; ++j) {
            //			value_int = op1[j] * 100;
            angle_index = ((int32)(op1[j]) % 628);
            res[j] = cos_lut[angle_index];
        }
        window_writeincr(out, res);
    }
};

} // namespace L1
} // namespace us
