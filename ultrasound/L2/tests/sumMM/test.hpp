/*
 * Copyright 2021 Xilinx, Inc.
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

#pragma once

#include "kernels.hpp"

namespace us {
namespace L2 {

class sumMM : public adf::graph {
   public:
    sumMM() {
        // Kernel definition
        sumMMKernel = adf::kernel::create(L1::sumMM<float, LENGTH, SPACE_DIMENSION, SIMD_DEPTH>);

        // input and output port
        input1_sumMM = adf::input_plio::create(adf::plio_32_bits, "data/input1.txt");
        input2_sumMM = adf::input_plio::create(adf::plio_32_bits, "data/input2.txt");
        output_sumMM = adf::output_plio::create(adf::plio_32_bits, "data/output.txt");

        // connections
        adf::connect<adf::window<WIN_SIZE_MATRIX> > input1_sumMM_q(input1_sumMM.out[0], sumMMKernel.in[0]);
        adf::connect<adf::window<WIN_SIZE_MATRIX> > input2_sumMM_q(input2_sumMM.out[0], sumMMKernel.in[1]);
        adf::connect<adf::stream> output_sumMM_q(sumMMKernel.out[0], output_sumMM.in[0]);

        // source kernel
        adf::source(sumMMKernel) = "sumMM/sumMM.cpp";

        // setting kernel ratio
        adf::runtime<adf::ratio>(sumMMKernel) = KERNEL_RATIO;

        // setting FIFO depth
        adf::fifo_depth(input1_sumMM_q) = FIFO_DEPTH;
        adf::fifo_depth(input2_sumMM_q) = FIFO_DEPTH;
        adf::fifo_depth(output_sumMM_q) = FIFO_DEPTH;
    }

    adf::input_plio input1_sumMM;
    adf::input_plio input2_sumMM;
    adf::output_plio output_sumMM;

   private:
    // Kernel declaration
    adf::kernel sumMMKernel;
};
} // namespace L2
} // namespace us
