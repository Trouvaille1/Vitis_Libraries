.. 
   Copyright (C) 2022-2023, Advanced Micro Devices, Inc.
   SPDX-License-Identifier: X11
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   X CONSORTIUM BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
   
   Except as contained in this notice, the name of Advanced Micro Devices 
   shall not be used in advertising or otherwise to promote the sale,
   use or other dealings in this Software without prior written authorization 
   from Advanced Micro Devices, Inc.

Vitis Motor Control Library
=============================

Motor Control Library is an open-sourced library written in C/C++ for accelerating developments of motor control applications. It has covered 4 algorithm-level L1 APIs including FOC, SVPWM__DUTY, PWM_GEN and QEI. These four APIs have AXI configuration interfaces and high integration level, that can be directly integrated into the system. The ap_fixed type has been adopted in these APIs which makes it easier to understand the physical meaning of variables' value. A virtual motor model is provided for doing the verifications of FOC in the Vitis environment.The 4 algorithm-level algorithms APIs implemented by Vitis Motor Control Library include:

- FOC: the API is for sensor based field-orientated control (FOC). The eight control modes it supports cover basic speed and torque control modes, as well as field-weakning control. Besides signal ports, AXI-Lite interface is provided for system control and monitor.
- SVPWM_DUTY: the API is the front-end for Space Vector Pulse Width Modulation (SVPWM) to calculate ratios. Besides signal ports, AXI-Lite interface is provided for system control and monitor.
- PWM_GEN: the API is the back-end for Space Vector Pulse Width Modulation (SVPWM) to generate output signals based on ratios. Besides signal ports, AXI-Lite interface is provided for system control and monitor.
- QEI: the API is for quadrature encoder interface(QEI). Besides signal ports, AXI-Lite interface is provided for system control and monitor.

Besides the four algorithm-level L1 APIs, from 24.1 release 12 new fine-grained function-level APIs are provided for supporting troditional IP integeration flow. These APIs are based on integer types and can simplify computational logic in suitable scenarios.The 12 new fine-grained function-level APIs include 1) angle_generation, 2) Clarke_Direct, 3) Clarke_Inverse, 4) demuxer_pi, 5) ps_iir_filter, 6) muxer_pi, 7) Park_Direct, 8) Park_Inverse, 9) PI_Control, 10) PI_Control_stream, 11) SVPWM and 12) voltage_modulation.


.. toctree::
   :caption: Introduction
   :maxdepth: 1

   Overview <overview.rst>
   Release Note <release.rst>
   Vitis Motor Control Library Tutorial <tutorial.rst>

.. toctree::
   :caption: L1 User Guide
   :maxdepth: 3

   API Document <guide_L1/api.rst>

.. toctree::
   :maxdepth: 2

   Design Internals <guide_L1/internals.rst>

.. toctree::
..    :caption: L2 User Guide
..    :maxdepth: 3

..    API Document <guide_L2/api.rst>

.. toctree::
..    :maxdepth: 2

..    Design Internals <guide_L2/internals.rst>

.. toctree::
   :caption: Benchmark 
   :maxdepth: 1 

   Benchmark <benchmark.rst>

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`

.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:
