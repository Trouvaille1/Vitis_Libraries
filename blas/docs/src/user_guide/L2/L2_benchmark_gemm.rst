.. Copyright © 2019–2023 Advanced Micro Devices, Inc

.. `Terms and Conditions <https://www.amd.com/en/corporate/copyright>`_.

.. meta::
   :keywords: BLAS, Library, Vitis BLAS Library, L2, level 2
   :description: Vitis BLAS library level 2 application programming interface reference. Intel Math Kernel Library provides performance improvement of math functions, e.g. GEMM, when running with Intel processors.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _benchmark_gemm_l2:

***********************
L2 GEMM Benchmark
***********************

1. gemm_4CU
================

This example resides in the ``L2/benchmarks/memKernel/gemm_4CU`` directory. The tutorial provides a step-by-step guide that covers commands for building and running the kernel. It performs the matrix-matrix multiplication (A * B = C); M is number of rows of matrix A/C, K is number of columns of matrix A/number of rows of matrix B, and N is number of columns of matrix B/C.

1.1 Executable Usage
------------------------

1.1.1 Work Directory (Step 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The steps for library download and environment setup can be found [here](https://github.com/Xilinx/Vitis_Libraries/tree/main/blas/L2/benchmarks#building). For getting the design:

.. code-block:: bash 

   cd L2/benchmarks/memKernel/gemm_4CU
   

1.1.2 Build the Kernel (Step 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following make command to build your XCLBIN and host binary targeting a specific device. This process will take a long time, maybe couple of hours.

.. code-block:: bash 

    make run TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms PLATFORM=xilinx_u250_xdma_201830_2


1.1.3 Run the Kernel (Step 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get the benchmark results, run the following command:

gemm_4CU Input Arguments:

.. code-block:: bash 

    <host application> <xclbin> m k n


For example:

.. code-block:: bash 

    build_dir.hw.xilinx_u250_xdma_201830_2/host.exe build_dir.hw.xilinx_u250_xdma_201830_2/blas.xclbin 64 64 64


1.1.4 Example Output (Step 4)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash 

    Added GEMM 64x64x64  In kernel 0 Added instruction GEMM (64x64 * 64x64) 
    Added GEMM 64x64x64  In kernel 1 Added instruction GEMM (64x64 * 64x64) 
    Added GEMM 64x64x64  In kernel 2 Added instruction GEMM (64x64 * 64x64) 
    Added GEMM 64x64x64  In kernel 3 Added instruction GEMM (64x64 * 64x64) 
    Added GEMM 64x64x64  Found Platform
    Platform Name: Xilinx
    INFO: device name is: xilinx_u250_xdma_201830_2
    INFO: Importing build_dir.hw.xilinx_u250_xdma_201830_2/blas.xclbin
    Loading: 'build_dir.hw.xilinx_u250_xdma_201830_2/blas.xclbin'
    INFO: created kernels
    loadXclbin  6960.979134 msec
    create kernels  13.595438 msec
    create buffers  0.176534 msec
    INFO: transferred data to kernel 0
    INFO: transferred data to kernel 1
    INFO: transferred data to kernel 2
    INFO: transferred data to kernel 3
    copy to kernels  0.884381 msec
    INFO: Executed kernel 0
    INFO: Executed kernel 1
    INFO: Executed kernel 2
    INFO: Executed kernel 3
    call kernels  0.398135 msec
    INFO: Transferred data from kernel0
    INFO: Transferred data from kernel1
    INFO: Transferred data from kernel2
    INFO: Transferred data from kernel3
    copyFromFpga  0.260636 msec
    total  6976.308826 msec
    subtotalFpga  1.750123 msec
    DATA_CSV:,DdrWidth,Freq,M,K,N,Ops,KernelCycles,TimeKernelMs,TimeApiMs,EffKernelPct,EffApiPct,PerfKernelTops,PerfApiTops
    DATA_CSV:,16,242.000000,64,64,64,2146304,2639,0.010905,1.750123,38.802577,0.241778,0.199516,0.001226
    
    ###########  Op Gemm  ###########
      C = postScale(A * B + X) 64x64 = 64x64 * 64x64 + 64 x 64
      Comparing ...
      Compared 4096 values:  exact match 1281  within tolerance 2815  mismatch 0
    Gemm C Matches
    pass


1.2 Profiling
----------------

The xclbin could be built in 242 MHz.
The hardware resource utilization and benchmark results are shown in the following two tables.

*Table 1 Hardware Resources*

+------------+----------+--------+-------+--------+---------+
|    Name    |   LUT    |  BRAM  |  URAM |   DSP  |    FF   |
+============+==========+========+=======+========+=========+
| blasKernel | 250679   | 94     | 24    | 1224   | 430512  |
+------------+----------+--------+-------+--------+---------+

*Table 2 Benchmark Results*

+------+------+------+------------------------------+--------------------------+-----------------+
|  M   |  N   |  K   |  Kernel Execution Time [ms]  |  API Execution Time [ms] | Kernel Eff [%]  |  
+======+======+======+==============================+==========================+=================+
| 64   | 64   | 64   | 0.010905                     | 1.750123                 | 38.802577       | 
+------+------+------+------------------------------+--------------------------+-----------------+
| 128  | 128  | 128  | 0.048517                     | 13.802416                | 69.772592       | 
+------+------+------+------------------------------+--------------------------+-----------------+
| 256  | 256  | 256  | 0.328314                     | 14.645931                | 82.485022       | 
+------+------+------+------------------------------+--------------------------+-----------------+
| 512  | 512  | 512  | 3.213388                     | 18.199255                | 67.420400       | 
+------+------+------+------------------------------+--------------------------+-----------------+
| 1024 | 1024 | 1024 | 24.113855                    | 45.519852                | 71.875005       | 
+------+------+------+------------------------------+--------------------------+-----------------+
| 2048 | 2048 | 2048 | 186.688153                   | 264.195138               | 74.270743       | 
+------+------+------+------------------------------+--------------------------+-----------------+
| 4096 | 4096 | 4096 | 1469.773731                  | 1708.938204              | 75.469945       | 
+------+------+------+------------------------------+--------------------------+-----------------+
