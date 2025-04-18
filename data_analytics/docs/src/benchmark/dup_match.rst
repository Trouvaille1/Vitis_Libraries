.. Copyright © 2019–2024 Advanced Micro Devices, Inc

.. `Terms and Conditions <https://www.amd.com/en/corporate/copyright>`_.

.. _l2_dup_match:

======================
Duplicate Record Match
======================

Duplicate Record Match resides in the ``L2/demos/text/dup_match`` directory and is to achieve the function of duplicate record matching, which includes modules such as Index, Predicate, Pair, Score, Cluster, etc.


Dataset
=======

- Input file: Randomly generate 10,000,000 lines (about 1 GB) of csv file similar to `L2/demos/text/dup_match/data/test.csv` as the test input file.
- The Demo execute time 8,215.56 s.
- Baseline (Dedupe Python: `https://github.com/dedupeio/dedupe`) execute time 35,030.751 s.
- Accelaration Ratio: 5.1X

.. note::
   | 1. The baseline version run on Intel® Xeon® CPU E5-2690 v4, clocked at 2.60 GHz.
   | 2. The training result of Baseline includes `self.predicate=((TfidfNGramCanopyPredicate: (0.8, Site name), TfidfTextCanopyPredicate: (0.8, Address)), (SimplePredicate: (alphaNumericPredicate, Site name), TfidfTextCanopyPredicate: (0.8, Site name)), (SimplePredicate: (wholeFieldPredicate, Site name), SimplePredicate: (wholeFieldPredicate, Zip)))`.


Executable Usage
===============

* **Work Directory (Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_data_analytics`. For getting the design:

.. code-block:: bash

   cd L2/demos/text/dup_match

* **Build the Kernel (Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. This process will take a long time, maybe couple of hours.

.. code-block:: bash

   make run TARGET=hw PLATFORM=xilinx_u50_gen3x16_xdma_201920_3 

* **Run the Kernel (Step 3)**

To get the benchmark results, run the following command:

.. code-block:: bash

   ./build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3/host.exe -xclbin ./build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3/TGP_Kernel.xclbin -in ./data/test.csv -golden ./data/golden.txt

Duplicate Record Match Input Arguments:

.. code-block:: bash

   Usage: host.exe -xclbin <xclbin_name> -in <input data>  -golden <golden data>
          -xclbin:     the kernel name
          -in    :     input data
          -golden:     golden data


* **Example Output (Step 4)** 

.. code-block:: bash


   ---------------------Duplicate Record Matching Flow-----------------
   DupMatch::run...
   TwoGramPredicate: column map size=14
   threshold=1000
   tf_value_ size is 238, index count=14, term count=122, skip=0
   config=15, 316
   config=15, 301
   Found Platform
   Platform Name: Xilinx
   Found Device=xilinx_u50_gen3x16_xdma_201920_3
   INFO: Importing build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3/TGP_Kernel.xclbin
   Loading: 'build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3/TGP_Kernel.xclbin'
   kernel has been created
   kernel start------
   threshold=1000
   index count=11, term count=65, skip=0
   threshold=1000
   index count=14, term count=36, skip=0
   CompoundPredicate: pair size=30
   CompoundPredicate: pair size=30
   CompoundPredicate: pair size=36
   duplicate sets 10
   DupMatch::run End
   Execution time 8.979s
   Pass validation.
   
   ------------------------------------------------------------

Profiling
=========

The duplicate record match design is validated on an AMD Alveo™ U50 board at a 270 MHz frequency. The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware Resources for Duplicate Record Match
    :align: center
 
    +-------------------+---------+-------+--------+--------+
    | Name              | LUT     | BRAM  | URAM   | DSP    |
    +-------------------+---------+-------+--------+--------+
    | Platform          | 135778  |  180  |   0    |    4   |
    +-------------------+---------+-------+--------+--------+
    | TGP_Kernel        | 272031  |   50  | 260    |  506   |
    +-------------------+---------+-------+--------+--------+
    |    TGP_Kernel_1   | 135974  |   25  | 130    |  253   |
    +-------------------+---------+-------+--------+--------+
    |    TGP_Kernel_2   | 136057  |   25  | 130    |  253   |
    +-------------------+---------+-------+--------+--------+
    | User Budget       | 734238  | 1164  | 640    | 5936   |
    +-------------------+---------+-------+--------+--------+
    |    Used Resources | 272031  |   50  | 260    |  506   |
    +-------------------+---------+-------+--------+--------+
    | Percentage        | 37.05%  | 4.30% | 40.63% | 8.52%  |
    +-------------------+---------+-------+--------+--------+


The performance is shown below.
  The input file is randomly generated 10,000,000 lines (about 1 GB) of csv file similar to `L2/demos/text/dup_match/data/test.csv` as the test input file, and its execute time is 8,215.56 s, so its throughput is 124.64 Mb/s.


.. toctree::
   :maxdepth: 1

