.. 
   Copyright 2019 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. _introduction_L1:

Introducton for Ultrasound Library Level 1  
==========================================

.. toctree::
   :hidden:
   :maxdepth: 1

Ultrasound Library - Level 1 (L1)
---------------------------------

As explained in the introduction, the lowest level of the libraries is a set of APIs which are created to a closest mapping possible between *NumPy* and the *C++* SIMD APIs of the AI Engine. The APIs results a more comprehensive and easy interface for the user to create applications using numpy-like interfaces. 
The APIs can be divided in two main groups:

1. Element-wise operations between vectors and matrices. Those operations varies between basic operations (i.e. sum, multiplication etc...) to more complex ones (i.e. reciprocal, sign etc...);
2. Vector managing and creation. Those operations are intended to be used by the user to create or modify the dimension of the vectorial operands. Two examples of those operations are *Tile* and *Ones*.

The APIs are written in a templatic way, such that the user might parametrize the length of the operands or of the SIMD operations to suites the necessities of the customers.
Last remark, before starting with the single kernels explanation, is the nomenclature used in the kernels. The kernels are thought to be used as the user would do in NumPy, but on contrary respect to what is possible in python, in C++ the dimensionality of the operands must be known a priori to write approriate and correct operations. For common DSP and BLAS application, such as Ultrasound Imaging, it is very important to control the dimensionality of the data, in order to build appropriate applications. For this reason the kernels have been written in the following way:

The first part of the name indicates the type of operation chosen. The second and the third parts of the names of the interface are optional, depending on the number of operands of the operation. Provided that they are two, the letter S/V/M indicates their dimensionality. S stands for **Scalar**, V for **Vector** and M for **Matrix**.
To give an example, `mulMV` is the operation which multiplies (element-wise) the rows of a Matrix (as first operand, M) with a Vector of the same dimension (as second operand, V). Another example is `diffSV` which is the operation which substract a scalar value (as first operand, S) to every element of a Vector (as second operand, V).

We now detail every L1 kernel available in the library:

Kernel name: `absV`
####################

.. code-block:: cpp

	template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
	void absV(input_stream<T>* in1, output_stream<T>* out);


Element-Wise absolute values of a vector.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the vector to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.
	
Kernel name: `cosV`
###################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void cosV(input_stream<float>* in1, output_stream<float>* out);


Element-Wise cosine values of a vector. Elements must be expressed in radians with the range [0...2kPI] 

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the vector to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.

Kernel name: `diffMV`
######################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void diffMV(input_stream<T>* in1, input_stream<T>* in2, output_stream<T>* out);


Element-Wise difference between the of the row of a matrix and the values of a vector. The number of the column of the matrix and the entry of the vector must have the same size.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the first matrix to be passed to the kernel.
	- `in2`: elements of the array to be passed to the kernel.
	- `out`: elements of the result of the operation (matrix) to be passed from the kernel.

Kernel name: `diffSV`
######################

.. code-block:: cpp

    template<typename T, const unsigned int LEN, const unsigned int INCREMENT, const unsigned VECDIM>
    void diffSV(input_stream<T>* in1, input_stream<T>* in2, output_window<T>* out);


Element-Wise difference between a scalar and the values of a vector. For every iteration (expressed by `LEN`) we need to pass 4 times the scalar value to the stream of the scalar value.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: scalar element to be passed to the kernel.
	- `in2`: elements of the array to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.

Kernel name: `diffVS`
######################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void diffVS(input_stream<T>* in1, input_stream<T>* in2, output_stream<T>* out);


Element-Wise difference between the values of a vector and a scalar. For every iteration (expressed by `LEN`) we need to pass 4 times the scalar value to the stream of the scalar value.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the array to be passed to the kernel.
	- `in2`: scalar element to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.
	
Kernel name: `divVS`
#####################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void divVS(input_stream<T>* in1, output_stream<T>* out);


Element-Wise division between the values of a vector and a scalar. For every iteration (expressed by `LEN`) we need to pass 4 times the scalar value to the stream of the scalar value.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the array to be passed to the kernel.
	- `in2`: scalar element to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.
	
Kernel name: `equalS`
######################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM, const unsigned SCALAR>
    void equalS(input_stream<T>* in1, output_stream<T>* out);


Check whether the element of an array are equal to a specific number. An array of 0s or 1s is returned. 1 means that the element at that specific position is equal to the scalar, otherwise 0 is returned.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
	- `SCALAR`: scalar to which the elements of the array are compared to;
- **Function params**:
	- `in1`: elements of the vector to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.
	
Kernel name: `lessOrEqualThanS`
################################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM, const unsigned SCALAR>
    void lessOrEqualThanS(input_stream<T>* in1, output_stream<T>* out);


Check whether the element of an array are less or equal to a specific number. An array of 0s or 1s is returned. 1 means that the element at that specific position is less or equal to the scalar, otherwise 0 is returned.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
	- `SCALAR`: scalar to which the elements of the array are compared to;
- **Function params**:
	- `in1`: elements of the vector to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.

Kernel name: `mulMM`
#####################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void mulMM(input_stream<T>* in1, input_stream<T>* in2, output_stream<T>* out);


Element-Wise multiplication of two matrixes. The first matrix and the second one must have the same size.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the first matrix to be passed to the kernel.
	- `in2`: elements of the second matrix to be passed to the kernel.
	- `out`: elements of the result of the operation (matrix) to be passed from the kernel.

Kernel name: `mulVS`
#####################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void mulVS(input_stream<T>* in1, output_stream<T>* out);


Element-Wise multiplication between the values of a vector and a scalar. For every iteration (expressed by `LEN`) we need to pass 4 times the scalar value to the stream of the scalar value.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the array to be passed to the kernel.
	- `in2`: scalar element to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.

Kernel name: `mulVV`
#####################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void mulVV(input_stream<T>* in1, input_stream<T>* in2, output_stream<T>* out);


Element-Wise multiplication of two vectors. The first vector and the second one must have the same size.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the first vector to be passed to the kernel.
	- `in2`: elements of the second vector to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.

Kernel name: `norm_axis_1`
###########################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void norm_axis_1(input_stream<T>* in1, output_stream<T>* out);


Perform row wise the euclidean norm of a matrix of the columns. Because for every row returns a number, the result is a vector of values which represents for every row the magnitude of the euclidean norm.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the vector to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.

Kernel name: `ones`
####################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void ones_stream(output_stream<T>* out);


Return a vector of with all entry set to 1. 

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.

Kernel name: `outer`
#####################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void outer(input_window<T>* in1, input_window<T>* in2, output_stream<T>* out);


Perform the outer product (also named cross-product or vector-product) between two vectors. The result of this operation is a matrix which rows are the number of the entry of the first vector and the column the number of the entry of the second one.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the first vector to be passed to the kernel.
	- `in2`: elements of the second vector to be passed to the kernel.
	- `out`: elements of the result of the operation (matrix) to be passed from the kernel.

Kernel name: `reciprocalV`
##########################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void reciprocalV(input_stream<T>* in1, output_stream<T>* out);


Element-wise inverse operation of the entry of the vector.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the vector to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.

Kernel name: `sqrtV`
#####################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void sqrtV(input_stream<T>* in1, output_stream<T>* out);


Element-wise square root operation of the entry of the vector.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the vector to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.

Kernel name: `squareV`
#######################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void squareV(input_stream<T>* in1, output_stream<T>* out);


Element-wise square operation of the entry of the vector.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the vector to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.

Kernel name: `sum_axis_1`
##########################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void sum_axis_1(input_stream<T>* in1, output_stream<T>* out);


Perform row wise the reduce add of a matrix of the columns. Because for every row returns a number, the result is a vector of values which represents for every row the magnitude of the reduce add operation.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the vector to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.

Kernel name: `sumMM`
#####################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void sumMM(input_window<T>* in1, input_window<T>* in2, output_stream<T>* out);


Element-Wise sum of two matrixes. The first matrix and the second one must have the same size.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the first matrix to be passed to the kernel.
	- `in2`: elements of the second matrix to be passed to the kernel.
	- `out`: elements of the result of the operation to be passed from the kernel.

Kernel name: `sumVS`
#####################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void sumVS(input_stream<T>* in1, input_stream<T>* in2, output_stream<T>* out);


Element-Wise addition between the values of a vector and a scalar. For every iteration (expressed by `LEN`) we need to pass 4 times the scalar value to the stream of the scalar value.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the array to be passed to the kernel.
	- `in2`: scalar element to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.

Kernel name: `sumVV`
#####################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void sumVV(input_stream<T>* in1, input_stream<T>* in2, output_stream<T>* out);


Element-Wise addition of two vectors. The first vector and the second one must have the same size.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the first vector to be passed to the kernel.
	- `in2`: elements of the second vector to be passed to the kernel.
	- `out`: elements of the result of the operation (vector) to be passed from the kernel.

Kernel name: `tileV`
#####################

.. code-block:: cpp

    template<typename T, const unsigned LEN, const unsigned INCREMENT, const unsigned VECDIM>
    void tileV(input_stream<T>* in1, output_stream<T>* out);


This kernel read in input a vector and returns it `LEN` times. This operation creates a matrix with the rows all equal to the others.

- **Template params**:
	- `T`: type of the operation;
	- `LEN`: number of elements to be processed in the kernel per iteration;
	- `INCREMENT`: parameter which indicates how much iterations have been performed by the SIMD with respect to the intended total length;
	- `VECDIM`: dimension of the SIMD to be performed. Addressed in the Xilinx UG1076, it depends on the type chosen;
- **Function params**:
	- `in1`: elements of the vector to be passed to the kernel.
	- `out`: elements of the result of the operation (matrix) to be passed from the kernel.