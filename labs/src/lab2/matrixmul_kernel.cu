/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	const size_t TILE_SIZE = 4*4;
	__shared__ float mTile[TILE_SIZE];
	__shared__ float nTile[TILE_SIZE];
	__shared__ float pTile[TILE_SIZE];

	for(int i = 0;i < M.width/blockDim.x; i++){//Loop through all tiles in row/column
		mTile[threadIdx.x + (threadIdx.y * blockDim.x)] = M.elements[(blockDim.x * i + threadIdx.x) +(threadIdx.y * M.width) + (blockDim.y * blockIdx.y * M.width)];
		nTile[threadIdx.x + (threadIdx.y * blockDim.x)] = N.elements[(blockDim.x * blockIdx.x) + (i * blockDim.y * N.width) + threadIdx.x + (N.width * threadIdx.y)];
		pTile[threadIdx.x + (threadIdx.y * blockDim.x)] = 0;
		
		/*
		mTile[threadIdx.x + (threadIdx.y * blockDim.x)] = 1;
		nTile[threadIdx.x + (threadIdx.y * blockDim.x)] = 1;
		pTile[threadIdx.x + (threadIdx.y * blockDim.x)] = 1;
		*/
		
		__syncthreads();
		for(int j = 0; j < blockDim.x; j++){//M*N
			pTile[threadIdx.x + (threadIdx.y * blockDim.y)] += mTile[(threadIdx.y * blockDim.x) + j] * nTile[threadIdx.x + j * blockDim.x];
		}
	}
	
	__syncthreads();
	P.elements[blockDim.y * P.width * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x + threadIdx.y * P.width] = pTile[threadIdx.x + (threadIdx.y * blockDim.x)];
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
