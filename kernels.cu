/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 /* This example demonstrates how to use the Cuda OpenGL bindings with the
  * runtime API.
  * Device code.
  */

#ifndef _KERNELS_CU_
#define _KERNELS_CU_

#include <curand_kernel.h>
#include "params.h"
#include "globals.h"
#include "utils/cuda_vector_math.cuh"
#include "utils/cuda_device.h"
#include <thrust/scan.h>

#include "utils/simple_io.h"

// simulation parameters in constant memory
__constant__ SimParams params;

cudaError __host__ copyParams(SimParams *s){
	return cudaMemcpyToSymbol(params, s, sizeof(SimParams));
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// KERNEL to set up RANDOM GENERATOR STATES
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__global__ void rngStateSetup_kernel(int * rng_Seeds, curandState * rngStates){
	//int tid = threadIdx.x;							// each block produces exactly the same random numbers
	int tid_u = threadIdx.x + blockIdx.x*blockDim.x;	// each block produces different random numbers
	
	curand_init (rng_Seeds[tid_u], 0, 0, &rngStates[tid_u]);
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// KERNELs to find NEAREST NEIGHBOURS using GRID
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void exclusive_scan(int * in, int * out, int n){
	out[0]=0;
	for (int i=1; i<n; ++i){
		out[i] = out[i-1]+in[i-1];
	}
}

// given the position, get the cell ID on a square grid of dimensions nxGrid x nxGrid,
// with each cell of size cellSize
// this function returns cellId considering 0 for 1st grid cell. With multiple blocks, user must add appropriate offset
//		|---|---|---|---|---|
//		|   |   |   |   |   |
//		|---|---|---|---|---|
//		|   |   | x |   |   |	<-- x = (pos.x, pos.y)
//		|---|---|---|---|---|
//		|   |   |   |   |   |
//		|---|---|---|---|---|
//      ^ 0 = (xmin, ymin)	^ nx = xmin + nx*cellWidth

inline __device__ int getCellId(float2 pos, int nxGrid, int cellwidth){//, SimParams *s){
	int ix = (pos.x-params.xmin)/(cellwidth+1e-6);	// add 1e-6 to make sure that particles on edge of last cell are included in that cell
	int iy = (pos.y-params.ymin)/(cellwidth+1e-6);
	return iy*nxGrid + ix;
}

// calculate cellID for each particle and count # particles / cell
// while storing, full cellID is stored 
// This kernel MUST BE launched with <<< nBlocks x nFish >>> config.
//  ^ this constraint is kept for intuitive reasons. To remove it, use pid/np in place of blockIdx.x
__global__ void gridCount_kernel(float2 * pos_array, int * cellId_array, int * gridCount_array, int nxGrid, int _cellSize, /*SimParams *s,*/ const int np, const int nb){
	unsigned int pid = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (pid < np*nb){
		int cellId_p = getCellId(pos_array[pid], nxGrid, _cellSize);//, s);	// get principle cell Id of particle pid
		int cellId = ix2(cellId_p, blockIdx.x, nxGrid*nxGrid);			// grid dimension is nCells x nBlocks
		atomicAdd(&gridCount_array[cellId],1);					// gridCount must be addressed using full cellId
	
		//++gridCount_array[pid];
		cellId_array[pid] = cellId;		// cellIds array stores principle cellId
	}
}

// rewrite particles in blockwise sorted order using results of scan
// full particle IDs are written in sorted order
// This kernel MUST BE launched with <<< nBlocks x nFish >>> config
__global__ void sortParticles_kernel(int* cummCount_array, int *filledCount_array, int * cellIds_array, int *sortedIds_array, int nxGrid, const int np, const int nb){

	unsigned int pid = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (pid < np*nb){
		int cell = cellIds_array[pid];
		//int cell = ix2(cell_p, blockIdx.x, nxGrid*nxGrid);
		int sortedAddress_p = cummCount_array[cell] + atomicAdd(&filledCount_array[cell],1); // atomicAdd returns old value
		int sortedAddress = ix2(sortedAddress_p, blockIdx.x, np);	// use of blockIdx.x here necessitates launching with  <<< nBlocks x nFish >>> config
		sortedIds_array[sortedAddress] = pid;
	}
}

// this kernel gets the start and end ids of particles in each grid cell. 
// Hence it must be run using 1 thread for each grid cell = nBlocks*nCells threads
// startId and endId arrays store full particle IDs. 
// This kernel MUST BE launched with <<< nBlocks x nFish >>> config
__global__ void getParticleIds_kernel(int * pStartIds_array, int* pEndIds_array, int * cellIds_array, int* cummCount_array, int* gridCount_array, const int np, const int nb){
	unsigned int pid = blockIdx.x*blockDim.x + threadIdx.x;

	if (pid < np*nb){
		int gid = cellIds_array[pid];		// full pid automatically gives full gid
		int count = gridCount_array[gid];	// number of particles in cell gid
		int scan = cummCount_array[gid];	// cummulative number of particles till cell gid in block blockIdx.x
		
		int startId = scan*float(count!=0) + -1*float(count==0);	// if count is zero, set startId to -1
		int endId = startId + count-1;	// if count is zero, this will become -2. This is FINE.

		pStartIds_array[gid] = ix2(startId, blockIdx.x, np);
		pEndIds_array[gid]   = ix2(  endId, blockIdx.x, np);
	}
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// KERNEL TO EXECUTE MOVEMENT OF A SINGLE FISH
// This is a CRACKING KERNEL because ALL conditionals 
// are replaced with indicator variables !!! :) :)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__global__ void movement_kernel(float2* pos_array, float2* vel_array, float* Rs_array, 
								int* cellIds_array, int* sortedIds_array, 
								int* pStartIds_array, int* pEndIds_array, int nxGrid, int nGrid,
								unsigned int nfish, //SimParams * dev_params,
								curandState * RNGstates, float* kA_array, float* kO_array){
	
	unsigned int myID = blockIdx.x*blockDim.x + threadIdx.x;	// full particle ID. Principle particle ID is used only for accessing shared memory

	extern __shared__ float sharedMem[];
	float2 * pos_all = (float2*) sharedMem;
	float2 * vel_all = (float2*) &pos_all[nfish];
	float  * Rs_all  = (float*)  &vel_all[nfish];
	
	// copy self data from global memory to shared memory and wait for all threads to finish
	pos_all[threadIdx.x] = pos_array[myID];	// pos array has positions at alternate locations
	vel_all[threadIdx.x] = vel_array[myID];
	 Rs_all[threadIdx.x] =  Rs_array[myID]; 
	
	// init directions of attraction and repulsion. Only one of these will be considered
	float2 dirR = make_float2(0,0);	// initialize direction of repulsion with 0 norm
	float2 dirA = make_float2(0,0);	// initialize direction of attraction with 0 norm
	float2 dirO = make_float2(0,0);	// initialize direction of orientation with 0 norm

	__syncthreads();


	// local copies of my pos and vel. These will be modified.
//	float2 myPos = pos_all[threadIdx.x];
//	float2 myVel = vel_all[threadIdx.x];
	#define myPos (pos_all[threadIdx.x])
	#define myVel (vel_all[threadIdx.x])


	int myCell = cellIds_array[myID] % nGrid;	// get priciple cellId of focal particle. Note: myID is full particle address
	int myCellx = myCell % nxGrid;		// convert grid cell to x and y indices
	int myCelly = myCell / nxGrid;

	// loop over NINE neighbouring cells (including own cell) to find particles
	for (int innx=-1; innx<2; ++innx){			//  offsets to add in x and y indices to get neighbour cells
		for (int inny=-1; inny<2; ++inny){	

			// periodify the neighboring cell and get its address. The following code is equivalent to
			// 	 otherCellx = myCellx + innx; if (otherCellx < 0) otherCellx += nxGrid; if (otherCellx >= nxGrid) otherCellx -= nxGrid;
			int otherCellx = myCellx + innx;
			otherCellx = otherCellx + int(otherCellx < 0)*nxGrid - int(otherCellx >= nxGrid)*nxGrid;
			int otherCelly = myCelly + inny;
			otherCelly = otherCelly + int(otherCelly < 0)*nxGrid - int(otherCelly >= nxGrid)*nxGrid;

			int otherCell = ix2(otherCellx, otherCelly, nxGrid); 		// calculate principle cellId from (x,y) 
			int otherCellFull = ix2(otherCell, blockIdx.x, nGrid);		// get full cellId 

			// loop across particles found in the cell
			int start = pStartIds_array[otherCellFull];
			int end   = pEndIds_array[otherCellFull];
			for (int si=start; si <=end; ++si){ // si is particle index in sorted array
				int i = sortedIds_array[si] % nfish;	// get principle particle index from sorted particles array - Note that sorted array gives full particle ID
				if (i == threadIdx.x) continue;			// Exclude self
			
				// get direction and distance to other 
				float2 v2other = periodicDisplacement(	myPos, pos_all[i], 
														params.xmax-params.xmin, 
														params.ymax-params.ymin  );
				float d2other = length(v2other);
		
				// indicator variables 
				float Irr = float(d2other < params.Rr); //? 1:0;
				float Ira = float(d2other < Rs_all[threadIdx.x]); //? 1:0;
		
				// keep adding to dirR and dirA so that average direction or R/A will be taken
				v2other = normalize(v2other); // normalise to consider direction only

				dirR = dirR - v2other*Irr;				// add repulsion only if other fish lies in inside Rr
				dirA = dirA + v2other*Ira*(1-Irr); 		// add attraction only if other fish lies in (Rr < r < Ra)
				dirO = dirO + vel_all[i]*Ira*(1-Irr);	// add alignment only if other fish lies in (Rr < r < Ra)
			}
		}
	}

	// calculate direction of orientation (either unit or zero)
	dirO = normalizeSafeZero(dirO);
	dirA = normalizeSafeZero(dirA);

	
	// calculate direction of social interaction
	float Ir = float(length(dirR) > 1e-6);	// fish in Rr
	float Ia = float(length(dirA) > 1e-6);	// fish in Ra and hence also in Rr

	float2 dirS = myVel*(1-params.kA-params.kO) + (dirA*params.kA + dirO*params.kO);	
	//dirS = normalize(dirS);


	// final direction is either of the three terms below:
	// 1 - previous direction if no one is in either Ra or Rr (Ia = Ir = 0)
	// 2 - only repulsion direction if someone is in Rr (Ir = 1, Ia = 1)
	// 3 - social direction if someone is in Ra but no one in Rr (Ir = 0, Ia = 1)
	float2 finalDir = myVel*(1-Ir)*(1-Ia) + dirR*Ir + dirS*Ia*(1-Ir);	// dir is guaranteed to be non-zero
	finalDir = normalize(finalDir);
	
	// introduce error in following direction
	finalDir += curand_normal2(&RNGstates[myID])*params.errSd; 
	finalDir = normalize(finalDir);


	// impose a turning rate constraint
	float sinT = myVel.x*finalDir.y - myVel.y*finalDir.x;		// sinT = myVel x finalDir
	float cosT = dot(finalDir, myVel);	// Desired turning angle. Both vectors are unit so dot product is cos(theta) 
	float cosL = clamp( max(cosT, params.cosphi), -1.f, 1.f);
	float sinL = sqrtf(1-cosL*cosL);
	sinL = sinL - 2*sinL*float(sinT < 0);	// equivalent to: if (sinT < 0) sinL = -sinL;
	float2 a = make_float2(myVel.x*cosL - myVel.y*sinL, myVel.x*sinL + myVel.y*cosL);

	// no writing to shared memory has happened till here.
	// wait for all threads to finish their work in (reading from) shared memory. Then update shared memory
	__syncthreads();

	myVel = normalize(a);		// final velocity 
	myPos = myPos + myVel * (params.speed * params.dt);	
	makePeriodic(myPos.x, params.xmin, params.xmax);
	makePeriodic(myPos.y, params.ymin, params.ymax);
	
	// wait for all threads to update shared memory, then update global memory
	__syncthreads();


	// update pos and vel in global memory
	pos_array[myID] = myPos; 
	vel_array[myID] = myVel;
//	RNGstates[myID] = localState;
	
}

void print_devArray(int * vdev, int n){
	int * v = new int[n];
	cudaMemcpy(v, vdev, n*sizeof(int), cudaMemcpyDeviceToHost);
	printArray(v,n);
	delete [] v;
}

// Launcher for movement kernel - Uses many global variables
void launch_movement_kernel(){


	// reset counting arrays to Zero
	thrust::fill( (thrust::device_ptr <int>)gridCount_dev,   (thrust::device_ptr <int>)gridCount_dev   + nCells*nBlocks, (int)0);
	thrust::fill( (thrust::device_ptr <int>)filledCount_dev, (thrust::device_ptr <int>)filledCount_dev + nCells*nBlocks, (int)0);
	
//	cout << "GRIDCOUNT\n";
//	print_devArray(gridCount_dev, nCells*nBlocks);

	// count particles / grid cell
	gridCount_kernel <<<nBlocks, nFish >>> (pos_dev, cellIds_dev, gridCount_dev, nCellsX, cellSize, /*dev_params,*/ nFish, nBlocks);
	getLastCudaError("Grid Count");

//	cout << "GRIDCOUNT\n";
//	print_devArray(gridCount_dev, nCells);

	// scan - calc cummulative particles/cell
	cudaMemcpy(gridCount, gridCount_dev, nBlocks*nCells*sizeof(int), cudaMemcpyDeviceToHost);
	for (int iblock=0; iblock<nBlocks; ++iblock){
//		thrust::device_ptr <int> v_in  = (thrust::device_ptr <int>) &gridCount_dev[ix2(0,iblock,nCells)];
//		thrust::device_ptr <int> v_out = (thrust::device_ptr <int>) &cummCount_dev[ix2(0,iblock,nCells)];
//		thrust::exclusive_scan(v_in, v_in+nCells, v_out);
		exclusive_scan(&gridCount[ix2(0,iblock,nCells)], &cummCount[ix2(0,iblock,nCells)], nCells);
	}
	cudaMemcpy(cummCount_dev, cummCount, nBlocks*nCells*sizeof(int), cudaMemcpyHostToDevice);
	

//	cout << "CUMMCOUNT\n";
//	print_devArray(cummCount_dev, nCells*nBlocks);

	// sort particles by grid ID - using scan results
	sortParticles_kernel <<<nBlocks, nFish >>> (cummCount_dev, filledCount_dev, cellIds_dev, sortedIds_dev, nCellsX, nFish, nBlocks);

//	cout << "SORTEDIDS\n";
//	print_devArray(sortedIds_dev, nFish*nBlocks);

	// set start and end Ids of grids with no particles to -1 and -2 respectively
	thrust::fill((thrust::device_ptr <int>)pStartIds_dev, (thrust::device_ptr <int>)pStartIds_dev+nCells*nBlocks, (int)-1);
	thrust::fill((thrust::device_ptr <int>)pEndIds_dev,   (thrust::device_ptr <int>)pEndIds_dev  +nCells*nBlocks, (int)-2);

	// get the particle IDs for each cell
	getParticleIds_kernel <<<nBlocks, nFish >>>(pStartIds_dev, pEndIds_dev, cellIds_dev, cummCount_dev, gridCount_dev, nFish, nBlocks);

//	print_devArray(cellIds_dev, nFish);
//	print_devArray(pStartIds_dev, nCells);
//	print_devArray(pEndIds_dev, nCells);

    // execute the movement kernel
	int sharedMemSize = nFish * (sizeof(float2) + sizeof(float2) + sizeof(float));
    movement_kernel <<< nBlocks, nFish, sharedMemSize >>>(	pos_dev, vel_dev, Rs_dev, 
    														cellIds_dev, sortedIds_dev,
    														pStartIds_dev, pEndIds_dev, nCellsX, nCells,
    														nFish, //dev_params,
    														dev_XWstates, NULL, NULL);
}

// Wrapper for state setup kernel 
void launch_rngStateSetup_kernel(int * rng_blockSeeds, curandState * rngStates){
	rngStateSetup_kernel <<< nBlocks, nFish >>> (rng_blockSeeds, rngStates);
}

#endif // #ifndef _KERNEL_H_


