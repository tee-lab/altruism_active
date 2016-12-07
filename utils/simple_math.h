#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

#include <fstream>
#include <iostream>
#include <map>
#include <cmath>
#include <cstdlib>
#include <algorithm>
using namespace std;


const float pi = 3.14159265;

#ifdef __CUDACC__
#define DEVICE_NAME __device__ __host__
#else
#define DEVICE_NAME 
#endif

// =============================================================================
// 		Simple indexing utils
// 		ADDED by : JAIDEEP
// 		6 May 2013
// =============================================================================

inline DEVICE_NAME int ix2(int ix, int iy, int nx){
	return iy*nx + ix;
}

inline DEVICE_NAME int ix3(int ix, int iy, int iz, int nx, int ny){
	return iz*nx*ny + iy*nx + ix;
}

// =============================================================================
// 		Simple array operations
// 		ADDED by : JAIDEEP
// 		4 Mar 2013
// =============================================================================

template <class T>
T arraySum(T *v, int n){
	T sum=0;
	for (int i=0; i<n; ++i) sum += v[i];
	return sum;
}

template <class T>
T arrayMin(T *v, int n){
	T amin=v[0];
	for (int i=1; i<n; ++i) amin = min(amin, v[i]);
	return amin;
}

template <class T>
T arrayMax(T *v, int n){
	T amax=v[0];
	for (int i=1; i<n; ++i) amax = max(amax, v[i]);
	return amax;
}

// =============================================================================
// 		Random numbers from the default C++ generator
// 		ADDED by : JAIDEEP
// 		6 May 2013
// =============================================================================

inline float runif_cpp(float min = 0, float max=1){
	float r = float(rand())/RAND_MAX; 
	return min + (max-min)*r;
}

inline float rnorm_cpp(float mu = 0, float sd = 1){
	float u = runif_cpp(), v = runif_cpp();		// uniform rn's [0,1] for box-muller
	float x = sqrt(-2.0*log(u)) * cos(2*pi*v);
	return mu + sd*x;
}


// =============================================================================
// 		Map operations
// 		ADDED by : JAIDEEP
// 		21 Dec 2013
// =============================================================================

template <class T>
T mapSum(map <int, T> &m){
	T sum = 0;
	for (typename map<int,T>::iterator it= m.begin(); it != m.end(); ++it){
		sum += it->second;
	}
	return sum;
}

template <class T>
float mapAvg(map <int, T> &m){
	return mapSum(m)/float(m.size());
}



#endif

