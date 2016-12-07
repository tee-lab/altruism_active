#ifndef GLOBALS_H
#define GLOBALS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <string>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <iostream>
using namespace std;

#include "params.h"
#include "particles.h"
#include "utils/cuda_vector_math.cuh"

struct SimParams{	// using struct for C compatibility in kernel functions
	// time step
	float dt;

	// particle constants
	float Rr;
	float Rs;
	float speed;
	float turnRateMax;
	float cosphi;   // cos(P_turnRateMax*dt) !!
	float kA;
	float kO;
	float errSd;
		
	// boundaries
	float xmin; 
	float xmax;
	float ymin; 
	float ymax;

};


enum Strategy {Cooperate = 1, Defect = 0};

extern int cDevice;		// device on which to run
extern int genMax;		// max generations to simulate
extern int plotStep;	// # steps after which progress if displayed on terminal
extern string outDir, dataDir, framesDir;	// path of putput and data dir
extern int moveStepsPerGen;	// movement steps per generation
extern float arenaSize;		// arena size determines density
extern float RsNoiseSd; 	// mutation rate in Rs

// sim params (to be set manually before each run)
extern SimParams host_params;//, *dev_params;
extern float fitness_base; 		// base fitness
extern vector <float> c, cS;	// costs
extern vector <float> mu; 		// mutation rate
extern float Rg, Rs_base; 				// Radius for grouping

// ensemble params
extern int iEns;	// formerly iRunSet. ensemble number. each ensemble consists of nBlocks runs
extern int iExpt;	// formerly exptID.  each expt consists of 1 or more ensembles. 
extern string exptDesc, exptDescFull;	// exptDesc is unique for each ensemble, edfull is unique for each run

// GPU random number generator states and seeds
extern curandState * dev_XWstates;
extern int *seeds_h, *seeds_dev; 	// seeds will have size nFish*nBlocks (each thread will get unique seed) 

// Host random number generator
extern curandGenerator_t generator_host;
extern int seed_cpu;

// movement
extern int gRun;		// which run to display
//extern int imstep;		// current movement step
extern bool b_anim_on;	// movement on?

// device state arrays
extern float2 * pos_dev, * vel_dev;
extern float  * Rs_dev;
extern float  * kA_dev, * kO_dev;	//  kA and kO, the attraction and orientation constants

// host state arrays
extern vector <Particle> animals; 		// vector for all parents (this must include all blocks)
extern vector <Particle> offspring;	// vector for all offspring (1 block is sufficient)

extern map <int, int> g2ng_map, g2kg_map; 	// gid -> x maps, x is ng or kg
extern int genNum, stepNum;							// current generation number
extern int nCoop;							// current # atruists
extern float EpgNg, varPg, r, pbar, dp, Skg2bNg, SkgbNg, r2;	// 

// grid
extern int *gridCount, *cummCount;
extern int *gridCount_dev, *cummCount_dev, *filledCount_dev, *pStartIds_dev, *pEndIds_dev;;	// grid sized arrays
extern int *cellIds_dev, *sortedIds_dev;	// n-particles sized arrays
extern int nCellsX, nCells;				// number of cells in X, Y and total (X*Y)
extern int nCellsMaxX, nCellsMax;			// this depends on Rr, and memory of upto these many cells will be allocated.
extern float cellSize;


// output streams
//extern ofstream ng_fout[nBlocks], kg_fout[nBlocks], gid_fout[nBlocks], 
//	   wa_fout[nBlocks], rs_fout[nBlocks], p_fout[nBlocks], fit_fout[nBlocks], gix_fout[nBlocks],
//	   ko_fout[nBlocks], ka_fout[nBlocks];
//extern ofstream fout_fcs;


extern string homeDir_path, outDir_name, exptName;
extern int wks, gpu;					// gpu on which to run
extern ofstream * p_fout;

// population
extern int nFish;	// number of fish in each run
extern int nBlocks;	// number of runs

// graphics
extern int graphicsQual;			// 0 = no graphics, 1 = basic graphics, 2 = good graphics, 3 = fancy graphics, charts etc
extern int dispInterval;
extern bool b_displayEveryStep;

// experiment properties
extern float rDisp;				// Dispersal - Radius of dispersal. (-1 for random dispersal)
extern bool  b_baseline;		// Baseline  - is this a baseline experiment (Rs = 0)
extern bool  b_constRg;		// Grouping  - use constant radius of grouping?

// selection and mutation
extern float b;				// benefit value

// init
extern float fC0; 				// initial frequency of cooperators

// output
extern int  ngen_plot;			// number of points in generation axis to output in terminal/plots 
extern int  ngenScan;		 	// number of end generations to average over for parameter scan graphs

extern bool dataOut;			// output all values in files?
extern bool plotsOut;		// output plots
extern bool framesOut;		// output frames

extern vector <float> fb_sweep, as_sweep, nfish_sweep, rsb_sweep, rg_sweep, ens_sweep, mu_sweep, nm_sweep;



// overload random generator functions to use the generator_host declared above
inline float runif(float rmin=0, float rmax=1){ return runif(generator_host, rmin, rmax); }
inline float rnorm(float mu=0, float sd=1){		return rnorm(generator_host, mu, sd);}
inline float2 runif2(float xlim, float ylim){	return runif2(generator_host, xlim, ylim);} 
inline float2 rnorm2(float mu=0, float sd=1){	return rnorm2(generator_host, mu, sd); }
inline float2 runif2(float _norm){ 				return runif2(generator_host, _norm); }
inline float2 rnorm2_bounded(float mu, float sd, float rmin, float rmax){ 
	return rnorm2_bounded(generator_host, mu, sd, rmin, rmax);
}



#endif


