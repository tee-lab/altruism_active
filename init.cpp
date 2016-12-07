#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
using namespace std;

#include "utils/cuda_vector_math.cuh"
#include "utils/cuda_device.h"
#include "utils/simple_io.h"

#include "init.h"
#include "params.h"
#include "globals.h"
#include "particles.h"
//#include "altruism.h"

#define PP_SEED time(NULL)

void launch_rngStateSetup_kernel(int * rng_blockSeeds, curandState * rngStates);

extern cudaError __host__ copyParams(SimParams *s);

int initStateArrays(){

	// UPDATE SIMPARAMS BEFORE CALLING THIS FUNCTION
//	cudaMemcpy( dev_params, &host_params, sizeof(SimParams), cudaMemcpyHostToDevice);
//	cudaMemcpyToSymbol(&params, &host_params, sizeof(SimParams));
	copyParams(&host_params);

	// ~~~~~~~~~~~~~~~~~ EXPT DESC ~~~~~~~~~~~~~~~~~~~~	 
	stringstream sout;
	sout << setprecision(3);
	sout << exptName;
	if (b_baseline) sout << "_base";
	sout << "_n("   << nFish
		 << ")_nm(" << moveStepsPerGen
		 << ")_rd(" << rDisp
	 	 << ")_mu(" << mu[0]
		 << ")_fb(" << fitness_base
		 << ")_as(" << arenaSize
	 	 << ")_rg(";
	if (b_constRg) sout << Rg;
	else sout << "-1";
	if (b_baseline)  sout << ")_Rs(" << Rs_base;
	sout << ")";
	exptDesc = sout.str(); sout.clear();


	// ~~~~~~~~~~~~~~ CPU random generator (MT) ~~~~~~~~~~~~~~~~~~~~~~~~~~~
	curandCreateGeneratorHost(&generator_host, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(generator_host, PP_SEED);	// seed by time in every expt

	// ~~~~~~~~~~~~~~ GPU random generator (XORWOW) ~~~~~~~~~~~~~~~~~~~~~~~
	srand(PP_SEED);
	for (int i=0; i<nFish*nBlocks; ++i) seeds_h[i] = rand(); 
	cudaMemcpy( seeds_dev, seeds_h, sizeof(int)*nFish*nBlocks, cudaMemcpyHostToDevice);
	launch_rngStateSetup_kernel(seeds_dev, dev_XWstates);
	getLastCudaError("RNG_kernel_launch");


	// ~~~~~~~~~~~~~~ initial state ~~~~~~~~~~~~~~~~~~~
	// each block gets different Initial state
	for (int i=0; i<nFish; ++i){
		for (int iblock=0; iblock<nBlocks; ++iblock){

			int ad = ix2(i, iblock, nFish);
			
			animals[ad].pos = runif2(host_params.xmax, host_params.ymax);
			animals[ad].vel = runif2(1.0);
			animals[ad].wA  = (i< fC0*nFish)? Cooperate:Defect; 
			animals[ad].Rs  = (b_baseline)? Rs_base:host_params.Rs;		// init with Rs_base for baseline, with specified value otherwise
			animals[ad].kA  = host_params.kA; //	runif(); // 
			animals[ad].kO  = host_params.kO; //	runif(); // 
			animals[ad].ancID = i;	// each fish within a block gets unique ancestor ID
			animals[ad].fitness = 0;

//			// set differential Rs for A and D to check self-sorting
//			if (animals[ad].wA == Cooperate) animals[ad].Rs = 1.3;
//			else animals[ad].Rs = 1.1;

		}
	}

	updateGroups(gRun);
	updateGroupSizes(gRun);

	printParticles(&animals[0], 5);

	// copy arrays to device
	//             v dst           v dst pitch     v src                     v src pitch       v bytes/elem    v n_elem       v direction
	cudaMemcpy2D( (void*) pos_dev, sizeof(float2), (void*)&(animals[0].pos), sizeof(Particle), sizeof(float2), nFish*nBlocks, cudaMemcpyHostToDevice);
	cudaMemcpy2D( (void*) vel_dev, sizeof(float2), (void*)&(animals[0].vel), sizeof(Particle), sizeof(float2), nFish*nBlocks, cudaMemcpyHostToDevice);
	cudaMemcpy2D( (void*) Rs_dev,  sizeof(float),  (void*)&(animals[0].Rs),  sizeof(Particle), sizeof(float),  nFish*nBlocks, cudaMemcpyHostToDevice);

	// grid
	cellSize = 10;
	nCellsX = int(arenaSize/(cellSize+1e-6))+1;
	nCells = nCellsX * nCellsX;	


	//init files for output
	if (dataOut){
		for (int iblock=0; iblock<nBlocks; ++iblock){
			// get expt desc including block details
			stringstream sout;
			sout << setprecision(3) << exptDesc 
			 	 << "_c(" << c[iblock];
			if (!b_baseline) sout << ")_cS(" << cS[iblock];
			sout << ")_ens(" << iEns << ")";
			string edFull = sout.str();

			p_fout[iblock].close();
			p_fout[iblock].open(string(dataDir + "/p_"  +edFull).c_str());
		}
	}
		
}




int allocArrays(){

	p_fout = new ofstream[nBlocks];

	// ~~~~~~~~~~~~~~~ sim params ~~~~~~~~~~~~~~~~~~~~
//	cudaMalloc( (void**) &dev_params, sizeof(SimParams));					// sim params

	// ~~~~~~~~~~~~~~~~ RNG ~~~~~~~~~~~~~~~~
	seeds_h = new int [nFish*nBlocks];
	cudaMalloc( (void**) &seeds_dev,    nFish*nBlocks*sizeof(int));			// seeds
	cudaMalloc( (void**) &dev_XWstates, nFish*nBlocks*sizeof(curandState));	// rng states

	// ~~~~~~~~~~~~~ state variables on GPU ~~~~~~~~~~~~~~~~~~~~~
	cudaMalloc( (void**) &pos_dev, nFish*nBlocks*sizeof(float2));			// state variables
	cudaMalloc( (void**) &vel_dev, nFish*nBlocks*sizeof(float2));
	cudaMalloc( (void**) &Rs_dev,  nFish*nBlocks*sizeof(float ));

	// ~~~~~~~~~~~~~~~~~~~~~~~ Grid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	gridCount = new int[nBlocks*nCellsMax];
	cummCount = new int[nBlocks*nCellsMax];

	// alloc memory for the maximum possible size of grid
	cudaMalloc( (void**) &gridCount_dev,   nCellsMax*nBlocks*sizeof(int));
	cudaMalloc( (void**) &cummCount_dev,   nCellsMax*nBlocks*sizeof(int));
	cudaMalloc( (void**) &filledCount_dev, nCellsMax*nBlocks*sizeof(int));
	cudaMalloc( (void**) &pStartIds_dev,   nCellsMax*nBlocks*sizeof(int));
	cudaMalloc( (void**) &pEndIds_dev,     nCellsMax*nBlocks*sizeof(int));

	// alloc memory for particle properties arrays 
	cudaMalloc( (void**) &cellIds_dev,     nFish*nBlocks*sizeof(int));
	cudaMalloc( (void**) &sortedIds_dev,   nFish*nBlocks*sizeof(int));

}

void freeArrays(){
	// ~~~~~~~~~~~~~~~ sim params ~~~~~~~~~~~~~~~~~~~~
//	cudaFree( dev_params);					// sim params

	// ~~~~~~~~~~~~~~~~ RNG ~~~~~~~~~~~~~~~~
	delete [] seeds_h;
	cudaFree( seeds_dev);					// seeds
	cudaFree( dev_XWstates);	// rng states

	// ~~~~~~~~~~~~~ state variables ~~~~~~~~~~~~~~~~~~~~~
	cudaFree( pos_dev);			// state variables
	cudaFree( vel_dev);
	cudaFree( Rs_dev);

	// ~~~~~~~~~~~~~~~~~~ Grid ~~~~~~~~~~~~~~~~~~~~~~~~~
	delete [] gridCount;
	delete [] cummCount;

	cudaFree( gridCount_dev);
	cudaFree( filledCount_dev);
	cudaFree( cummCount_dev);
	cudaFree( cellIds_dev);
	cudaFree( sortedIds_dev);
	cudaFree( pStartIds_dev);
	cudaFree( pEndIds_dev);
	
}



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	--> read_ip_params_file()

	READ INPUT PARAMS FILE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
int read_execution_config_file(string filename){
	ifstream fin;
	fin.open(filename.c_str());
	
	string s, u, v, w, y;
	int n, m, l;
	float f;
	
	string attrbegin = ">";
	while (fin >> s && s != attrbegin);	// read until 1st > is reached
	
	fin >> s; 
	if (s != "DIR") {cout << "directories not found!"; return 1;}
	while (fin >> s && s != attrbegin){
		if (s == "") continue;	// skip empty lines
		if (s == "#") {getline(fin,s,'\n'); continue;}	// skip #followed lines (comments)
		fin >> u;
			 if (s == "homeDir_path") 	outDir = u;
		else if (s == "outDir_name") 	outDir = outDir + "/" + u;
		else if (s == "exptName")		exptName = u;
	}

	fin >> s; 
	if (s != "GPU") {cout << "desired GPU number not found!"; return 1;}
	while (fin >> s && s != attrbegin){
		if (s == "") continue;
		if (s == "#") {getline(fin,s,'\n'); continue;}	// skip #followed lines (comments)
		fin >> n;
		
		if (s == "wks") 		wks = n;
		else if (s == "gpu") 	gpu = n;
	}

	
	fin >> s; 
	if (s != "GPU_CONFIG") {cout << "gpu config data not found!"; return 1;}
	while (fin >> s && s != attrbegin){
		if (s == "") continue;	// skip empty lines
		if (s == "#") {getline(fin,s,'\n'); continue;}	// skip # following stuff (comments)
		fin >> n;		
				
		if (s == "particles") 	nFish = n;
		else if (s == "runs") 	nBlocks = n;

	}	
		
	fin >> s; 
	if (s != "GRAPHICS") {cout << "static input files not found!"; return 1;}
	while (fin >> s && s != attrbegin){
		if (s == "") continue;	// skip empty lines
		if (s == "#") {getline(fin,s,'\n'); continue;}	// skip #followed lines (comments)
		fin >> n;
		
		if (s == "graphicsQual") 		graphicsQual = n;
		else if (s == "dispInterval") 	dispInterval = n;	
		else if (s == "gRun") 			gRun = n;
		else if (s == "b_anim_on") 		b_anim_on = n;
	}
	
	
	fin >> s; 
	if (s != "EXPT") {cout << "sim time not found!"; return 1;}
	while (fin >> s && s != attrbegin){
		if (s == "") continue;	// skip empty lines
		if (s == "#") {getline(fin,s,'\n'); continue;}	// skip #followed lines (comments)
		fin >> f;
		
		if		(s == "b_baseline")		b_baseline = f;
		else if (s == "b_constRg")		b_constRg = f;
		else if (s == "rGroup")			Rg = (b_constRg)? f:-1;		// if !b_constRg, set Rg = -1. This will help grouping function
		else if (s == "Rs_base")		Rs_base = f;
		else if (s == "rDisp")			rDisp = f;
	}

	fin >> s; 
	if (s != "SIM") {cout << "model grid not found!"; return 1;}
	while (fin >> s && s != attrbegin){
		if (s == "") continue;	// skip empty lines
		if (s == "#") {getline(fin,s,'\n'); continue;}	// skip #followed lines (comments)
		fin >> n;
		
		if		(s == "arenaSize")		arenaSize = n;
		else if (s == "moveSteps")		moveStepsPerGen = n;
		else if (s == "genMax")			genMax = n;
	}
	
	fin >> s; 
	if (s != "PARTICLES") {cout << "output variables not found!"; return 1;}
	while (fin >> s && s != attrbegin){
		if (s == "") continue;	// skip empty lines
		if (s == "#") {getline(fin,s,'\n'); continue;}	// skip #followed lines (comments)
		fin >> f;

		if 		(s == "Rs") 			host_params.Rs = f;
		else if (s == "kA")				host_params.kA = f;
		else if (s == "kO")				host_params.kO = f;
		else if (s == "dt")				host_params.dt = f;
		else if (s == "speed")			host_params.speed = f;
		else if (s == "Rr")				host_params.Rr = f;
		else if (s == "errSd")			host_params.errSd = f;
		else if (s == "turnRateMax")	host_params.turnRateMax = f*pi/180;		// convert to radians
	}

	fin >> s; 
	if (s != "SELECTION") {cout << "vars to use (debugging) not found!"; return 1;}
	while (fin >> s && s != attrbegin){
		if (s == "") continue;	// skip empty lines
		if (s == "#") {getline(fin,s,'\n'); continue;}	// skip #followed lines (comments)
		fin >> f;

		if (s == "b") 				b = f;
		else if (s == "RsNoiseSd")	RsNoiseSd = f;
	}

	fin >> s; 
	if (s != "INIT") {cout << "vars to use (debugging) not found!"; return 1;}
	while (fin >> s && s != attrbegin){
		if (s == "") continue;	// skip empty lines
		if (s == "#") {getline(fin,s,'\n'); continue;}	// skip #followed lines (comments)
		fin >> f;

		if (s == "fC0") 	fC0 = f;
	}

	fin >> s; 
	if (s != "OUTPUT") {cout << "vars to use (debugging) not found!"; return 1;}
	while (fin >> s && s != attrbegin){
		if (s == "") continue;	// skip empty lines
		if (s == "#") {getline(fin,s,'\n'); continue;}	// skip #followed lines (comments)
		fin >> n;

		if (s == "ngen_plot") 		ngen_plot = n;
		else if (s == "ngen_scan")	ngenScan = n;
		else if (s == "dataOut")	dataOut = n;
		else if (s == "plotsOut")	plotsOut = n;
		else if (s == "framesOut")	framesOut = n;
	}

	fin >> s; 
	if (s != "PARAM_SWEEPS") {cout << "parameter sweep vectors not found!"; return 1;}
	while (fin >> s && s != attrbegin){
		if (s == "") continue;	// skip empty lines
		if (s == "#") {getline(fin,s,'\n'); continue;}	// skip #followed lines (comments)

			 if (s == "c") 		while (fin >> f && f != -1) c.push_back(f);
		else if (s == "cS")		while (fin >> f && f != -1) cS.push_back(f);
		
		else if (s == "Fbase")		while (fin >> f && f != -1) fb_sweep.push_back(f);
//		else if (s == "arenaSize")	while (fin >> f && f != -1) as_sweep.push_back(f);
		else if (s == "Rs_base")	while (fin >> f && f != -1) rsb_sweep.push_back(f);
		else if (s == "rGroup")		while (fin >> f && f != -1) rg_sweep.push_back(f);
		else if (s == "ens")		while (fin >> f && f != -1) ens_sweep.push_back(f);
		else if (s == "mu")			while (fin >> f && f != -1) mu_sweep.push_back(f);
		else if (s == "moveSteps")	while (fin >> f && f != -1) nm_sweep.push_back(f);
	}

	fin.close();

	// errors 
	if (host_params.kA + host_params.kO > 1) {cout << "Fatal: kA + kO = " << host_params.kA + host_params.kO << " (>1)\n"; return 1;}
	if (nBlocks != c.size()) {cout << "Fatal: c values (" << c.size() << ") must match the number of blocks (" << nBlocks << ")\n"; return 1;}
	if (nBlocks != cS.size()) {cout << "Fatal: cS values (" << cS.size() << ") must match the number of blocks (" << nBlocks << ")\n"; return 1;}
	// since c and cS pairs will be assigned to each block, their sizes must be same as nBlocks

	// no errors found. Therefore continue with setting parameters
	
	printArray(&c[0], c.size(), "c");	
	printArray(&cS[0], cS.size(), "cS");	
	printArray(&fb_sweep[0], fb_sweep.size(), "Base fitness");	
//	printArray(&as_sweep[0], as_sweep.size(), "arena size");	
	printArray(&rsb_sweep[0], rsb_sweep.size(), "rs_base");
	printArray(&rg_sweep[0], rg_sweep.size(), "rg");
	printArray(&ens_sweep[0], ens_sweep.size(), "ensembles");
	printArray(&mu_sweep[0], mu_sweep.size(), "mutation rate");
	printArray(&nm_sweep[0], nm_sweep.size(), "move steps", " ", "\n");
	

	genMax = ((genMax-1)/ngen_plot +1)*ngen_plot;
	plotStep = genMax/ngen_plot;
	dataDir = outDir + "/data";
	framesDir = outDir + "/frames";
	b_anim_on 	= (graphicsQual == 0)? true:b_anim_on;
	nCoop = fC0*nFish;
	b_displayEveryStep = (graphicsQual > 0) && (dispInterval < 0);	// set if display must be updated every step
	
	animals.resize(nFish*nBlocks);
	offspring.resize(nFish);

	// grid
	int cellSizeMin = 10;
	nCellsMaxX = int(arenaSize/(cellSizeMin+1e-6))+1;
	nCellsMax = nCellsMaxX*nCellsMaxX;
	
	host_params.cosphi = cos(host_params.turnRateMax*host_params.dt);
	
	// boundaries
	float a = arenaSize/2;
	host_params.xmin = -a;
	host_params.xmax =  a;
	host_params.ymin = -a;
	host_params.ymax =  a;


	//initSimParams_default(host_params);
	return 0;
}



