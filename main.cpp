#include <iostream>
using namespace std;

#include "globals.h"
#include "init.h"
#include "graphics.h"
#include "altruism.h"
#include "utils/simple_timer.h"
#include "utils/simple_io.h"
#include "utils/cuda_device.h" 


// this function expects that simulation params be updated beforehand
int launchExpt(){

	SimpleTimer clockExpt; clockExpt.reset();
	clockExpt.start();

	cout << "\n> Launch Experiment: " << exptDesc << "_runSet(" << iEns << ")\n";
	cout << "   > Simulate " << genMax << " generations, plot after every " << plotStep << " steps.\n   > "; 
	cout.flush();

	// start run
//	int k=0, g=0;
	while(1){	// infinite loop needed to poll anim_on signal.
		if (graphicsQual > 0) glutMainLoopEvent();		

		// animate particles
		if (b_anim_on) {
			animateParticles(); 
			++stepNum;
			if (b_displayEveryStep && stepNum % (-dispInterval) == 0) displayDevArrays();	// update display if every step update is on
		}

		// if indefinite number of steps desired, skip gen-advance check
		if (moveStepsPerGen < 0) continue;	

		// check if generation is to be advanced.
		if (stepNum >= moveStepsPerGen){
			stepNum = 0;
			advanceGen();	// Note: CudaMemCpy at start and end of advanceGen() ensure that all movement kernel calls are done
			++genNum;			
			if (genNum % plotStep == 0) { cout << "."; cout.flush(); }
		}

		// when genMax genenrations are done, end run
		if (genNum >= genMax) break;
	}

	// end run, reset counters
	cout << " > DONE. "; // << "Return to main().\n\n";
	printTime_hhmm(clockExpt.getTime());
	stepNum = genNum = 0;
	++iExpt;
	return 0;
}

#define SWEEP_LOOP(x) 	for (int i_##x =0; i_##x < x##_sweep.size(); ++i_##x)


int main(int argc, char **argv){

	// select device
	cDevice = initDevice(argc, argv);

	// read execution parameters
	string config_filename = "execution_config.r";
	if (argc >2) config_filename = argv[2];
	int e = read_execution_config_file(config_filename);
	if (e == 1) {cout << "Fatal error(s) in parameter values. Cant continue\n\n"; return 1;}

	// create output dirs
	if (dataOut || plotsOut || framesOut) system(string("mkdir " + outDir).c_str());
	if (dataOut)   system(string("mkdir " + dataDir).c_str());
	if (plotsOut)  system(string("mkdir " + outDir + "/plots").c_str());
	if (framesOut) system(string("mkdir " + framesDir).c_str());
	
	// allocate memory
	allocArrays();

	// if graphics are on, initGL
	if (graphicsQual > 0) initGL(&argc, argv, host_params);

	// for all the chosen parameter sweeps, init arrays and launch simulations
	SWEEP_LOOP(mu){ 
	SWEEP_LOOP(nm){ 
	SWEEP_LOOP(fb){ 
//	SWEEP_LOOP(as){ 
	SWEEP_LOOP(rg){ 
	SWEEP_LOOP(rsb){ 

		// set parameters
		mu[0] 			= mu_sweep[i_mu];
		moveStepsPerGen = nm_sweep[i_nm];
		fitness_base 	= fb_sweep[i_fb];
//		arenaSize 		= as_sweep[i_as];
		Rg 				= rg_sweep[i_rg];
		Rs_base 		= rsb_sweep[i_rsb];

		// for each parameter set, perform as many ensembles as specified in ens_sweep[]
		SWEEP_LOOP(ens){ 
			iEns = ens_sweep[i_ens];
			initStateArrays();
			launchExpt();
		}
			
	}}}}}//}
		
	return 0;
}


