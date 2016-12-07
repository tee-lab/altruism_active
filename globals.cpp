#include "params.h"
#include "globals.h"
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	Note: All arrays will be of the 
	 	  following alignment
	
		x   ----->  
	y
			f1 f2 f3 f4 .... f_nFish 
	|	b1  
	|	b2  
	v	b3 
  		.
		.
		b_nBlocks
	
	single array (say wS for block iB)
	will be indexed as

	wS[ix2(i=1:nFish, iB, nFish)] 	
			 ^ ^

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


// constants
int cDevice = 0;
int genMax;	// set it to least multiple of  >= P_genMax
int plotStep;
string outDir;		// full output dir path
string dataDir;
string framesDir;
int moveStepsPerGen;	// movement steps per generation
float arenaSize;		// arena size determines density
float RsNoiseSd; 	// mutation rate in Rs


// sim params (to be set manually before each run)
SimParams host_params;//, *dev_params;
float fitness_base = 0; 		// base fitness
vector <float> c, cS;	// costs
vector <float> mu(1,0); 		// mutation rate
float Rg, Rs_base; // R for grouping and social interaction

// ensemble params
int iEns  = 0;	// formerly iRunSet. ensemble number. each ensemble consists of nBlocks runs
int iExpt = 0;	// formerly exptID.  each expt consists of 1 or more ensembles. 
string exptDesc, exptDescFull;	// exptDesc is unique for each ensemble, edfull is unique for each run

// GPU random number generator states and seeds
curandState * dev_XWstates;
int *seeds_h, *seeds_dev; 

// Host random number generator
curandGenerator_t generator_host;
int seed_cpu;

// movement
int gRun;	// which run to display
//int imstep = 0;		// current movement step
bool b_anim_on; // start immediately if graphics off, else user's choice

// device state arrays
float2 * pos_dev, * vel_dev;
float  * Rs_dev;
float  * kA_dev, * kO_dev;	//  kA and kO, the attraction and orientation constants

// host state arrays
vector <Particle> animals; 		// vector for all parents (this must include all blocks)
vector <Particle> offspring;	// vector for all offspring (1 block is sufficient)

map <int, int> g2ng_map, g2kg_map; 		// gid -> x maps, x is ng or kg
int genNum = 0, stepNum=0;							// current generation number and movement step number
int nCoop;					// current # atruists
float EpgNg = 0, varPg = 0, r = 0, pbar = 0, dp = 0, Skg2bNg = 0, SkgbNg = 0, r2 = 0;		// 

// grid
int *gridCount, *cummCount;
int *gridCount_dev, *cummCount_dev, *filledCount_dev, *pStartIds_dev, *pEndIds_dev;	// grid sized arrays
int *cellIds_dev, *sortedIds_dev; 	// n-particles sized arrays
int nCellsX = 0, nCells = 0; 		// number of cells in X, total (X*X), and cell Size. these must be init from max Rs in population
int nCellsMaxX;		// This is the maximum number of cells possible. Memory will be allocated for these many cells
int nCellsMax;
float cellSize = 1;

// output streams
//ofstream ng_fout[nBlocks], kg_fout[nBlocks], gid_fout[nBlocks], 
//	   wa_fout[nBlocks], rs_fout[nBlocks], p_fout[nBlocks], fit_fout[nBlocks], gix_fout[nBlocks],
//	   ko_fout[nBlocks], ka_fout[nBlocks];
//ofstream fout_fcs;

ofstream * p_fout;

// dirs
string homeDir_path, outDir_name, exptName;
int wks, gpu;					// gpu on which to run

// population
int nFish   = 512;	// number of fish in each run
int nBlocks = 1;	// number of runs

// graphics
int graphicsQual = 3;			// 0 = no graphics, 1 = basic graphics, 2 = good graphics, 3 = fancy graphics, charts etc
int dispInterval;
bool b_displayEveryStep;

bool  b_baseline = false;		// Baseline  - is this a baseline experiment (Rs = 0)
bool  b_constRg  = true;		// Grouping  - use constant radius of grouping?

// experiment properties
float rDisp = -1;				// Dispersal - Radius of dispersal. (-1 for random dispersal)

// selection and mutation
float b = 100;				// benefit value

// init
float fC0 = 0.0; 				// initial frequency of cooperators

// output
int  ngen_plot = 25;			// number of points in generation axis to output in terminal/plots 
int  ngenScan = 100;		 	// number of end generations to average over for parameter scan graphs

bool dataOut = true;			// output all values in files?
bool plotsOut = false;		// output plots
bool framesOut = false;		// output frames

// parameter sweep vectors
vector <float> fb_sweep, as_sweep, nfish_sweep, rsb_sweep, rg_sweep, ens_sweep, mu_sweep, nm_sweep;



/*


xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

int cDevice = 0;
int genMax, plotStep;
float fitness_base; 
// sim params
SimParams host_params;
SimParams * dev_params;
string outDir, plotsDir, dataDir;


// ========= Parameters that change after 1 ensemble of sims is done ===========
// current device, and ensembleID
int iEns;	// formerly iRunSet
float *c, *cS, *mu;
float Rag; // Ra for grouping

int exptID;
string exptDesc, exptDescFull;

// GPU random number generator
curandState * dev_XWstates;
int *seeds_h, *seeds_dev; 


// =========== rapidly changing parameters =====================================
int gRun = blockToDisplay;
int imstep = 0;
bool b_anim_on = false;


// ====== device arrays ========================================================
float2 * pos_array;
float2 * vel_array;
float * Rs_array;
float * kA_array, * kO_array;	//  kA and kO, the attarction and orientation constants


// ======= host arrays =========================================================
vector <Particle> animals[nBlocks] = {vector <Particle> (nFish)}; 		// vector for all parents
vector <Particle> offsprings(nFish); 	// vector for all offspring

// ======= quantities that change at generation advance ========================
// 	all the following maps map each group to various quantities related to it. 
// 	Each group has a unique ID.
// 	Size of the map is the total number of groups
map <int, int> g2ng_map, g2kg_map; 
int genNum = 0;
int nCoop = fC0*nFish;
float EpgNg, varPg, r, pbar, dp;


// output streams
ofstream ng_fout[nBlocks], kg_fout[nBlocks], gid_fout[nBlocks], 
	   wa_fout[nBlocks], rs_fout[nBlocks], p_fout[nBlocks], fit_fout[nBlocks], gix_fout[nBlocks];
//ofstream fout_fcs;

*/
