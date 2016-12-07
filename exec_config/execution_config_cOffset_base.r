# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Input parameters for Simulations
# If you change the order of parameters below, you will get what you deserve
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

> DIR
# Directories for data output
homeDir_path	/home/jaideep/expts_5/output_n1024		# home dir - no spaces allowed
outDir_name  	pVsC_offset		# output dir name
exptName 	 	ens 						# expt name

> GPU
# GPU and workstation
wks 			1			# workstation on which to run
gpu 			1			# gpu on which to run

> GPU_CONFIG
# population
particles 		1024		# number of particles in each run
runs	 		32			# number of parallel runs

> GRAPHICS
# graphics
graphicsQual 	0			# 0 = no graphics, 1 = basic graphics, 2 = good graphics, 3 = fancy graphics, charts etc
dispInterval  	-8 		# display interval in ms, -ve = number of mv steps to run before display
gRun 			0			# Run index for which to show graphics at start
b_anim_on 		0		  	# turn animation on immediately on start? if false (0), wait for user to press "a" to start sim

> EXPT
# experiment properties
b_baseline 		1			# Baseline  - is this a baseline experiment (Rs = 0)
b_constRg  		1			# Grouping  - use constant radius of grouping?
rGroup			2			# Grouping  - radius of grouping / social interaction
rDisp 			-1			# Dispersal - Radius of dispersal. (-1 for random dispersal)

> SIM
# movement and SimParams
arenaSize 		300			# the size of the entire space (x & y), when body size = 1 --> determines density
genMax 			2000		# number of generations to simulate. **Nearest upper multiple of nPoints_plot will be taken 

> PARTICLES
# default particle parameters
Rs 				1			# initial Rs = radius of attraction (social interaction radius) (min Value = Rr = 1)
kA  			0.4			# initial kA = weight given to attraction direction while moving
kO  			0.4			# initial kO
dt  			0.2			# time step (for movement)
speed  			1			# particle speed
Rr 				1			# radius of repulsion (body size) (NEVER CHANGE THIS)
errSd 			0.05		# SD of error in following desired direction
turnRateMax 	50			# degrees per sec	

> SELECTION
# selection and mutation
b 				100			# benefit value
RsNoiseSd 		0.1 		# SD of noise in ws at selection wS = wS(parent) + noise*rnorm(0,1)

> INIT
# init
fC0 			0.0			# initial frequency of cooperators

> OUTPUT
# output
ngen_plot 		25			# number of points in generation axis to output in terminal/plots 
dataOut  		1			# output all values in files?
plotsOut  		0			# output plots
framesOut 		0			# output frames


> PARAM_SWEEPS
# parameter sets to loop over.
# IF YOU CHANGE THE ORDER OF THE PARAMETERS BELOW, YOU WILL GET WHAT YOU DESERVE
# c 			0.1 0.2 0.4 0.8	-1
c			0.1 0.12 0.14 0.16 0.19 0.22 0.26 0.3 0.35 0.41 0.48 0.56 0.65 0.77 0.89 1.05 1.22 1.43 1.67 1.96 2.29 2.68 3.13 3.66 4.28 5 5.85 6.84 8 9.36 10.95 12.8  -1
#c 			0.09 0.11 0.13 0.15 0.17 0.2 0.24 0.28 0.32 0.38 0.44 0.52 0.6 0.71 0.83 0.97 1.13 1.32 1.55 1.81 2.12 2.47 2.89 3.38 3.96 4.63 5.41 6.33 7.4 8.66 10.12 11.84 -1
cS 			2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2  -1
Fbase		1 	-1
Rs_base		0	-1
rGroup		2	-1
mu			0.5		-1
moveSteps	1000	-1
ens			1 2 3 4 5 6 7 8 9 10	-1




