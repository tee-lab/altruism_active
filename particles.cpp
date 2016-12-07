#include "particles.h"
#include "utils/cuda_vector_math.cuh"
#include "utils/simple_io.h"
#include "utils/cuda_device.h"

#include "graphics.h"

using namespace std;

void printParticles(Particle * pvec, int n){
	cout << "Particles:\n";
	cout << "ancID " << "\t" 
		 << "gID   " << "\t" 
		 << "wA   " << "\t" 
		 << "Rs   " << "\t" 
		 << "ng   " << "\t" 
		 << "kg   " << "\t" 
		 << "fit  " << "\t" 
		 << "wkg  " << "\n";
	for (int i=0; i<n; ++i){
		cout << pvec[i].ancID << "\t" 
			 << pvec[i].gID << "\t" 
			 << pvec[i].wA << "\t" 
			 << pvec[i].Rs << "\t" 
			 << pvec[i].ng << "\t" 
			 << pvec[i].kg << "\t" 
			 << pvec[i].fitness << "\t" 
			 << pvec[i].wkg << "\n";
	}
	cout << "\n";
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 	Indentify groups of particles based on equivalence classes algo
//
//  Inputs: Vector of particles - pvec
//  		number of particles to consider - n
//			dimensions of the total space - dx, dy
//			radius of grouping - rGrp (default = -1)
//				- if rGrp < 0, particle Rs is used. Else, rGrp is used
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void findGroups(Particle * pvec, int n, float dx, float dy, float rGrp){

	vector <int> eq(n);	// temporary array of group indices

	// loop row by row over the lower triangular matrix
	for (int myID=0; myID<n; ++myID){
		eq[myID] = myID;
		for (int otherID=0; otherID< myID; ++otherID) {	

			eq[otherID] = eq[eq[otherID]];
			// calculate distance
			float2 v2other = periodicDisplacement( pvec[myID].pos, pvec[otherID].pos, dx, dy);
			float d2other = length(v2other);
			
			// set Radius of grouping from const or Rs
			float R_grp = (rGrp < 0)? pvec[myID].Rs : rGrp;	// rGrp < 0 means use particle Rs. Else, use rGrp
			// if distance is < R_grp, assign same group
			if (d2other < R_grp){
				eq[eq[eq[otherID]]] = myID;
			} 

		}
	}
	
	// complete assignment of eq. class for all individuals.
	for (int j = 0; j < n; j++) eq[j] = eq[eq[j]]; 

	// copy these group indices into the iblock'th row of particles
	memcpy2D( (Particle*) &(pvec[0].gID), (int*) &eq[0], sizeof(int), n);
	
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Functions specific to this particular implementation
// - these functions use the global variables defined in globals.h
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include "globals.h"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// updateGroups() - use global variables specific to this code with fingGroups()
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void updateGroupIndices(int iblock){
	findGroups( &animals[ix2(0, iblock, nFish)], nFish, 								// pvec, n
				host_params.xmax-host_params.xmin, host_params.ymax-host_params.ymin, 	// dx, dy
				Rg);																	// rGrp
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// update group-sizes and cooperators/group from group Indices 
// relies on : groupIndices - must be called after updateGroups
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int updateGroupSizes(int iblock){

	// delete previous group sizes
	g2ng_map.clear(); g2kg_map.clear();
	nCoop = 0;
	
	// calculate new group sizes and indices
	for (int i=0; i<nFish; ++i) {
		Particle &p = animals[ix2(i,iblock,nFish)];
	
		++g2ng_map[p.gID]; 
		if (p.wA == Cooperate ) {
			++g2kg_map[p.gID];
			++nCoop;
		}
	}
	
	return nCoop;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// calculate r by 2 different methods. 
// Assumes fitness  V = (k-wA)b/n - c wA 
// relies on : ng and kg maps updated by updateGroupSizes()
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
float update_r(int iblock){
	// calculate r and related quantities
	pbar = nCoop/float(nFish);
	r = varPg = EpgNg = 0;
	for (int i=0; i<nFish; ++i){
		Particle *p = &animals[ix2(i,iblock,nFish)];
		p->kg = g2kg_map[p->gID];	// number of cooperators in the group
		p->ng = g2ng_map[p->gID];		// number of individuals in the group

		EpgNg += float(p->kg)/p->ng/p->ng;
		varPg += (float(p->kg)/p->ng-pbar)*(float(p->kg)/p->ng-pbar);
	}
	EpgNg /= nFish;
	varPg /= nFish;
	
	// calc r by another method (should match with r calc above)
	r2 = Skg2bNg = SkgbNg = 0;
	for (map <int,int>::iterator it = g2kg_map.begin(); it != g2kg_map.end(); ++it){
		float kg_g = it->second;
		float ng_g = g2ng_map[it->first];
		Skg2bNg += kg_g*kg_g/ng_g;
		SkgbNg  += kg_g/ng_g;
	}

	if (nCoop == 0 || nCoop == nFish) r = r2 = -1e20;	// put nan if p is 0 or 1
	else {
		r  = varPg/pbar/(1-pbar) - EpgNg/pbar;
		r2 = float(nFish)/nCoop/(nFish-nCoop)*Skg2bNg - float(nCoop)/(nFish-nCoop) - SkgbNg/nCoop;
	}

	return r;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// call the 3 functions above to update all info about groups
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int updateGroups(int iblock){
	updateGroupIndices(iblock);
	updateGroupSizes(iblock);
	update_r(iblock);
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Disperse particles to random locations within radius R of current pos 
// if R == -1, disperse in the entire space
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int disperse(int iblock, int R = -1){
	for (int i=0; i<nFish; ++i){
		Particle *p = &animals[ix2(i, iblock, nFish)];
		
		// new velocity in random direction
		p->vel = runif2(1.0); 
		
		if (R == -1){ // random dispersal
			p->pos  = runif2(host_params.xmax, host_params.ymax); 
		}
		else{	// dispersal within radius R
			float2 dx_new = runif2(R, R);  // disperse offspring within R radius of parent
			p->pos += dx_new;
			makePeriodic(p->pos.x, host_params.xmin, host_params.xmax); 
			makePeriodic(p->pos.y, host_params.ymin, host_params.ymax); 
		}
	}	
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Copy particles from GPU and display them 
// THIS FUNCTION IS SLOW. USE ONLY WHEN PARTICLES ARE NOT UPDATED ON CPU
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void displayDevArrays(){
	// copy only the particles that need to be displayed, i.e. in the gRun'th block
	cudaMemcpy2D( (void*)&(animals[ix2(0,gRun,nFish)].pos),  sizeof(Particle), (void*) &pos_dev[ix2(0,gRun,nFish)],  sizeof(float2), sizeof(float2), nFish,    cudaMemcpyDeviceToHost);
	cudaMemcpy2D( (void*)&(animals[ix2(0,gRun,nFish)].vel),  sizeof(Particle), (void*) &vel_dev[ix2(0,gRun,nFish)],  sizeof(float2), sizeof(float2), nFish,    cudaMemcpyDeviceToHost);
	//             ^ dst                                     ^ dst pitch        ^ src                                ^ src pitch     ^ bytes/elem    ^ n_elem  ^ direction

	updateGroups(gRun);
	glutPostRedisplay();
}


void launch_movement_kernel();
									
										
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// move particles for 1 movement step
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int animateParticles(){

//	static int n_movement_steps = 0;	
	
	// execute 1 movement step
	launch_movement_kernel();
	getLastCudaError("movement_kernel_launch");
	calcFPS(1);
//	++n_movement_steps; 

//	if (n_movement_steps % 100 == 0) b_anim_on = false;

	// update display with GPU arrays, if every-step-update is on 

	return 0;
}



