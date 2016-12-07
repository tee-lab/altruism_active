#ifndef PARTICLES_H
#define PARTICLES_H

#include <string>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

// Definition of class particle and associated functions. This can function as an independent library
// variables can be added to this class

class Particle{
	public:
	float2 pos;		// position
	float2 vel;		// velocity
	int	   wA;		// Strategy
	float  Rs;		// radius of flocking interaction
	float  kA;		// k attraction
	float  kO;		// k alignment

	int gID;		// group ID
	int ng;			// group size
	int kg;			// number of cooperators in group
	int ancID;		// original ancestor from which this individual descends

	float fitness;	// fitness

	int wkg;		// number of individuals in group with kO > kA 
};

void printParticles(Particle * pvec, int n = 6);

void findGroups(Particle * pvec, int n, float dx, float dy, float rGrp = -1);


// ------------ this code specific functions --------------------

void updateGroupIndices(int iblock);
int updateGroupSizes(int iblock);
float update_r(int iblock);
int updateGroups(int iblock);

int animateParticles();

int disperse(int iblock, int R);

void displayDevArrays();

#endif
