#include "GridLookup.cuh"

__global__ void makeGrid(const int g, const int GM, const int GN, const int GN1, float2* grid, const float2* hashTable,
	const int2* hashPosTag, const int2* offsetTable, const int2* tableSize, const char count, const int sym) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < GN && j < GM) {
		float2 lookup = hashLookup({ i, j }, hashTable, hashPosTag, offsetTable, tableSize, g);
		grid[count * GM * GN1 + i * GM + j] = lookup;
		if (sym == 1) {	//FIX HERE!!!
			if (lookup.x != -1 && lookup.x != -2) lookup.x = PIc - lookup.x;
			int x = GN1 - i - 1;
			grid[count * GM * GN1 + x * GM + j] = lookup;
			//printf("%d %d %d %f\n", i, x, j, lookup.x);

		}
	}
}

/// <summary>
/// Checks and corrects phi values for 2-pi crossings.
/// </summary>
/// <param name="p">The phi values to check.</param>
/// <param name="factor">The factor to check if a point is close to the border.</param>
/// <returns></returns>
__device__ bool piCheck(float* p, float factor) {
	float factor1 = PI2c * (1.f - factor);
	bool check = false;
#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (p[q] > factor1) {
			check = true;
			break;
		}
	}
	if (!check) return false;
	check = false;
	float factor2 = PI2c * factor;
#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (p[q] < factor2) {
			p[q] += PI2c;
			check = true;
		}
	}
	return check;
}

__device__ void findBlock(const float theta, const float phi, const int g, const float2* grid,
	const int GM, const int GN, int& i, int& j, int& gap, const int level) {

	for (int s = 0; s < level + 1; s++) {
		int ngap = gap / 2;
		int k = i + ngap;
		int l = j + ngap;
		if (gap <= 1 || grid[g * GN * GM + k * GM + l].x == -2.f) return;
		else {
			float thHalf = PI2c * k / (1.f * GM);
			float phHalf = PI2c * l / (1.f * GM);
			if (thHalf <= theta) i = k;
			if (phHalf <= phi) j = l;
			gap = ngap;
		}
	}
}

/// <summary>
/// Checks and corrects phi values for 2-pi crossings.
/// </summary>
/// <param name="p">The phi values to check.</param>
/// <param name="factor">The factor to check if a point is close to the border.</param>
/// <returns></returns>
__device__ bool piCheckTot(float2* tp, float factor, int size) {
	float factor1 = PI2c * (1.f - factor);
	bool check = false;
	for (int q = 0; q < size; q++) {
		if (tp[q].y > factor1) {
			check = true;
			break;
		}
	}
	if (!check) return false;
	check = false;
	float factor2 = PI2c * factor;
	for (int q = 0; q < size; q++) {
		if (tp[q].y < factor2) {
			tp[q].y += PI2c;
			check = true;
		}
	}
	return check;
}

// Set values for projected pixel corners & update phi values in case of 2pi crossing.
__device__ void retrievePixelCorners(const float2* thphi, float* t, float* p, int& ind, const int M, bool& picheck, float offset) {
	t[0] = thphi[ind + M1].x;
	t[1] = thphi[ind].x;
	t[2] = thphi[ind + 1].x;
	t[3] = thphi[ind + M1 + 1].x;
	p[0] = thphi[ind + M1].y;
	p[1] = thphi[ind].y;
	p[2] = thphi[ind + 1].y;
	p[3] = thphi[ind + M1 + 1].y;

#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (p[q] < 0 || t[q] < 0) {
			ind = -1;
			return;
		}
		p[q] = fmodf(p[q] + offset, PI2c);
	}
	// Check and correct for 2pi crossings.
	picheck = piCheck(p, .2f);
}


__device__ void wrapToPi(float& thetaW, float& phiW) {
	thetaW = fmodf(thetaW, PI2c);
	while (thetaW < 0.f) thetaW += PI2c;
	if (thetaW > PIc) {
		thetaW -= 2.f * (thetaW - PIc);
		phiW += PIc;
	}
	while (phiW < 0.f) phiW += PI2c;
	phiW = fmod(phiW, PI2c);
}

__device__ int2 hash1(int2 key, int ow) {
	return{ (key.x + ow) % ow, (key.y + ow) % ow };
}

__device__ int2 hash0(int2 key, int hw) {
	return{ (key.x + hw) % hw, (key.y + hw) % hw };
}

__device__ float2 hashLookup(int2 key, const float2* hashTable, const int2* hashPosTag, const int2* offsetTable, const int2* tableSize, const int g) {

	int ow = tableSize[g].y;
	int hw = tableSize[g].x;
	int ostart = 0;
	int hstart = 0;
	for (int q = 0; q < g; q++) {
		hstart += tableSize[q].x * tableSize[q].x;
		ostart += tableSize[q].y * tableSize[q].y;
	}

	int2 index = hash1(key, ow);
	//printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", g, ow, hw, ow*ow, hw*hw, ostart, hstart, index.x, index.y, ostart + index.x*ow + index.y, );

	int2 add = { hash0(key, hw).x + offsetTable[ostart + index.x * ow + index.y].x,
				 hash0(key, hw).y + offsetTable[ostart + index.x * ow + index.y].y };
	int2 hindex = hash0(add, hw);

	if (hashPosTag[hstart + hindex.x * hw + hindex.y].x != key.x || hashPosTag[hstart + hindex.x * hw + hindex.y].y != key.y) return{ -2.f, -2.f };
	else return hashTable[hstart + hindex.x * hw + hindex.y];
}
