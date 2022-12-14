#pragma once
#include "GridInterpolation.cuh"



__global__ void camUpdate(const float alpha, const int g, const float* camParam, float* cam) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < 7) cam[i] = (1.f - alpha) * camParam[g * 7 + i] + alpha * camParam[(g + 1) * 7 + i];
}


__global__ void pixInterpolation(const float2* viewthing, const int M, const int N, const int Gr, float2* thphi, const float2* grid,
	const int GM, const int GN, const float hor, const float ver, int* gapsave, int gridlvl,
	const float2* bhBorder, const int angleNum, const float alpha) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < N1 && j < M1) {
		float theta = viewthing[i * M1 + j].x + ver;
		float phi = fmodf(viewthing[i * M1 + j].y + hor + PI2c, PI2c);
		if (Gr > 1) {
			float2 A, B;
			float2 center = { .5f * bhBorder[0].x + .5f * bhBorder[0].y, .5f * bhBorder[1].x + .5f * bhBorder[1].y };
			float stretchRad = max(bhBorder[0].y - bhBorder[0].x, bhBorder[1].x - bhBorder[1].y) * 0.75f;
			float centerdist = (theta - center.x) * (theta - center.x) + (phi - center.y) * (phi - center.y);
			if (centerdist < stretchRad * stretchRad) {
				float angle = atan2(center.x - theta, phi - center.y);
				angle = fmodf(angle + PI2c, PI2c);
				int angleSlot = angle / PI2c * angleNum;

				float2 bhBorderNew = { (1.f - alpha) * bhBorder[2 * angleSlot + 2].x + alpha * bhBorder[2 * angleSlot + 3].x,
									   (1.f - alpha) * bhBorder[2 * angleSlot + 2].y + alpha * bhBorder[2 * angleSlot + 3].y };

				if (centerdist <= (bhBorderNew.x - center.x) * (bhBorderNew.x - center.x) + (bhBorderNew.y - center.y) * (bhBorderNew.y - center.y)) {
					thphi[i * M1 + j] = { -1, -1 };
					return;
				}

				float tStoB = (center.x - stretchRad * sinf(angle) - bhBorderNew.x);
				float pStoB = (center.y + stretchRad * cosf(angle) - bhBorderNew.y);

				float thetaPerc = fabsf(tStoB) < 1E-5 ? 0 : 1.f - (theta - bhBorderNew.x) / tStoB;
				float phiPerc = fabsf(pStoB) < 1E-5 ? 0 : 1.f - (phi - bhBorderNew.y) / pStoB;
				float thetaA = theta - thetaPerc * (bhBorderNew.x - bhBorder[2 * angleSlot + 2].x);
				float phiA = phi - phiPerc * (bhBorderNew.y - bhBorder[2 * angleSlot + 2].y);
				float thetaB = theta - thetaPerc * (bhBorderNew.x - bhBorder[2 * angleSlot + 3].x);
				float phiB = phi - phiPerc * (bhBorderNew.y - bhBorder[2 * angleSlot + 3].y);

				A = interpolatePix(thetaA, phiA, M, N, 0, gridlvl, grid, GM, GN, gapsave, i, j);
				B = interpolatePix(thetaB, phiB, M, N, 1, gridlvl, grid, GM, GN, gapsave, i, j);
			}
			else {
				A = interpolatePix(theta, phi, M, N, 0, gridlvl, grid, GM, GN, gapsave, i, j);
				B = interpolatePix(theta, phi, M, N, 1, gridlvl, grid, GM, GN, gapsave, i, j);

			}
			if (A.x == -1 || B.x == -1) thphi[i * M1 + j] = { -1, -1 };
			else {

				if (A.y < .2f * PI2c && B.y > .8f * PI2c) A.y += PI2c;
				if (B.y < .2f * PI2c && A.y > .8f * PI2c) B.y += PI2c;

				thphi[i * M1 + j] = { (1.f - alpha) * A.x + alpha * B.x, fmodf((1.f - alpha) * A.y + alpha * B.y, PI2c) };
			}
		}
		else {
			thphi[i * M1 + j] = interpolatePix(theta, phi, M, N, 0, gridlvl, grid, GM, GN, gapsave, i, j);
		}
	}
}

__device__ float2 interpolatePix(const float theta, const float phi, const int M, const int N, const int g, const int gridlvl,
	const float2* grid, const int GM, const int GN, int* gapsave, const int i, const int j) {
	int half = (phi < PIc) ? 0 : 1;
	int a = 0;
	int b = half * GM / 2;
	int gap = GM / 2;

	findBlock(theta, phi, g, grid, GM, GN, a, b, gap, gridlvl);
	gapsave[i * M1 + j] = gap;

	int k = a + gap;
	int l = b + gap;

	float factor = PI2c / (1.f * GM);
	float cornersCam[4] = { factor * a, factor * b, factor * k, factor * l };
	l = l % GM;
	float2 nul = { -1, -1 };
	float2 cornersCel[12] = { grid[g * GN * GM + a * GM + b], grid[g * GN * GM + a * GM + l], grid[g * GN * GM + k * GM + b], grid[g * GN * GM + k * GM + l],
									nul, nul, nul, nul, nul, nul, nul, nul };
	float2 thphiInter = interpolateSpline(a, b, gap, GM, GN, theta, phi, g, cornersCel, cornersCam, grid);

	if (!(thphiInter.x == -1 && thphiInter.y == -1)) wrapToPi(thphiInter.x, thphiInter.y);
	return thphiInter;
}

/// <summary>
/// Interpolates the corners of a projected pixel on the celestial sky to find the position
/// of a star in the (normal, unprojected) pixel in the output image.
/// </summary>
/// <param name="t0 - t4">The theta values of the projected pixel.</param>
/// <param name="p0 - p4">The phi values of the projected pixel.</param>
/// <param name="start, starp">The star theta and phi.</param>
/// <param name="sgn">The winding order of the polygon + for CW, - for CCW.</param>
/// <returns></returns>
__device__ void interpolate(float t0, float t1, float t2, float t3, float p0, float p1, float p2, float p3,
	float& start, float& starp, int sgn, int i, int j) {
	float error = 0.00001f;

	float midT = (t0 + t1 + t2 + t3) * .25f;
	float midP = (p0 + p1 + p2 + p3) * .25f;

	float starInPixY = 0.5f;
	float starInPixX = 0.5f;

	float perc = 0.5f;
	#pragma unroll
	for (int q = 0; q < 10; q++) {
		if ((fabs(start - midT) < error) && (fabs(starp - midP) < error)) break;

		float half01T = (t0 + t1) * .5f;
		float half23T = (t2 + t3) * .5f;
		float half12T = (t2 + t1) * .5f;
		float half03T = (t0 + t3) * .5f;
		float half01P = (p0 + p1) * .5f;
		float half23P = (p2 + p3) * .5f;
		float half12P = (p2 + p1) * .5f;
		float half03P = (p0 + p3) * .5f;

		float line01to23T = half23T - half01T;
		float line03to12T = half12T - half03T;
		float line01to23P = half23P - half01P;
		float line03to12P = half12P - half03P;

		float line01toStarT = start - half01T;
		float line03toStarT = start - half03T;
		float line01toStarP = starp - half01P;
		float line03toStarP = starp - half03P;

		int a = (((line03to12T * line03toStarP) - (line03toStarT * line03to12P)) > 0.f) ? 1 : -1;
		int b = (((line01to23T * line01toStarP) - (line01toStarT * line01to23P)) > 0.f) ? 1 : -1;

		perc *= 0.5f;

		if (sgn * a > 0) {
			if (sgn * b > 0) {
				t2 = half12T;
				t0 = half01T;
				t3 = midT;
				p2 = half12P;
				p0 = half01P;
				p3 = midP;
				starInPixX -= perc;
				starInPixY -= perc;
			}
			else {
				t2 = midT;
				t1 = half01T;
				t3 = half03T;
				p2 = midP;
				p1 = half01P;
				p3 = half03P;
				starInPixX -= perc;
				starInPixY += perc;
			}
		}
		else {
			if (sgn * b > 0) {
				t1 = half12T;
				t3 = half23T;
				t0 = midT;
				p1 = half12P;
				p3 = half23P;
				p0 = midP;
				starInPixX += perc;
				starInPixY -= perc;
			}
			else {
				t0 = half03T;
				t1 = midT;
				t2 = half23T;
				p0 = half03P;
				p1 = midP;
				p2 = half23P;
				starInPixX += perc;
				starInPixY += perc;
			}
		}
		midT = (t0 + t1 + t2 + t3) * .25f;
		midP = (p0 + p1 + p2 + p3) * .25f;
	}
	start = starInPixY;
	starp = starInPixX;
}

__device__ float2 intersection(const float ax, const float ay, const float bx, const float by, const float cx, const float cy, const float dx, const float dy) {
	// Line AB represented as a1x + b1y = c1 
	double a1 = by - ay;
	double b1 = ax - bx;
	double c1 = a1 * (ax)+b1 * (ay);

	// Line CD represented as a2x + b2y = c2 
	double a2 = dy - cy;
	double b2 = cx - dx;
	double c2 = a2 * (cx)+b2 * (cy);

	double determinant = a1 * b2 - a2 * b1;
	if (determinant == 0) {
		return{ -1, -1 };
	}
	float x = (b2 * c1 - b1 * c2) / determinant;
	float y = (a1 * c2 - a2 * c1) / determinant;
	return{ x, y };
}

__device__ float2 interpolateLinear(int i, int j, float percDown, float percRight, float2* cornersCel) {
	float phi[4] = { cornersCel[0].y, cornersCel[1].y, cornersCel[2].y, cornersCel[3].y };
	float theta[4] = { cornersCel[0].x, cornersCel[1].x, cornersCel[2].x, cornersCel[3].x };

	piCheck(phi, 0.2f);
	float leftT = theta[0] + percDown * (theta[2] - theta[0]);
	float leftP = phi[0] + percDown * (phi[2] - phi[0]);
	float rightT = theta[1] + percDown * (theta[3] - theta[1]);
	float rightP = phi[1] + percDown * (phi[3] - phi[1]);
	float upT = theta[0] + percRight * (theta[1] - theta[0]);
	float upP = phi[0] + percRight * (phi[1] - phi[0]);
	float downT = theta[2] + percRight * (theta[3] - theta[2]);
	float downP = phi[2] + percRight * (phi[3] - phi[2]);
	return intersection(upT, upP, downT, downP, leftT, leftP, rightT, rightP);
}

__device__ float2 hermite(float aValue, float2 const& aX0, float2 const& aX1, float2 const& aX2, float2 const& aX3,
	float aTension, float aBias) {
	/* Source:
	* http://paulbourke.net/miscellaneous/interpolation/
	*/

	float const v = aValue;
	float const v2 = v * v;
	float const v3 = v * v2;

	float const aa = (1.f + aBias) * (1.f - aTension) / 2.f;
	float const bb = (1.f - aBias) * (1.f - aTension) / 2.f;

	float const m0T = aa * (aX1.x - aX0.x) + bb * (aX2.x - aX1.x);
	float const m0P = aa * (aX1.y - aX0.y) + bb * (aX2.y - aX1.y);

	float const m1T = aa * (aX2.x - aX1.x) + bb * (aX3.x - aX2.x);
	float const m1P = aa * (aX2.y - aX1.y) + bb * (aX3.y - aX2.y);

	float const u0 = 2.f * v3 - 3.f * v2 + 1.f;
	float const u1 = v3 - 2.f * v2 + v;
	float const u2 = v3 - v2;
	float const u3 = -2.f * v3 + 3.f * v2;

	return{ u0 * aX1.x + u1 * m0T + u2 * m1T + u3 * aX2.x, u0 * aX1.y + u1 * m0P + u2 * m1P + u3 * aX2.y };
}

__device__ float2 findPoint(const int i, const int j, const int GM, const int GN, const int g,
	const int offver, const int offhor, const int gap, const float2* grid, int count) {
	float2 gridpt = grid[GM * GN * g + i * GM + j];
	if (gridpt.x == -2 && gridpt.y == -2) {
		//return{ -1, -1 };
		int j2 = (j + offhor * gap + GM) % GM;
		int i2 = i + offver * gap;
		float2 ij2 = grid[GM * GN * g + i2 * GM + j2];
		if (ij2.x == -1 && ij2.y == -1) return{ -1, -1 };

		else if (ij2.x != -2 && ij2.y != -2) {
			//return{ -1, -1 };

			int j0 = (j - offhor * gap + GM) % GM;
			int i0 = (i - offver * gap);

			float2 ij0 = grid[GM * GN * g + i0 * GM + j0];
			if (ij0.x < 0) return{ -1, -1 };

			int jprev = (j - 3 * offhor * gap + GM) % GM;
			int jnext = (j + 3 * offhor * gap + GM) % GM;
			int iprev = i - offver * 3 * gap;
			int inext = i + offver * 3 * gap;
			if (offver != 0) {
				if (i2 == 0) {
					jnext = (j0 + GM / 2) % GM;
					inext = i0;
				}
				else if (i0 == GN - 1) {
					jprev = (j0 + GM / 2) % GM;
					iprev = i2;
				}
				else if (i2 == GN - 1) {
					inext = i0;
					jnext = (j0 + GM / 2) % GM;
				}
			}
			float2 ijprev = grid[GM * GN * g + iprev * GM + jprev];
			float2 ijnext = grid[GM * GN * g + inext * GM + jnext];

			if (ijprev.x > -2 && ijnext.x > -2) {
				float2 pt[4] = { ijprev, ij0, ij2, ijnext };
				if (pt[0].x != -1 && pt[3].x != -1) {
					piCheckTot(pt, 0.2f, 4);
					return hermite(0.5f, pt[0], pt[1], pt[2], pt[3], 0.f, 0.f);
				}
			}
			float2 pt[2] = { ij2, ij0 };
			piCheckTot(pt, 0.2f, 2);
			return{ .5f * (pt[0].x + pt[1].x), .5f * (pt[0].y + pt[1].y) };
		}
		else {
			if (count > 0) return { -1.f, -1.f };

			int j0 = (j + gap) % GM;
			int j1 = (j - gap + GM) % GM;
			if (i - gap < 0) return{ -1, -1 };

			float2 cornersCel2[12];

			cornersCel2[0] = grid[GM * GN * g + (i + gap) * GM + j0];
			cornersCel2[1] = grid[GM * GN * g + (i - gap) * GM + j0];
			cornersCel2[2] = grid[GM * GN * g + (i - gap) * GM + j1];
			cornersCel2[3] = grid[GM * GN * g + (i + gap) * GM + j1];

			for (int q = 0; q < 4; q++) {
				if (cornersCel2[q].x == -1 || cornersCel2[q].x == -2) return { -1, -1 };
			}
			return interpolateHermite(i - gap, j1, 2 * gap, GM, GN, .5f, .5f, g, cornersCel2, grid, 1);
		}
	}
	return gridpt;
}

__device__ float2 interpolateHermite(const int i, const int j, const int gap, const int GM, const int GN, const float percDown, const float percRight,
	const int g, float2* cornersCel, const float2* grid, int count) {

	int k = i + gap;
	int l = (j + gap) % GM;
	int imin1 = i - gap;
	int kplus1 = k + gap;
	int jmin1 = (j - gap + GM) % GM;
	int lplus1 = (l + gap) % GM;
	int jx = j;
	int jy = j;
	int lx = l;
	int ly = l;

	if (i == 0) {
		jx = (j + GM / 2) % GM;
		lx = (jx + gap) % GM;
		imin1 = k;
	}
	else if (k == GN - 1) {
		jy = (j + GM / 2) % GM;
		ly = (jy + gap) % GM;
		kplus1 = i;
	}

	cornersCel[4] = findPoint(i, jmin1, GM, GN, g, 0, -1, gap, grid, count);		//4 upleft
	cornersCel[5] = findPoint(i, lplus1, GM, GN, g, 0, 1, gap, grid, count);		//5 upright
	cornersCel[6] = findPoint(k, jmin1, GM, GN, g, 0, -1, gap, grid, count);		//6 downleft
	cornersCel[7] = findPoint(k, lplus1, GM, GN, g, 0, 1, gap, grid, count);		//7 downright
	cornersCel[8] = findPoint(imin1, jx, GM, GN, g, -1, 0, gap, grid, count);		//8 lefthigh
	cornersCel[9] = findPoint(imin1, lx, GM, GN, g, -1, 0, gap, grid, count);		//9 righthigh
	cornersCel[10] = findPoint(kplus1, jy, GM, GN, g, 1, 0, gap, grid, count);		//10 leftdown
	cornersCel[11] = findPoint(kplus1, ly, GM, GN, g, 1, 0, gap, grid, count);		//11 rightdown

	for (int q = 4; q < 12; q++) {
		if (cornersCel[q].x == -1) return interpolateLinear(i, j, percDown, percRight, cornersCel);
	}
	piCheckTot(cornersCel, 0.2f, 12);

	float2 interpolateUp = hermite(percRight, cornersCel[4], cornersCel[0], cornersCel[1], cornersCel[5], 0.f, 0.f);
	float2 interpolateDown = hermite(percRight, cornersCel[6], cornersCel[2], cornersCel[3], cornersCel[7], 0.f, 0.f);
	float2 interpolateUpUp = { cornersCel[8].x + (cornersCel[9].x - cornersCel[8].x) * percRight,
		cornersCel[8].y + (cornersCel[9].y - cornersCel[8].y) * percRight };
	float2 interpolateDownDown = { cornersCel[10].x + (cornersCel[11].x - cornersCel[10].x) * percRight,
		cornersCel[10].y + (cornersCel[11].y - cornersCel[10].y) * percRight };
	//HERMITE FINITE
	return hermite(percDown, interpolateUpUp, interpolateUp, interpolateDown, interpolateDownDown, 0.f, 0.f);
}

__device__ float2 interpolateSpline(const int i, const int j, const int gap, const int GM, const int GN, const float thetaCam, const float phiCam, const int g,
	float2* cornersCel, float* cornersCam, const float2* grid) {

	float thetaUp = cornersCam[0];
	float thetaDown = cornersCam[2];
	float phiLeft = cornersCam[1];
	float phiRight = cornersCam[3];

	if (thetaUp == thetaCam) {
		if (phiLeft == phiCam)	return cornersCel[0];
		if (phiRight == phiCam)	return cornersCel[1];
		if (i == 0.f) return cornersCel[0];

	}
	else if (thetaDown == thetaCam) {
		if (phiLeft == phiCam) return cornersCel[2];
		if (phiRight == phiCam) return cornersCel[3];
	}

	for (int q = 0; q < 4; q++) {
		if (cornersCel[q].x == -1 && cornersCel[q].y == -1) return{ -1.f, -1.f };
	}

	float percDown = (thetaCam - thetaUp) / (thetaDown - thetaUp);
	float percRight = (phiCam - phiLeft) / (phiRight - phiLeft);

	return interpolateHermite(i, j, gap, GM, GN, percDown, percRight, g, cornersCel, grid, 0);
	//return interpolateLinear(i, j, percDown, percRight, cornersCel);
}