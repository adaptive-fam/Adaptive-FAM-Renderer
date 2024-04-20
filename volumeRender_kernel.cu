#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>

#define X_BLOCK_NUM_LEVEL_1	  2
#define Y_BLOCK_NUM_LEVEL_1   2
#define Z_BLOCK_NUM_LEVEL_1	  2
#define X_BLOCK_NUM_LEVEL_2	  4
#define Y_BLOCK_NUM_LEVEL_2	  4
#define Z_BLOCK_NUM_LEVEL_2	  4
#define X_BLOCK_NUM_LEVEL_3	  8
#define Y_BLOCK_NUM_LEVEL_3	  8
#define Z_BLOCK_NUM_LEVEL_3	  8
#define X_BLOCK_NUM_LEVEL_4	  16
#define Y_BLOCK_NUM_LEVEL_4	  16
#define Z_BLOCK_NUM_LEVEL_4	  16


typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;

typedef float VolumeType;

texture<VolumeType, 3, cudaReadModeElementType> tex;         // 3D texture
cudaTextureObject_t texObj = 0;
cudaTextureObject_t *texObjList;
texture<float4, 1, cudaReadModeElementType>         transferTex; // 1D transfer function texture

cudaExtent allBlockSize = make_cudaExtent(2, 2, 2); // dimension of sub blocks of the original volume
int* selectListKernel;
int* visibleBlocksListKernel;
int visibleBlocksSizeKernel;
int selectBlockSizeKernel;

float* cacheKnotDataKernel;
float* cacheCtrlptsDataKernel;
int* cacheKnotSizeListKernel;

int knotDimensionSampleSizeKernel; // 79
int ctrlptsDimensionSampleSizeKernel; // 76
int degreeKernel;

typedef struct
{
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__
float3 getNormal(float px, float py, float pz, int size_x, int size_y, int size_z)
{
     float3 gradient;
     float3 theCentDiffSpace;

	 theCentDiffSpace.x = 1.0/size_x;
	 theCentDiffSpace.y = 1.0/size_y;
	 theCentDiffSpace.z = 1.0/size_z;

	 gradient.x = tex3D(tex, px + theCentDiffSpace.x, py, pz)
           		  -tex3D(tex, px, py, pz);
      
     gradient.y = tex3D(tex, px, py + theCentDiffSpace.y, pz)
           		  -tex3D(tex, px, py, pz);
      
     gradient.z = tex3D(tex, px, py, pz + theCentDiffSpace.z)
		   		  -tex3D(tex, px, py, pz);

      
     gradient = gradient * 10.0;
      
     if(length(gradient) > 0.0) {
         gradient = normalize(gradient);
     }
      
     return gradient;
}

__device__
int findSpan(float* start, float v, int knotDimensionSampleSize, int ctrlptsDimensionSampleSizeKernel, int degreeKernel)
{
	if (v == start[ctrlptsDimensionSampleSizeKernel]) {
		return ctrlptsDimensionSampleSizeKernel - 1;
	}
	
	int low = degreeKernel;
	int high = ctrlptsDimensionSampleSizeKernel;
	int mid = (low + high)/2;
	while (v < start[mid] || v >= start[mid + 1]){
		if (v < start[mid]) {
			high = mid;
		} else {
			low = mid;
		}
		mid = (low + high)/2;
	}

	return mid;
}

__global__ void
d_render(uint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale, int* selectList, int* visibleBlocksList, int visibleBlocksSize, 
		 int allBlockSizeWidth_level_1, int allBlockSizeHeight_level_1, int allBlockSizeDepth_level_1,
		 int allBlockSizeWidth_level_2, int allBlockSizeHeight_level_2, int allBlockSizeDepth_level_2,
		 int allBlockSizeWidth_level_3, int allBlockSizeHeight_level_3, int allBlockSizeDepth_level_3,
		 int allBlockSizeWidth_level_4, int allBlockSizeHeight_level_4, int allBlockSizeDepth_level_4,
		 int selectBlockSize, 
		 float sampleDistance, 
		 int size_x, int size_y, int size_z,
		 int knotDimensionSampleSizeKernel, int ctrlptsDimensionSampleSizeKernel,
		 float* cacheKnotDataKernel, float* cacheCtrlptsDataKernel)
{
    const int maxSteps = 500000000;
    const float tstep = sampleDistance;

    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;


    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -8.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;


	// bool flag = true;
    for (int i = 0; i < maxSteps; i++)
    {
		float allBlockSizeWidth;
		float allBlockSizeHeight;
		float allBlockSizeDepth;
		bool found = false;
		float x_shift;
		float y_shift;
		float z_shift;
		float unit_x;
		float unit_y;
		float unit_z;
		int x_idx;
		int y_idx;
		int z_idx;
		int idx;
		int current_level;
		for (int level = 0; level < 4; level++) {
			if (level == 0) {
				allBlockSizeWidth = allBlockSizeWidth_level_1;
				allBlockSizeHeight = allBlockSizeHeight_level_1;
				allBlockSizeDepth = allBlockSizeDepth_level_1;
			}
			if (level == 1) {
				allBlockSizeWidth = allBlockSizeWidth_level_2;
				allBlockSizeHeight = allBlockSizeHeight_level_2;
				allBlockSizeDepth = allBlockSizeDepth_level_2;
			}
			if (level == 2) {
				allBlockSizeWidth = allBlockSizeWidth_level_3;
				allBlockSizeHeight = allBlockSizeHeight_level_3;
				allBlockSizeDepth = allBlockSizeDepth_level_3;
			}
			if (level == 3) {
				allBlockSizeWidth = allBlockSizeWidth_level_4;
				allBlockSizeHeight = allBlockSizeHeight_level_4;
				allBlockSizeDepth = allBlockSizeDepth_level_4;
			};

			// find the block containing the current sample position
			x_shift = pos.x*0.5f+0.5f; // from (-1, 1) to (0, 1)
			y_shift = pos.y*0.5f+0.5f; // from (-1, 1) to (0, 1)
			z_shift = pos.z*0.5f+0.5f; // from (-1, 1) to (0, 1)
			unit_x = 1/(float)allBlockSizeWidth;
			unit_y = 1/(float)allBlockSizeHeight;
			unit_z = 1/(float)allBlockSizeDepth;
			x_idx = x_shift/unit_x;
			y_idx = y_shift/unit_y;
			z_idx = z_shift/unit_z;
			idx = z_idx*allBlockSizeWidth*allBlockSizeHeight + y_idx*allBlockSizeWidth + x_idx;
			if (level == 0) {
				idx = idx;
				current_level = level;
			}
			if (level == 1) {
				idx = idx + 8;
				current_level = level;
			}
			if (level == 2) {
				idx = idx + 8 + 64;
				current_level = level;
			}
			if (level == 3) {
				idx = idx + 8 + 64 + 512;
				current_level = level;
			}

			if (selectList[idx] != 0) {
				found = true;
				break;
			}
		}

		float sample = 0.0;
		float px, py, pz;
		float diffuse = 1.0; 

		if (found) {
			// find index in select block list
			int counter = 0;
			for (int j = 0; j < visibleBlocksSize; j++) { // all blocks in cache_mem
				if (visibleBlocksList[j] == idx) {
					break;
				} else {
					counter++;
				}
			}
			float unit = 1.0/((float)visibleBlocksSize*76 + (float)visibleBlocksSize - 1 + 1); //  for initCuda
			float z_shift_unit = unit*(76 + 1);

			float x_left = x_shift - unit_x*(float)x_idx;
			float y_left = y_shift - unit_y*(float)y_idx;
			float z_left = z_shift - unit_z*(float)z_idx;
			px = x_left*(float)allBlockSizeWidth;
			py = y_left*(float)allBlockSizeHeight;
			pz = z_left*(float)allBlockSizeDepth;
						
			if (px > 1.0) {px = 1.0;}
			if (py > 1.0) {py = 1.0;}
			if (pz > 1.0) {pz = 1.0;}

			if (px < 0.0) {px = 0.0;}
			if (py < 0.0) {py = 0.0;}
			if (pz < 0.0) {pz = 0.0;}

			// Adjustment for gradient extended block size, extended block size = traditional block size + 1
			
			float uunit = 1.0/77.0;
			px = px*75.0*uunit + uunit/2.0;
			py = py*75.0*uunit + uunit/2.0;
        	pz = (float)counter*z_shift_unit + unit/2.0 + pz*75.0*unit;

			sample = tex3D(tex, px, py, pz);
			
		}

        // lookup in transfer function texture
        float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale);

        col.w *= density;

        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;

#if 1 // shading on
	    if (col.w > 0.001) {
			float3 normal = getNormal(px, py, pz, size_x, size_y, size_z);
			// diffuse = max(dot(normal, eyeRay.d), dot(eyeRay, -normal));
			diffuse = 0.5 +  max(normal.x*eyeRay.d.x + normal.y*eyeRay.d.y + normal.z*eyeRay.d.z, 
						  (-normal.x*eyeRay.d.x) + (-normal.y*eyeRay.d.y) + (-normal.z*eyeRay.d.z));
			col.x *= diffuse;
			col.y *= diffuse;
			col.z *= diffuse;
		}
#endif

        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);

        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t += tstep;

        if (t > tfar) break;

        pos += step;
    }

    sum *= brightness;

    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

__global__ void
d_render_mfa_adaptive(uint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale, int* selectList, int* visibleBlocksList, int visibleBlocksSize, 
		 int allBlockSizeWidth_level_1, int allBlockSizeHeight_level_1, int allBlockSizeDepth_level_1,
		 int allBlockSizeWidth_level_2, int allBlockSizeHeight_level_2, int allBlockSizeDepth_level_2,
		 int allBlockSizeWidth_level_3, int allBlockSizeHeight_level_3, int allBlockSizeDepth_level_3,
		 int allBlockSizeWidth_level_4, int allBlockSizeHeight_level_4, int allBlockSizeDepth_level_4,
		 int selectBlockSize, 
		 float sampleDistance, 
		 int size_x, int size_y, int size_z,
		 int knotDimensionSampleSizeKernel, int ctrlptsDimensionSampleSizeKernel,
		 float* cacheKnotDataKernel, float* cacheCtrlptsDataKernel, int degreeKernel, int* cacheKnotSizeListKernel)
{
    // const int maxSteps = 50000;
    const int maxSteps = 500000000;
    // const float tstep = 0.1f;
    const float tstep = sampleDistance;

    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    // const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    // eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = normalize(make_float3(u, v, -8.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;


	// MFA related container -------start-------
	// 1. Decoder Class members used: N, cs, ds, q, tot_iters, jumps 
	float N_0[3];
	float N_1[3];
	float N_2[3];
	N_0[0] = 1;
	N_0[1] = 0;
	N_0[2] = 0;
	N_1[0] = 1;
	N_1[1] = 0;
	N_1[2] = 0;
	N_2[0] = 1;
	N_2[1] = 0;
	N_2[2] = 0;

	float cs[3];
	cs[0] = 1;
	cs[1] = 1;
	cs[2] = 1;
	int ds[3];
	ds[0] = 1;
	ds[1] = 1;
	ds[2] = 1;
	int q[3];
	q[0] = 3;
	q[1] = 3;
	q[2] = 3;

	int tot_iters = (degreeKernel + 1)*(degreeKernel + 1)*(degreeKernel + 1); // 27
	float ct[27][3]; // 27x3
	for (int i = 0; i < tot_iters; i++) {
		int div = tot_iters;
		int i_temp = i;
		for (int j = 3 - 1; j >= 0; j--) {
			div /= (degreeKernel + 1);
			ct[i][j] = i_temp/div;
			i_temp -= (ct[i][j]*div);
		}
	}
	float jumps[27];

	// 2. FastDecoderInfo Struct members used: t, td
	float t_0[9];
	float t_1[3];
	float t_2[1];

	float td_0_0[9];
	float td_0_1[3];
	float td_0_2[1];
	float td_1_0[9];
	float td_1_1[3];
	float td_1_2[1];
	float td_2_0[9];
	float td_2_1[3];
	float td_2_2[1];
	float td_3_0[9];
	float td_3_1[3];
	float td_3_2[1];

	memset(td_0_0, 0, sizeof(td_0_0));
	memset(td_0_1, 0, sizeof(td_0_1));
	memset(td_0_2, 0, sizeof(td_0_2));
	memset(td_1_0, 0, sizeof(td_1_0));
	memset(td_1_1, 0, sizeof(td_1_1));
	memset(td_1_2, 0, sizeof(td_1_2));
	memset(td_2_0, 0, sizeof(td_2_0));
	memset(td_2_1, 0, sizeof(td_2_1));
	memset(td_2_2, 0, sizeof(td_2_2));
	memset(td_3_0, 0, sizeof(td_3_0));
	memset(td_3_1, 0, sizeof(td_3_1));
	memset(td_3_2, 0, sizeof(td_3_2));

	// 3. FastDecoderInfo.ResizeDers(1)
	float D[3][2][3];
	memset(D, 0, sizeof(D));

	float* M_0[3];
	float* M_1[3];
	float* M_2[3];
	float* M_3[3];
	
	M_0[0] = &D[0][1][0]; 
	M_0[1] = &D[1][0][0];
	M_0[2] = &D[2][0][0];

	M_1[0] = &D[0][0][0];
	M_1[1] = &D[1][1][0];
	M_1[2] = &D[2][0][0];

	M_2[0] = &D[0][0][0];
	M_2[1] = &D[1][0][0];
	M_2[2] = &D[2][1][0];

	M_3[0] = &D[0][0][0];
	M_3[1] = &D[1][0][0];
	M_3[2] = &D[2][0][0];
	
	// 4. Find span
	int span_x, span_y, span_z;

	float left[3];
	float right[3];
	left[0] = 0;
	left[1] = 0;
	left[2] = 0;
	right[0] = 0;
	right[1] = 0;
	right[2] = 0;

	float ndu[3][3];
	memset(ndu, 0, sizeof(ndu));

	int start_ctrl_idx = 0;
	int q0 = degreeKernel + 1; // 3
	int ctrl_idx;
	
	float* cacheKnotDataKernelX;
	float* cacheKnotDataKernelY;
	float* cacheKnotDataKernelZ;
	float* cacheCtrlptsDataKernelXYZ;
	// MFA related container -------end-------

    for (int f = 0; f < maxSteps; f++)
    {
        // read from 3D texture
        // remap position to [0, 1] coordinates
		// regulare x, y, z
		if (pos.x <= -1)
			pos.x = -1 + 0.000001;
		if (pos.x >= 1)
			pos.x = 1 - 0.000001;
		if (pos.y <= -1)
			pos.y = -1 + 0.000001;
		if (pos.y >= 1)
			pos.y = 1 - 0.000001;
		if (pos.z <= -1)
			pos.z = -1 + 0.000001;
		if (pos.z >= 1)
			pos.z = 1 - 0.000001;

		float allBlockSizeWidth;
		float allBlockSizeHeight;
		float allBlockSizeDepth;
		bool found = false;
		float x_shift;
		float y_shift;
		float z_shift;
		float unit_x;
		float unit_y;
		float unit_z;
		int x_idx;
		int y_idx;
		int z_idx;
		int idx;
		for (int level = 0; level < 4; level++) {
			if (level == 0) {
				allBlockSizeWidth = allBlockSizeWidth_level_1;
				allBlockSizeHeight = allBlockSizeHeight_level_1;
				allBlockSizeDepth = allBlockSizeDepth_level_1;
			}
			if (level == 1) {
				allBlockSizeWidth = allBlockSizeWidth_level_2;
				allBlockSizeHeight = allBlockSizeHeight_level_2;
				allBlockSizeDepth = allBlockSizeDepth_level_2;
			}
			if (level == 2) {
				allBlockSizeWidth = allBlockSizeWidth_level_3;
				allBlockSizeHeight = allBlockSizeHeight_level_3;
				allBlockSizeDepth = allBlockSizeDepth_level_3;
			}
			if (level == 3) {
				allBlockSizeWidth = allBlockSizeWidth_level_4;
				allBlockSizeHeight = allBlockSizeHeight_level_4;
				allBlockSizeDepth = allBlockSizeDepth_level_4;
			};
			
			// find the block containing the current sample position
			x_shift = pos.x*0.5f+0.5f; // from (-1, 1) to (0, 1)
			y_shift = pos.y*0.5f+0.5f; // from (-1, 1) to (0, 1)
			z_shift = pos.z*0.5f+0.5f; // from (-1, 1) to (0, 1)
			if (x_shift <= 0)
				x_shift = 0 + 0.000001;
			if (x_shift >= 1)
				x_shift = 1 - 0.000001;
			if (y_shift <= 0)
				y_shift = 0 + 0.000001;
			if (y_shift >= 1)
				y_shift = 1 - 0.000001;
			if (z_shift <= 0)
				z_shift = 0 + 0.000001;
			if (z_shift >= 1)
				z_shift = 1 - 0.000001;

			unit_x = 1/(float)allBlockSizeWidth;
			unit_y = 1/(float)allBlockSizeHeight;
			unit_z = 1/(float)allBlockSizeDepth;
			x_idx = x_shift/unit_x;
			y_idx = y_shift/unit_y;
			z_idx = z_shift/unit_z;
			idx = z_idx*allBlockSizeWidth*allBlockSizeHeight + y_idx*allBlockSizeWidth + x_idx;
			if (level == 0) {
				idx = idx;
			}
			if (level == 1) {
				idx = idx + 8;
			}
			if (level == 2) {
				idx = idx + 8 + 64;
			}
			if (level == 3) {
				idx = idx + 8 + 64 + 512;
			}
			if (selectList[idx] != 0) { // if idx is one of the visable block
				found = true;
				break;
			}
		}

		float sample = 0.0;
		float px, py, pz;
		float diffuse = 1.0; 
		float3 normal;
		
		if (found) {
			// find index in select block list
			int counter = 0;
			for (int j = 0; j < visibleBlocksSize; j++) { // all blocks in cache_mem
				if (visibleBlocksList[j] == idx) {
					break;
				} else {
					counter++;
				}
			}
			

			float x_left = x_shift - unit_x*(float)x_idx;
			float y_left = y_shift - unit_y*(float)y_idx;
			float z_left = z_shift - unit_z*(float)z_idx;
			px = x_left*(float)allBlockSizeWidth;
			py = y_left*(float)allBlockSizeHeight;
			pz = z_left*(float)allBlockSizeDepth;
			
			if (px > 1.0) {px = 1.0;}
			if (py > 1.0) {py = 1.0;}
			if (pz > 1.0) {pz = 1.0;}

			if (px < 0.0) {px = 0.0;}
			if (py < 0.0) {py = 0.0;}
			if (pz < 0.0) {pz = 0.0;}		

			// find starting pointer location
			int start_x = counter*knotDimensionSampleSizeKernel*3;
			int start_y = counter*knotDimensionSampleSizeKernel*3 + cacheKnotSizeListKernel[counter];
			int start_z = counter*knotDimensionSampleSizeKernel*3 + cacheKnotSizeListKernel[counter] + cacheKnotSizeListKernel[counter];
			cacheKnotDataKernelX = cacheKnotDataKernel + start_x;
			cacheKnotDataKernelY = cacheKnotDataKernel + start_y;
			cacheKnotDataKernelZ = cacheKnotDataKernel + start_z;
			
			// find span for x, y, and z dimension from FindSpan()
			int start_xyz = counter*ctrlptsDimensionSampleSizeKernel*ctrlptsDimensionSampleSizeKernel*ctrlptsDimensionSampleSizeKernel;
			cacheCtrlptsDataKernelXYZ = cacheCtrlptsDataKernel + start_xyz;
			span_x = findSpan(cacheKnotDataKernelX, px, cacheKnotSizeListKernel[counter], cacheKnotSizeListKernel[counter] - 1 - 2, degreeKernel);
			span_y = findSpan(cacheKnotDataKernelY, py, cacheKnotSizeListKernel[counter], cacheKnotSizeListKernel[counter] - 1 - 2, degreeKernel);
			span_z = findSpan(cacheKnotDataKernelZ, pz, cacheKnotSizeListKernel[counter], cacheKnotSizeListKernel[counter] - 1 - 2, degreeKernel);

			left[0] = 0;
			left[1] = 0;
			left[2] = 0;
			right[0] = 0;
			right[1] = 0;
			right[2] = 0;

			// fill ndu, compute 0th derivatives, for x dimention
			memset(ndu, 0, sizeof(ndu));
			ndu[0][0] = 1;
            for (int j = 1; j <= 2; j++)
            {
                left[j]  = px - cacheKnotDataKernelX[span_x + 1 - j];
                right[j] = cacheKnotDataKernelX[span_x + j] - px;
                float saved = 0.0;
                for (int r = 0; r < j; r++)
                {
                    // lower triangle
                    ndu[j][r] = 1/(right[r + 1] + left[j - r]);
                    float temp = ndu[r][j - 1]*ndu[j][r];
                    // upper triangle
                    ndu[r][j] = saved + right[r + 1]*temp;
                    saved = left[j - r]*temp;
                }
                ndu[j][j] = saved;
            }
			// Copy 0th derivatives
			for (int j = 0; j <= 2; j++)
                D[0][0][j] = ndu[j][2];
			// Compute 1st derivatives
            float d = 0.0;
            D[0][1][0] = -ndu[0][1]*ndu[2][0];
            D[0][1][2] = ndu[1][1]*ndu[2][1];
            for (int r = 1; r < 2; r++)
            {
                d = ndu[r-1][1]*ndu[2][r - 1];
                d += -ndu[r][1]*ndu[2][r];
                D[0][1][r] = d;
            }
			// multiply through by the correct factors in eq. 2.10
            for (int i = 0; i <= 2; i++)
            {
                D[0][1][i] *= 2;
            }

			// fill ndu, compute 0th derivatives, for y dimention
			memset(ndu, 0, sizeof(ndu));
			ndu[0][0] = 1;
            for (int j = 1; j <= 2; j++)
            {
                left[j]  = py - cacheKnotDataKernelY[span_y + 1 - j];
                right[j] = cacheKnotDataKernelY[span_y + j] - py;
                float saved = 0.0;
                for (int r = 0; r < j; r++)
                {
                    // lower triangle
                    ndu[j][r] = 1/(right[r + 1] + left[j - r]);
                    float temp = ndu[r][j - 1]*ndu[j][r];
                    // upper triangle
                    ndu[r][j] = saved + right[r + 1]*temp;
                    saved = left[j - r]*temp;
                }
                ndu[j][j] = saved;
            }
			// Copy 0th derivatives
			for (int j = 0; j <= 2; j++)
                D[1][0][j] = ndu[j][2];
			// Compute 1st derivatives
            d = 0.0;
            D[1][1][0] = -ndu[0][1]*ndu[2][0];
            D[1][1][2] = ndu[1][1]*ndu[2][1];
            for (int r = 1; r < 2; r++)
            {
                d = ndu[r-1][1]*ndu[2][r - 1];
                d += -ndu[r][1]*ndu[2][r];
                D[1][1][r] = d;
            }
			// multiply through by the correct factors in eq. 2.10
            for (int i = 0; i <= 2; i++)
            {
                D[1][1][i] *= 2;
            }

			// fill ndu, compute 0th derivatives, for z dimention
			memset(ndu, 0, sizeof(ndu));
			ndu[0][0] = 1;
            for (int j = 1; j <= 2; j++)
            {
                left[j]  = pz - cacheKnotDataKernelZ[span_z + 1 - j];
                right[j] = cacheKnotDataKernelZ[span_z + j] - pz;
                float saved = 0.0;
                for (int r = 0; r < j; r++)
                {
                    // lower triangle
                    ndu[j][r] = 1/(right[r + 1] + left[j - r]);
                    float temp = ndu[r][j - 1]*ndu[j][r];
                    // upper triangle
                    ndu[r][j] = saved + right[r + 1]*temp;
                    saved = left[j - r]*temp;
                }
                ndu[j][j] = saved;
            }
			// Copy 0th derivatives
			for (int j = 0; j <= 2; j++)
                D[2][0][j] = ndu[j][2];
			// Compute 1st derivatives
            d = 0.0;
            D[2][1][0] = -ndu[0][1]*ndu[2][0];
            D[2][1][2] = ndu[1][1]*ndu[2][1];
            for (int r = 1; r < 2; r++)
            {
                d = ndu[r-1][1]*ndu[2][r - 1];
                d += -ndu[r][1]*ndu[2][r];
                D[2][1][r] = d;
            }
			// multiply through by the correct factors in eq. 2.10
            for (int i = 0; i <= 2; i++)
            {
                D[2][1][i] *= 2;
            }

			// update the cs and jumps according to current ctrlptsDimensionSampleSizeKernel which is cacheKnotSizeListKernel[counter] - 1 - 2
			for (int i = 0; i < 3; i++) {
					if (i > 0) {
						// cs[i] = cs[i - 1]*ctrlptsDimensionSampleSizeKernel;
						cs[i] = cs[i - 1]*(cacheKnotSizeListKernel[counter] - 1 - 2);
						ds[i] = ds[i - 1]*q[i];
					}
			}
			for (int i = 0; i < tot_iters; i++) {
				jumps[i] = ct[i][0]*cs[0] + ct[i][1]*cs[1] + ct[i][2]*cs[2];
			}

			start_ctrl_idx += (span_x - degreeKernel)*cs[0];
			start_ctrl_idx += (span_y - degreeKernel)*cs[1];
			start_ctrl_idx += (span_z - degreeKernel)*cs[2];

		    for (int m = 0, id = 0; m < tot_iters; m += q0, id++)
            {
                ctrl_idx = start_ctrl_idx + jumps[m];
                // Separate 1st iteration to avoid zero-initialization
                td_0_0[id] = M_0[0][0]*cacheCtrlptsDataKernelXYZ[ctrl_idx];
                t_0[id] = M_1[0][0]*cacheCtrlptsDataKernelXYZ[ctrl_idx];
                for (int a = 1; a < q0; a++)
                {
                    // For this first loop, there are only two cases: multiply control 
                    // points by the basis functions, or multiply control points by 
                    // derivative of basis functions. We save time by only computing 
                    // each case once, and then copying the result as needed, below.
                    td_0_0[id] += M_0[0][a]*cacheCtrlptsDataKernelXYZ[ctrl_idx + a];    // der basis fun * ctl pts
                    t_0[id] += M_1[0][a]*cacheCtrlptsDataKernelXYZ[ctrl_idx + a];        // basis fun * ctl pts
                }
            }
			for (int id = 0; id < 9; id++) {
				td_1_0[id] = t_0[id];
			}
			for (int id = 0; id < 9; id++) {
				td_2_0[id] = t_0[id];
			}
			for (int id = 0; id < 9; id++) {
				td_3_0[id] = t_0[id];
			}

		// d = 0, 1, 2, 3; k = 1, 2
            int qcur = 0, tsz = 0;
		// d = 0; k = 1;
			qcur = 3;
            tsz = 9; // size of td_0_0
            for (int m = 0, id = 0; m < tsz; m += qcur, id++)
            {
                td_0_1[id] = M_0[1][0] * td_0_0[m];
                for (int l = 1; l < qcur; l++)
                {
                    td_0_1[id] += M_0[1][l] * td_0_0[m + l];
                }
            }
		// d = 0; k = 2;
			qcur = 3;
        	tsz = 3; // size of td_0_1
			for (int m = 0, id = 0; m < tsz; m += qcur, id++)
            {
                td_0_2[id] = M_0[2][0] * td_0_1[m];
                for (int l = 1; l < qcur; l++)
                {
                    td_0_2[id] += M_0[2][l] * td_0_1[m + l];
                }
            }
		// d = 1; k = 1;
			qcur = 3;
        	tsz = 9; // size of td_1_0            
			for (int m = 0, id = 0; m < tsz; m += qcur, id++)
            {
                td_1_1[id] = M_1[1][0] * td_1_0[m];
                for (int l = 1; l < qcur; l++)
                {
                    td_1_1[id] += M_1[1][l] * td_1_0[m + l];
                }
            }
		// d = 1; k = 2;		
			qcur = 3;
        	tsz = 3; // size of td_1_1           
			for (int m = 0, id = 0; m < tsz; m += qcur, id++)
            {
                td_1_2[id] = M_1[2][0] * td_1_1[m];
                for (int l = 1; l < qcur; l++)
                {
                    td_1_2[id] += M_1[2][l] * td_1_1[m + l];
                }
            }
		// d = 2; k = 1;		
			qcur = 3;
        	tsz = 9; // size of td_2_0            
			for (int m = 0, id = 0; m < tsz; m += qcur, id++)
            {
                td_2_1[id] = M_2[1][0] * td_2_0[m];
                for (int l = 1; l < qcur; l++)
                {
                    td_2_1[id] += M_2[1][l] * td_2_0[m + l];
                }
            }
		// d = 2; k = 2;		
			qcur = 3;
        	tsz = 3; // size of td_2_1           
			for (int m = 0, id = 0; m < tsz; m += qcur, id++)
            {
                td_2_2[id] = M_2[2][0] * td_2_1[m];
                for (int l = 1; l < qcur; l++)
                {
                    td_2_2[id] += M_2[2][l] * td_2_1[m + l];
                }
            }
		// d = 3; k = 1;		
			qcur = 3;
        	tsz = 9; // size of td_3_0            
			for (int m = 0, id = 0; m < tsz; m += qcur, id++)
            {
                td_3_1[id] = M_3[1][0] * td_3_0[m];
                for (int l = 1; l < qcur; l++)
                {
                    td_3_1[id] += M_3[1][l] * td_3_0[m + l];
                }
            }
		// d = 3; k = 2;		
			qcur = 3;
        	tsz = 3; // size of td_3_1            
			for (int m = 0, id = 0; m < tsz; m += qcur, id++)
            {
                td_3_2[id] = M_3[2][0] * td_3_1[m];
                for (int l = 1; l < qcur; l++)
                {
                    td_3_2[id] += M_3[2][l] * td_3_1[m + l];
                }
            }
			
			normal.x = td_0_2[0];
			normal.y = td_1_2[0];
			normal.z = td_2_2[0];
			// normal = normalize(normal);
			sample = td_3_2[0];
			if (idx == 0 || idx == 1 || idx == 2 || idx == 3) {
				// sample = 0.0;
			} else {
				// sample = 0.0;
			}
			// sample = 1.0;
		}

        // lookup in transfer function texture
        float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale); // transferOffset = 0; transferScale = 1
        col.w *= density; // density = 1

        // "under" operator for back-to-front blending

        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;

#if 1 // turn on and off shading
	    if (col.w > 0.001) {


			// -------query gradient (start) -------	
			normal = normalize(normal);

			float3 light;
			light.x = 0.0;
			light.y = 1.0;
			light.z = -1.0;
			light = normalize(light);
			diffuse = 0.5 +  max(normal.x*eyeRay.d.x + normal.y*eyeRay.d.y + normal.z*eyeRay.d.z, 
			 			  (-normal.x*eyeRay.d.x) + (-normal.y*eyeRay.d.y) + (-normal.z*eyeRay.d.z));
			col.x *= diffuse;
			col.y *= diffuse;
			col.z *= diffuse;
		}
#endif

        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);

        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t += tstep;

        if (t > tfar) break;

        pos += step;


		// Reset MFA container
		N_0[0] = 1;
		N_0[1] = 0;
		N_0[2] = 0;
		N_1[0] = 1;
		N_1[1] = 0;
		N_1[2] = 0;
		N_2[0] = 1;
		N_2[1] = 0;
		N_2[2] = 0;
			
		start_ctrl_idx = 0;
		memset(td_0_0, 0, sizeof(td_0_0));
		memset(td_0_1, 0, sizeof(td_0_1));
		memset(td_0_2, 0, sizeof(td_0_2));
		memset(td_1_0, 0, sizeof(td_1_0));
		memset(td_1_1, 0, sizeof(td_1_1));
		memset(td_1_2, 0, sizeof(td_1_2));
		memset(td_2_0, 0, sizeof(td_2_0));
		memset(td_2_1, 0, sizeof(td_2_1));
		memset(td_2_2, 0, sizeof(td_2_2));
		memset(td_3_0, 0, sizeof(td_3_0));
		memset(td_3_1, 0, sizeof(td_3_1));
		memset(td_3_2, 0, sizeof(td_3_2));
    }

    sum *= brightness;
    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

__global__ void
d_render_ml_gt(uint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale, int* selectList, int* visibleBlocksList, int visibleBlocksSize, 
		 int allBlockSizeWidth_level_1, int allBlockSizeHeight_level_1, int allBlockSizeDepth_level_1,
		 int allBlockSizeWidth_level_2, int allBlockSizeHeight_level_2, int allBlockSizeDepth_level_2,
		 int allBlockSizeWidth_level_3, int allBlockSizeHeight_level_3, int allBlockSizeDepth_level_3,
		 int allBlockSizeWidth_level_4, int allBlockSizeHeight_level_4, int allBlockSizeDepth_level_4,
		 int selectBlockSize, 
		 float sampleDistance, 
		 int size_x, int size_y, int size_z,
		 int knotDimensionSampleSizeKernel, int ctrlptsDimensionSampleSizeKernel,
		 float* cacheKnotDataKernel, float* cacheCtrlptsDataKernel)
{
    // const int maxSteps = 50000;
    const int maxSteps = 500000000;
    // const float tstep = 0.1f;
    const float tstep = sampleDistance;
    // const float tstep = 0.02f;

    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    // const float3 boxMin = make_float3(0.0f, 0.0f, 0.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;


    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    // eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = normalize(make_float3(u, v, -8.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;


	// bool flag = true;
    for (int i = 0; i < maxSteps; i++)
    {
		float fm = 6.0;
		float alpha = 0.05; // flat ML dataset
		float rad = sqrt(pos.x*pos.x + pos.y*pos.y);
		float rho = cos(2.0*M_PI*fm*cos(M_PI*rad/2.0));
		float sample = (1.0 - sin(M_PI*pos.z/2.0) + alpha*(1.0 + rho))/(2*(1.0 + alpha));
		float diffuse = 1.0; 

        // lookup in transfer function texture
        float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale);

        col.w *= density;

        // "under" operator for back-to-front blending

        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;
#if 1 // shading on
	    if (col.w > 0.001) {


			float3 normal;
			float radd = sqrt(pos.x*pos.x + pos.y*pos.y);
			float delta = -2.0*M_PI*fm*sin(M_PI/2.0*radd)*M_PI/4.0*1.0/radd;
			normal.x = -alpha/(1.0 + alpha)*sin(2.0*M_PI*fm*cos(M_PI*radd/2.0))*delta*2.0*pos.x;
			normal.y = -alpha/(1.0 + alpha)*sin(2.0*M_PI*fm*cos(M_PI*radd/2.0))*delta*2.0*pos.y;
			normal.z = -1.0/2.0/(1.0 + alpha)*cos(M_PI/2.0*pos.z)*M_PI/2.0;
          	normal = normalize(normal);


			// float3 normal = getNormal(px, py, pz, size_x, size_y, size_z);
			// diffuse = max(dot(normal, eyeRay.d), dot(eyeRay, -normal));
			diffuse = 0.5 +  max(normal.x*eyeRay.d.x + normal.y*eyeRay.d.y + normal.z*eyeRay.d.z, 
						  (-normal.x*eyeRay.d.x) + (-normal.y*eyeRay.d.y) + (-normal.z*eyeRay.d.z));
			col.x *= diffuse;
			col.y *= diffuse;
			col.z *= diffuse;
		}
#endif

        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);

        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t += tstep;

        if (t > tfar) break;

        pos += step;
    }

    sum *= brightness;

    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

extern "C"
void setTextureFilterMode(bool bLinearFilter)
{
    tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

float getValue(float start, float startValue, float end, float endValue, float ref) {
	if (startValue > endValue) {
		float l = end - ref;
		float ll = end - start;
		float hh = startValue - endValue;
		return l/ll*hh + endValue;
	} else if (startValue < endValue) {
		float l = ref;
		float ll = end - start;
		float hh = endValue - startValue;
		return l/ll*hh + startValue;
	} else {
		return startValue;
	}
}


void getTransferFunc(float4* transferFunc, int colorTFSize, float4* colorTransferFunc, int opacityTFSize, float2* opacityTransferFunc) {
	float r[101];
	float g[101];
	float b[101];
	float a[101];
	// fill color
	for (int i = 0; i < 101; i++) {
		float currentValue = (float)i/100; // [0, 1]
		for (int j = 0; j < colorTFSize - 1; j++) {
			float lowerBound = colorTransferFunc[j].x;
			float upperBound = colorTransferFunc[j + 1].x;
			if (currentValue >= lowerBound && currentValue <= upperBound) {
				r[i] = getValue(lowerBound, colorTransferFunc[j].y, upperBound, colorTransferFunc[j + 1].y, currentValue);
				g[i] = getValue(lowerBound, colorTransferFunc[j].z, upperBound, colorTransferFunc[j + 1].z, currentValue);
				b[i] = getValue(lowerBound, colorTransferFunc[j].w, upperBound, colorTransferFunc[j + 1].w, currentValue);
			}
		}
	}
	// fill opacity
	for (int i = 0; i < 101; i++) {
		float currentValue = (float)i/100; // [0, 1]
		for (int j = 0; j < opacityTFSize - 1; j++) {
			float lowerBound = opacityTransferFunc[j].x;
			float upperBound = opacityTransferFunc[j + 1].x;
			if (currentValue >= lowerBound && currentValue <= upperBound) {
				a[i] = getValue(lowerBound, opacityTransferFunc[j].y, upperBound, opacityTransferFunc[j + 1].y, currentValue);
			}
		}
	}
	for (int i = 0; i < 101; i++) {
		transferFunc[i] = {r[i], g[i], b[i], a[i]};
		// printf("%f", r[i]);
	}
}

extern "C"
void initCuda(void *h_volume,
			   cudaExtent volumeSize,
			   int* selectList,
			   int* visibleBlocksList,
			   int visibleBlocksSize,
			   int selectBlockSize,
			   int knot_dimension_sample_size,
			   int ctrlpts_dimension_sample_size,
			   int degree,
			   float* cache_knot_data_d,
			   float* cache_ctrlpts_data_d,
			   int* cache_knot_size_list)
{
#if 1
	checkCudaErrors(cudaFreeArray(d_volumeArray));
    checkCudaErrors(cudaFreeArray(d_transferFuncArray));

	selectListKernel = selectList;
	visibleBlocksListKernel = visibleBlocksList;
	visibleBlocksSizeKernel = visibleBlocksSize; // cache size = 200
	selectBlockSizeKernel = selectBlockSize; // visible block number

	knotDimensionSampleSizeKernel = knot_dimension_sample_size;
	ctrlptsDimensionSampleSizeKernel = ctrlpts_dimension_sample_size;
	cacheKnotDataKernel = cache_knot_data_d;
	cacheCtrlptsDataKernel = cache_ctrlpts_data_d;
	degreeKernel = degree;

	cacheKnotSizeListKernel = cache_knot_size_list;

    // create 3D array
	clock_t start, end;
	start = clock();
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));
	end = clock();
	// printf("Loading from CPU Mem to GPU Mem: %4.6f\n", (double)((double)(end - start)/CLOCKS_PER_SEC));

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

#endif
    // create transfer function texture

#if 1 // Flame dataset
	float4 transferFunc[101];
	int colorTFSize = 3;
	float4 colorTransferFunc[] = {
		{ 0, 0.0, 0.0, 1.0 },
		{ 0.5, 0.0, 1.0, 0.0 }, 
		{ 1.0, 1.0, 0.0, 0.0 },
	};
	int opacityTFSize = 16;
	float2 opacityTransferFunc[] = {
		{0, 0},
		{0.1, 0.0},
		{0.11, 0.1},
		{0.14, 0.1},
		{0.15, 0.0},
		{0.21, 0.0},
		{0.22, 0.2},
		{0.25, 0.2},
		{0.26, 0.0},
		{0.37, 0.0},
		{0.38, 0.7},
		{0.41, 0.7},
		{0.42, 0.0},
		{0.73, 0.0},
		{0.74, 1.0},
		{1.0, 1.0},
	};
	getTransferFunc(transferFunc,
					colorTFSize,
					colorTransferFunc,
					opacityTFSize,
					opacityTransferFunc);
#endif

#if 0 // ML dataset, multiple isosurface, used
	float4 transferFunc[101];
	int colorTFSize = 4;
	float4 colorTransferFunc[] = {
		{ 0, 0.0, 0.0, 1.0 },
		{ 0.1, 0.0, 1.0, 0.0 }, 
		{ 0.7, 1.0, 0.0, 0.0 },
		{ 1.0, 1.0, 0.0, 0.0 },
	};
	int opacityTFSize = 15;
	float level_1 = 0.011;
	float level_2 = 0.09;
	float level_3 = 0.35;
	float level_4 = 0.82;

	float2 opacityTransferFunc[] = {
		{0, 0.00},

		{level_1 - 0.01, 0.0},
		{level_1, 0.02},
		{level_1 + 0.01, 0.0},
	
		{level_2 - 0.01, 0.0},
		{level_2, 0.01},
		{level_2 + 0.01, 0.0},

		// {0.15 - 0.01, 0.0},
		// {0.15, 0.01},
		// {0.15 + 0.01, 0.0},

		{level_3 - 0.01, 0.0},
		{level_3, 0.001},
		{level_3 + 0.01, 0.0},

		// {0.5 - 0.01, 0.0},
		// {0.5, 0.01},
		// {0.5 + 0.01, 0.0},

		
		{level_4 - 0.01, 0.0},
		{level_4, 0.0005},
		{level_4 + 0.01, 0.0},
		
		{0.999, 0.0},
		{1.0, 0.0},
	};
	getTransferFunc(transferFunc,
					colorTFSize,
					colorTransferFunc,
					opacityTFSize,
					opacityTransferFunc);
#endif

#if 0 // ML dataset, solid single isosurface
	float4 transferFunc[101];
	int colorTFSize = 3;
	float4 colorTransferFunc[] = {
		{ 0, 0.0, 1.0, 0.0 },
		{ 0.5, 0.0, 1.0, 0.0 }, 
		{ 1.0, 0.0, 1.0, 0.0 },
	};
	int opacityTFSize = 4;
	float2 opacityTransferFunc[] = {
		{0, 0},
		{0.4999999999999999999999999999999, 0},
		{0.5, 1.0},
		{1.0, 1.0},
	};
	getTransferFunc(transferFunc,
					colorTFSize,
					colorTransferFunc,
					opacityTFSize,
					opacityTransferFunc);
#endif

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray *d_transferFuncArray;
    checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1));
    checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc2));
}

extern "C"
void freeCudaBuffers()
{
    checkCudaErrors(cudaFreeArray(d_volumeArray));
    checkCudaErrors(cudaFreeArray(d_transferFuncArray));
}


extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                   float density, float brightness, float transferOffset, float transferScale, float sampleDistance, int size_x, int size_y, int size_z, int model)
{
	if (model == 0) { // Downsampling method
	    d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, density,
    	                                  brightness, transferOffset, transferScale, selectListKernel, visibleBlocksListKernel, visibleBlocksSizeKernel,
										  X_BLOCK_NUM_LEVEL_1, Y_BLOCK_NUM_LEVEL_1, Z_BLOCK_NUM_LEVEL_1,
										  X_BLOCK_NUM_LEVEL_2, Y_BLOCK_NUM_LEVEL_2, Z_BLOCK_NUM_LEVEL_2,
										  X_BLOCK_NUM_LEVEL_3, Y_BLOCK_NUM_LEVEL_3, Z_BLOCK_NUM_LEVEL_3,
										  X_BLOCK_NUM_LEVEL_4, Y_BLOCK_NUM_LEVEL_4, Z_BLOCK_NUM_LEVEL_4,
										  selectBlockSizeKernel,
										  sampleDistance, 
										  size_x, size_y, size_z,
										  knotDimensionSampleSizeKernel, ctrlptsDimensionSampleSizeKernel,
									  	  cacheKnotDataKernel, cacheCtrlptsDataKernel);
	} else {
		d_render_mfa_adaptive<<<gridSize, blockSize>>>(d_output, imageW, imageH, density,
        		                              brightness, transferOffset, transferScale, selectListKernel, visibleBlocksListKernel, visibleBlocksSizeKernel,
											  X_BLOCK_NUM_LEVEL_1, Y_BLOCK_NUM_LEVEL_1, Z_BLOCK_NUM_LEVEL_1,
											  X_BLOCK_NUM_LEVEL_2, Y_BLOCK_NUM_LEVEL_2, Z_BLOCK_NUM_LEVEL_2,
											  X_BLOCK_NUM_LEVEL_3, Y_BLOCK_NUM_LEVEL_3, Z_BLOCK_NUM_LEVEL_3,
											  X_BLOCK_NUM_LEVEL_4, Y_BLOCK_NUM_LEVEL_4, Z_BLOCK_NUM_LEVEL_4,
									  		  selectBlockSizeKernel,
									  		  sampleDistance, 
									  		  size_x, size_y, size_z,
									  		  knotDimensionSampleSizeKernel, ctrlptsDimensionSampleSizeKernel,
									  		  cacheKnotDataKernel, cacheCtrlptsDataKernel, degreeKernel, cacheKnotSizeListKernel);
		#if 0 // render ground truth result of ML dataset for quatititive evaluation
		d_render_ml_gt<<<gridSize, blockSize>>>(d_output, imageW, imageH, density,
    	                                  brightness, transferOffset, transferScale, selectListKernel, visibleBlocksListKernel, visibleBlocksSizeKernel,
										  X_BLOCK_NUM_LEVEL_1, Y_BLOCK_NUM_LEVEL_1, Z_BLOCK_NUM_LEVEL_1,
										  X_BLOCK_NUM_LEVEL_2, Y_BLOCK_NUM_LEVEL_2, Z_BLOCK_NUM_LEVEL_2,
										  X_BLOCK_NUM_LEVEL_3, Y_BLOCK_NUM_LEVEL_3, Z_BLOCK_NUM_LEVEL_3,
										  X_BLOCK_NUM_LEVEL_4, Y_BLOCK_NUM_LEVEL_4, Z_BLOCK_NUM_LEVEL_4,
										  selectBlockSizeKernel,
										  sampleDistance, 
										  size_x, size_y, size_z,
										  knotDimensionSampleSizeKernel, ctrlptsDimensionSampleSizeKernel,
									  	  cacheKnotDataKernel, cacheCtrlptsDataKernel);
		#endif
	}
	cudaDeviceSynchronize();

}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
