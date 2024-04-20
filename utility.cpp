#include "utility.hpp"
#include <iostream>
#include <chrono>
#include <math.h>
#include <assert.h>

// Load raw data from disk
void *loadRawFile(char *filename, size_t size) {
    // printf("filename %s\n", filename);
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    return data;
}

// Load .mfab files from disk
void loadMfabFile_org(char *filename, int knot_block_size, int ctrlpts_block_size, void *knot_block, void *ctrlpts_block) {
    // printf("filename %s\n", filename);
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        exit(EXIT_SUCCESS);
    }

	unsigned int degree;
    fread(&degree, 1, 4, fp);
	// std::cout << "degree: " << degree << std::endl;
	unsigned int knot_size;
    fread(&knot_size, 1, 4, fp);
	// std::cout << "knot_size: " << knot_size << std::endl;
	fread(knot_block, 1, knot_size*3*4, fp);
	// std::cout << "knots[0]: " << ((float*)knot_block)[0] << std::endl;
	// std::cout << "knots[34]: " << ((float*)knot_block)[34] << std::endl;
	// std::cout << "knots[79*3 - 1]: " << ((float*)knot_block)[79*3 - 1] << std::endl;
	fread(ctrlpts_block, 1, (knot_size - 1 - degree)*(knot_size - 1 - degree)*(knot_size - 1 - degree)*4, fp);
	// std::cout << "ctrlpts_block[0]: " << ((float*)ctrlpts_block)[0] << std::endl;
	// std::cout << "ctrlpts_block[100000]: " << ((float*)ctrlpts_block)[100000] << std::endl;
	// std::cout << "ctrlpts_block[-1]: " << ((float*)ctrlpts_block)[76*76*76 - 1] << std::endl;

    fclose(fp);
}

// Load .mfab files from disk
int loadMfabFile(char *filename, void *knot_block, void *ctrlpts_block) {
    // printf("filename %s\n", filename);
    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        exit(EXIT_SUCCESS);
    }

	unsigned int degree;
    fread(&degree, 1, 4, fp);
	// std::cout << "degree: " << degree << std::endl;
	unsigned int knot_size;
    fread(&knot_size, 1, 4, fp);
	// std::cout << "knot_size: " << knot_size << std::endl;
	fread(knot_block, 1, knot_size*3*4, fp);
	// std::cout << "knots[0]: " << ((float*)knot_block)[0] << std::endl;
	// std::cout << "knots[34]: " << ((float*)knot_block)[34] << std::endl;
	// std::cout << "knots[79*3 - 1]: " << ((float*)knot_block)[79*3 - 1] << std::endl;
	fread(ctrlpts_block, 1, (knot_size - 1 - degree)*(knot_size - 1 - degree)*(knot_size - 1 - degree)*4, fp);
	// std::cout << "ctrlpts_block[0]: " << ((float*)ctrlpts_block)[0] << std::endl;
	// std::cout << "ctrlpts_block[100000]: " << ((float*)ctrlpts_block)[100000] << std::endl;
	// std::cout << "ctrlpts_block[-1]: " << ((float*)ctrlpts_block)[76*76*76 - 1] << std::endl;

    fclose(fp);
	return knot_size;
}

int isVisible(float x, float y, float z, float dx, float dy, float dz, float angle) {
	// testing position T: x, y, z; camera position C: dx, dy, dz; volume center position O: 0, 0, 0
	// CT: (x - dx), (y - dy), (z - dz)
	// CO: (0 - dx), (0 - dy), (0 - dz)
	// Handle when testing position is on volume center
	if (x == 0 && y == 0 && z == 0) {
		// The angle is 0 degree, should alway count
		return 1;
	}
	float CT_norm = sqrt((x - dx)*(x - dx) + (y - dy)*(y - dy) + (z - dz)*(z - dz));
	float CO_norm = sqrt(dx*dx + dy*dy + dz*dz);
	float CT_dot_CO = (x - dx)*(-dx) + (y - dy)*(-dy) + (z - dz)*(-dz); 
	if (CT_norm == 0) {
		return 1;
	}
	assert(CT_norm != 0);
	assert(CO_norm != 0);

	// std::cout << "acos argument: " << CT_dot_CO/CT_norm/CO_norm << std::endl;
	float result;
	if (CT_dot_CO/CT_norm/CO_norm > 1) {
		result = 0.0;
		// result = acos(1.0) * 180.0 / M_PI; // 0.0 degree
	} else if (CT_dot_CO/CT_norm/CO_norm < -1) {
		result = 180.0;
		// result = acos(-1.0) * 180.0 / M_PI; // 180.0 degree
	} else {
		result = acos(CT_dot_CO/CT_norm/CO_norm) * 180.0 / M_PI;
	}
	if (isnan(result)) {
		// std::cout << "CT_dot_CO: " << CT_dot_CO << std::endl;
		// std::cout << "CT_norm: " << CT_norm << std::endl;
		// std::cout << "CO_norm: " << CO_norm << std::endl;
		// std::cout << "x, y, z: " << x << ", " << y << ", " << z << std::endl;
		// std::cout << "dx, dy, dz: " << dx << ", " << dy << ", " << dz << std::endl;
		std::cout << "get nan-------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        exit(EXIT_SUCCESS);
	}
	if (result <= angle) {
		return 1;
	}
	else {
		return 0;
	}
}

bool isPresent(std::vector<int> cache, int index, int &hit_position) {

	for (int i = 0; i < cache.size(); i++) {
		if (cache.at(i) == index) {
			hit_position = i;
			return true;
		}
	}
	return false;
}

void tpd2xyz(float theta, float phi, float dis, float &x, float &y, float &z) {
	float cos_theta = cos ( theta * M_PI / 180.0 );
	float sin_theta = sin ( theta * M_PI / 180.0 );
	float cos_phi   = cos ( phi   * M_PI / 180.0 );
	float sin_phi   = sin ( phi   * M_PI / 180.0 );
	x = dis * sin_theta * cos_phi;
	y = dis * sin_theta * sin_phi;
	z = dis * cos_theta;
}

void xyz2tpd(float x, float y, float z, float &theta, float &phi, float &dis) {
    dis = sqrt(x*x + y*y + z*z);
    theta = atan(sqrt(x*x + y*y)/z)/M_PI*180.0;
    phi = atan(y/x)/M_PI*180.0;
        
    if (dis >= 2.0f){
        dis= 2.0f;
    }else if (dis <= 1.0f){
        dis= 1.0f;
    }


    if (theta < 0) {
        theta = 180 + theta;
    }
    if (phi > 0) {
        if (x > 0) {
            phi = phi;
        } else {
            phi = 180 + phi;
        }
    } else {
        if (x < 0) {
            phi = 180 + phi;
        } else {
            phi = 360 + phi;
        }
    }
}

float getDistance(float x_center, float y_center, float z_center, float camera_x, float camera_y, float camera_z) {
	float x_distance = camera_x - x_center;
	float y_distance = camera_y - y_center;
	float z_distance = camera_z - z_center;
	return sqrt(x_distance*x_distance + y_distance*y_distance + z_distance*z_distance);
}

float getDistanceFromBlock(int block_index,
						   float camera_x, float camera_y, float camera_z,
						   int block_num,
						   float unit) {
	int z = block_index/(block_num*block_num);
	int remain = block_index%(block_num*block_num);
	int y = remain/block_num;
	int x = remain%block_num;

	float x_lower = -1 + x*unit;
	float y_lower = -1 + y*unit;
	float z_lower = -1 + z*unit;
	float x_upper = x_lower + unit;
	float y_upper = y_lower + unit;
	float z_upper = z_lower + unit;
				
	return getDistance((x_lower + x_upper)/2, (y_lower + y_upper)/2, (z_lower + z_upper)/2, camera_x, camera_y, camera_z);
}

void getCurrentVisibleBlocksFast(float theta, float phi, float dis,
							 	 int volume_size_1, int volume_size_2, int volume_size_3, int volume_size_4,
							     int block_size,
							 	 float angle,
							 	 std::vector<int> &visible_blocks,
								 // std::map<int, std::vector<std::vector<int>>> &cornerBlockMap) {
								 int *cornerBlockMapArray,
								 int *blockCheck) {
	// translate from Spherical Coordinate System to Cartesian Coordinate System for the camera location
	float cos_theta = cos ( theta * M_PI / 180.0 );
	float sin_theta = sin ( theta * M_PI / 180.0 );
	float cos_phi   = cos ( phi   * M_PI / 180.0 );
	float sin_phi   = sin ( phi   * M_PI / 180.0 );
	float camera_x = dis * sin_theta * cos_phi;
	float camera_y = dis * sin_theta * sin_phi;
	float camera_z = dis * cos_theta;

	int block_num_1 = (volume_size_1 - 1)/(block_size - 2); // 2
	int block_num_2 = (volume_size_2 - 1)/(block_size - 2); // 4
	int block_num_3 = (volume_size_3 - 1)/(block_size - 2); // 8
	int block_num_4 = (volume_size_4 - 1)/(block_size - 2); // 16

	int all_block_num_1 = block_num_1*block_num_1*block_num_1; // 8
	int all_block_num_2 = block_num_2*block_num_2*block_num_2; // 64
	int all_block_num_3 = block_num_3*block_num_3*block_num_3; // 512
	int all_block_num_4 = block_num_4*block_num_4*block_num_4; // 4096

	float unit_1 = 2.0/(float)block_num_1;
	float unit_2 = 2.0/(float)block_num_2;
	float unit_3 = 2.0/(float)block_num_3;
	float unit_4 = 2.0/(float)block_num_4;

	float x_start = -1.0;
	float y_start = -1.0;
	float z_start = -1.0;


	int sample_num = block_num_4 + 1;  // 17
	std::vector<int> level_1_blocks;
	std::vector<int> level_2_blocks;
	std::vector<int> level_3_blocks;
	std::vector<int> level_4_blocks;

	std::vector<int> selected_blocks_level_1;
	std::vector<int> selected_blocks_level_2;
	std::vector<int> selected_blocks_level_3;
	std::vector<int> selected_blocks_level_4;

	std::chrono::high_resolution_clock::time_point t_start, t_end;
	t_start = std::chrono::high_resolution_clock::now();

	memset(blockCheck, 0, sizeof(int)*(8 + 64 + 512 + 4096));

	// for (int i = 0; i < cornerBlockMap.size(); i++) {
	for (int i = 0; i < 17*17*17; i++) {
		int z = i/(sample_num*sample_num); // 1/(17*17) integer portion
		int remain = i%(sample_num*sample_num); // 1%(17*17) remainding portion
		int y = remain/(sample_num); 
		int x = remain%(sample_num);

		float x_corner = x_start + x*unit_4;
		float y_corner = y_start + y*unit_4;
		float z_corner = z_start + z*unit_4;
		
		int b = isVisible(x_corner, y_corner, z_corner, camera_x, camera_y, camera_z, angle/2.0);
	
		/* find all visible blocks for all levels */
		int hit_position;
		if (b == 1) {
			/* Level 1 */
			int begin = i*4*9;
			int begin_level_1 = begin;
			int begin_level_2 = begin + 9;
			int begin_level_3 = begin + 9 + 9;
			int begin_level_4 = begin + 9 + 9 + 9;
			for (int j = 0; j < cornerBlockMapArray[begin_level_1]; j++) {
				if (blockCheck[cornerBlockMapArray[begin_level_1 + j + 1]] == 0)
				//if (!isPresent(level_1_blocks, cornerBlockMapArray[begin_level_1 + j + 1], hit_position)) {
					level_1_blocks.push_back(cornerBlockMapArray[begin_level_1 + j + 1]);
					blockCheck[cornerBlockMapArray[begin_level_1 + j + 1]] = 1;
				//}
			}
			/* Level 2 */
			for (int j = 0; j < cornerBlockMapArray[begin_level_2]; j++) {
				if (blockCheck[cornerBlockMapArray[begin_level_2 + j + 1]] == 0)
				//if (!isPresent(level_2_blocks, cornerBlockMapArray[begin_level_2 + j + 1], hit_position)) {
					level_2_blocks.push_back(cornerBlockMapArray[begin_level_2 + j + 1]);
					blockCheck[cornerBlockMapArray[begin_level_2 + j + 1]] = 1;
				//}
			}
			/* Level 3 */
			for (int j = 0; j < cornerBlockMapArray[begin_level_3]; j++) {
				if (blockCheck[cornerBlockMapArray[begin_level_3 + j + 1]] == 0)
				//if (!isPresent(level_3_blocks, cornerBlockMapArray[begin_level_3 + j + 1], hit_position)) {
					level_3_blocks.push_back(cornerBlockMapArray[begin_level_3 + j + 1]);
					blockCheck[cornerBlockMapArray[begin_level_3 + j + 1]] = 1;
				//}
			}
			/* Level 4 */
			for (int j = 0; j < cornerBlockMapArray[begin_level_4]; j++) {
				if (blockCheck[cornerBlockMapArray[begin_level_4 + j + 1]] == 0)
				//if (!isPresent(level_4_blocks, cornerBlockMapArray[begin_level_4 + j + 1], hit_position)) {
					level_4_blocks.push_back(cornerBlockMapArray[begin_level_4 + j + 1]);
					blockCheck[cornerBlockMapArray[begin_level_4 + j + 1]] = 1;
				//}
			}
		}

	}

	t_end = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()/(float)1000000;


	for (int i = 0; i < level_1_blocks.size(); i++) {
		int block_index = level_1_blocks.at(i) - 0;
		int z = block_index/(block_num_1*block_num_1);
		int remain = block_index%(block_num_1*block_num_1);
		int y = remain/block_num_1;
		int x = remain%block_num_1;

		float x_lower = x_start + x*unit_1;
		float y_lower = y_start + y*unit_1;
		float z_lower = z_start + z*unit_1;
		float x_upper = x_lower + unit_1;
		float y_upper = y_lower + unit_1;
		float z_upper = z_lower + unit_1;
				
		float distance = getDistance((x_lower + x_upper)/2, (y_lower + y_upper)/2, (z_lower + z_upper)/2, camera_x, camera_y, camera_z);
		if (distance >= 2.4) {
			selected_blocks_level_1.push_back(block_index);
			visible_blocks.push_back(block_index);
		}
	}
	for (int i = 0; i < level_2_blocks.size(); i++) {
		int block_index = level_2_blocks.at(i) - 8;
		int z = block_index/(block_num_2*block_num_2);
		int remain = block_index%(block_num_2*block_num_2);
		int y = remain/block_num_2;
		int x = remain%block_num_2;

		float x_lower = x_start + x*unit_2;
		float y_lower = y_start + y*unit_2;
		float z_lower = z_start + z*unit_2;
		float x_upper = x_lower + unit_2;
		float y_upper = y_lower + unit_2;
		float z_upper = z_lower + unit_2;
				
		float distance = getDistance((x_lower + x_upper)/2, (y_lower + y_upper)/2, (z_lower + z_upper)/2, camera_x, camera_y, camera_z);
	
		// skip current level 2 block if it is in the selected level 1 blocks
		int i_x_up_level_1 = x/2; // 0~2
		int i_y_up_level_1 = y/2;
		int i_z_up_level_1 = z/2;
		int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
		bool found_level_1 = false;
		for (int k = 0; k < selected_blocks_level_1.size(); k++) {
			if (index_up_level_1 == selected_blocks_level_1.at(k)) {
				found_level_1 = true;
				break;
			}
		}
		if (!found_level_1 && distance >= 1.6) {
			selected_blocks_level_2.push_back(block_index);
			visible_blocks.push_back(block_index + 8);
		}
	}
	for (int i = 0; i < level_3_blocks.size(); i++) {
		int block_index = level_3_blocks.at(i) - 8 - 64;
		int z = block_index/(block_num_3*block_num_3);
		int remain = block_index%(block_num_3*block_num_3);
		int y = remain/block_num_3;
		int x = remain%block_num_3;

		float x_lower = x_start + x*unit_3;
		float y_lower = y_start + y*unit_3;
		float z_lower = z_start + z*unit_3;
		float x_upper = x_lower + unit_3;
		float y_upper = y_lower + unit_3;
		float z_upper = z_lower + unit_3;
				
		float distance = getDistance((x_lower + x_upper)/2, (y_lower + y_upper)/2, (z_lower + z_upper)/2, camera_x, camera_y, camera_z);
		// skip current level 3 block if it is in the selected level 1 blocks
		int i_x_up_level_1 = x/4; // 0~2
		int i_y_up_level_1 = y/4;
		int i_z_up_level_1 = z/4;
		int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
		bool found_level_1 = false;
		for (int k = 0; k < selected_blocks_level_1.size(); k++) {
			if (index_up_level_1 == selected_blocks_level_1.at(k)) {
				found_level_1 = true;
				break;
			}
		}
		// skip current level 3 block if it is in the selected level 2 blocks
		int i_x_up_level_2 = x/2; // 0~4
		int i_y_up_level_2 = y/2;
		int i_z_up_level_2 = z/2;
		int index_up_level_2 = i_z_up_level_2*4*4 + i_y_up_level_2*4 + i_x_up_level_2;
		bool found_level_2 = false;
		for (int k = 0; k < selected_blocks_level_2.size(); k++) {
			if (index_up_level_2 == selected_blocks_level_2.at(k)) {			
				found_level_2 = true;
				break;
			}
		}
		if (!found_level_1 && !found_level_2 && distance >= 0.8) {
			selected_blocks_level_3.push_back(block_index);
			visible_blocks.push_back(block_index + 8 + 64);
		}
	}
	for (int i = 0; i < level_4_blocks.size(); i++) {
		int block_index = level_4_blocks.at(i) - 8 - 64 - 512;
		int z = block_index/(block_num_4*block_num_4);
		int remain = block_index%(block_num_4*block_num_4);
		int y = remain/block_num_4;
		int x = remain%block_num_4;

		float x_lower = x_start + x*unit_4;
		float y_lower = y_start + y*unit_4;
		float z_lower = z_start + z*unit_4;
		float x_upper = x_lower + unit_4;
		float y_upper = y_lower + unit_4;
		float z_upper = z_lower + unit_4;
				
		float distance = getDistance((x_lower + x_upper)/2, (y_lower + y_upper)/2, (z_lower + z_upper)/2, camera_x, camera_y, camera_z);
		// skip current level 3 block if it is in the selected level 1 blocks
		int i_x_up_level_1 = x/8; // 0~2
		int i_y_up_level_1 = y/8;
		int i_z_up_level_1 = z/8;
		int index_up_level_1 = i_z_up_level_1*2*2 + i_y_up_level_1*2 + i_x_up_level_1;
		bool found_level_1 = false;
		for (int k = 0; k < selected_blocks_level_1.size(); k++) {
			if (index_up_level_1 == selected_blocks_level_1.at(k)) {
				found_level_1 = true;
				break;
			}
		}
		// skip current level 3 block if it is in the selected level 2 blocks
		int i_x_up_level_2 = x/4; // 0~4
		int i_y_up_level_2 = y/4;
		int i_z_up_level_2 = z/4;
		int index_up_level_2 = i_z_up_level_2*4*4 + i_y_up_level_2*4 + i_x_up_level_2;
		bool found_level_2 = false;
		for (int k = 0; k < selected_blocks_level_2.size(); k++) {
			if (index_up_level_2 == selected_blocks_level_2.at(k)) {
				found_level_2 = true;
				break;
			}
		}
		// skip current level 3 block if it is in the selected level 3 blocks
		int i_x_up_level_3 = x/2; // 0~8
		int i_y_up_level_3 = y/2;
		int i_z_up_level_3 = z/2;
		int index_up_level_3 = i_z_up_level_3*8*8 + i_y_up_level_3*8 + i_x_up_level_3;
		bool found_level_3 = false;
		for (int k = 0; k < selected_blocks_level_3.size(); k++) {
			if (index_up_level_3 == selected_blocks_level_3.at(k)) {
				found_level_3 = true;
				break;
			}
		}
		if (!found_level_1 && !found_level_2 && !found_level_3) {
			selected_blocks_level_4.push_back(block_index);
			visible_blocks.push_back(block_index + 8 + 64 + 512);
		}
	}
}

int updateCacheAndCacheData(std::vector<int> &cache, std::vector<int> &cache_mem, void *cache_data, int visible_block_index, cudaExtent microBlockSize, std::string method, const char *volumeFilePath, int camera_index) {
	size_t block_size = microBlockSize.width*microBlockSize.height*microBlockSize.depth*sizeof(VolumeType);
	int hit_miss = -1; // hit: 1; miss: 0
	if (method == "lru") {
		int hit_position = 0;
		if (cache.size() == 0) { // miss and cache is empty
			hit_miss = 0;
			cache.push_back(visible_block_index);
			cache_mem.push_back(visible_block_index);

			std::string block_file_name = volumeFilePath + std::to_string(visible_block_index) + ".blk";
    		void *block = loadRawFile(const_cast<char*>(block_file_name.c_str()), block_size);
			memcpy(cache_data, block, block_size);

		} else if (isPresent(cache, visible_block_index, hit_position)) { // hit
			hit_miss = 1;
			int temp = cache.at(hit_position);
			cache.erase(cache.begin() + hit_position);
			cache.push_back(temp);
		} else if (cache.size() < CACHE_SIZE) { // miss and cache is not full, do appending
			hit_miss = 0;
			cache.push_back(visible_block_index);

			int write_position = cache_mem.size();
			cache_mem.push_back(visible_block_index);
			std::string block_file_name = volumeFilePath + std::to_string(visible_block_index) + ".blk";
    		void *block = loadRawFile(const_cast<char*>(block_file_name.c_str()), block_size);
			memcpy(cache_data + block_size*write_position, block, block_size);

		} else { // miss and cache is full, do replacement
			hit_miss = 0;
			int lru_index = cache.at(0);
			cache.erase(cache.begin());
			cache.push_back(visible_block_index);

			int replace_position;
			if(isPresent(cache_mem, lru_index, replace_position)) {
				cache_mem.at(replace_position) = visible_block_index;
				std::string block_file_name = volumeFilePath + std::to_string(visible_block_index) + ".blk";
				void *block = loadRawFile(const_cast<char*>(block_file_name.c_str()), block_size);
				memcpy(cache_data + block_size*replace_position, block, block_size);
			} else {
				std::cout << "Cannot find lru element in data_mem." << std::endl;
				exit(EXIT_FAILURE);
			}
		}
	}
	return hit_miss;
}

int updateCacheAndCacheDataMfa(std::vector<int> &cache,
							   std::vector<int> &cache_mem,
							   void *cache_knot_data,
							   void *cache_ctrlpts_data,
							   int visible_block_index,
							   int knot_block_size,
							   int ctrlpts_block_size,
							   std::string method,
							   const char *mfaFilePath,
							   std::vector<int> &cache_knot_size_list) {
	int hit_miss = -1; // hit: 1; miss: 0
	std::string block_file_name = mfaFilePath + std::to_string(visible_block_index) + ".mfab";
	if (method == "lru") {
		int hit_position = 0;
		if (cache.size() == 0) { // miss and cache is empty
			hit_miss = 0;
			cache.push_back(visible_block_index);
			cache_mem.push_back(visible_block_index);
			int knot_size = loadMfabFile(const_cast<char*>(block_file_name.c_str()),
						 				 cache_knot_data,
						 				 cache_ctrlpts_data);
			cache_knot_size_list.push_back(knot_size);
		} else if (isPresent(cache, visible_block_index, hit_position)) { // hit
			hit_miss = 1;
			int temp = cache.at(hit_position);
			cache.erase(cache.begin() + hit_position);
			cache.push_back(temp);
		} else if (cache.size() < CACHE_SIZE) { // miss and cache is not full, do appending
			hit_miss = 0;
			cache.push_back(visible_block_index);
			int write_position = cache_mem.size();
			cache_mem.push_back(visible_block_index);
			int knot_size = loadMfabFile(const_cast<char*>(block_file_name.c_str()),
										 cache_knot_data + knot_block_size*write_position,
										 cache_ctrlpts_data + ctrlpts_block_size*write_position);
			cache_knot_size_list.push_back(knot_size);
		} else { // miss and cache is full, do replacement
			hit_miss = 0;
			int lru_index = cache.at(0);
			cache.erase(cache.begin());
			cache.push_back(visible_block_index);

			int replace_position;
			if(isPresent(cache_mem, lru_index, replace_position)) {
				cache_mem.at(replace_position) = visible_block_index;
				int knot_size = loadMfabFile(const_cast<char*>(block_file_name.c_str()),
										     cache_knot_data + knot_block_size*replace_position,
										     cache_ctrlpts_data + ctrlpts_block_size*replace_position);
				cache_knot_size_list.at(replace_position) = knot_size;
			} else {
				std::cout << "Cannot find lru element in data_mem." << std::endl;
				exit(EXIT_FAILURE);
			}
		}
	}
	return hit_miss;
}
