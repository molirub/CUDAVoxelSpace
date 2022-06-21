#pragma once
#define N_MAX 1078
#define BMP_HEADER_SIZE 1078

#define IM_WIDTH 1024
#define IM_HEIGHT 1024

typedef struct camera_s
{
	float x;        // 512 // x position on the map
	float y;        // 800 // y position on the map
	float height;   // 78 // height of the camera
	float angle;    // 0 // direction of the camera
	float horizon;  // 100 // horizon position (look up and down)
	float distance; // 800   // distance of map
} camera_t;


int write_bmp(const char* filename, int width, int height, char* rgb);
void readBMP_RGB(char* filename, char** data_rgb, int* width_rgb, int* height_rgb);
void readBMP_HM(char* filename, unsigned char** valor_alturas, int* width_alturas, int* height_alturas);
void voxel_space(camera_t camera, int w, int h, char* rgb, int* w_colores, int* h_colores, char* rgb_colores, int* w_alturas, int* h_alturas, unsigned char* valor_alturas);
void obtain_voxel_bmp(int width, int height, BYTE* bmp_array, camera_t camera);
void init_voxel_maps(void);