//Basado en el siguiente proyecto:
//https://github.com/s-macke/VoxelSpace

//Pablo Esteve Reula
//Rubén Molinete Silván

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include <math.h> 
#include <windows.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include "generate_image.h"


int width_colores, height_colores;
unsigned char* rgb_colores;
int width_alturas, height_alturas;
unsigned char* valor_alturas;

BOOL LoadDll(HMODULE* hMod, char* dll_path) {
	if (*hMod != NULL) return false;
	*hMod = LoadLibrary(dll_path);
	if (*hMod == NULL) return false;
	return true;
}

BOOL UnloadDll(HMODULE* hMod) {
	if (*hMod == NULL) return false;
	FreeLibrary(*hMod);
	*hMod = NULL;
	return true;
}
HMODULE cuda_hMod = NULL;
char cudaPath[] = "..\\build\\dll\\CudaKernel.dll";
// Function definition in the DLL
typedef int (*func_type_cuda_dll)(int img_width, int img_height, int map_width, int map_height, camera_t camera, unsigned char* rgb_colormap, unsigned char* heightmap, unsigned char* rgb_result);
func_type_cuda_dll generate_voxel_image = NULL;


//void __declspec (dllimport) generate_voxel_image_(int img_width, int img_height, int map_width, int map_height, camera_t camera, unsigned char* rgb_colormap, unsigned char* heightmap, unsigned char* rgb_result);
// load cuda dll

int init_load_dll(void)
{
	char msg[1000];
	int ret;
	if (!LoadDll(&cuda_hMod, cudaPath))
	{
		sprintf_s(msg, "Error loading '%s', err: %d\r\n", cudaPath, GetLastError());
	}
	else 
	{
		sprintf_s(msg, "Successfully loaded '%s'\r\n", cudaPath);
	}
	// call dll functions
	generate_voxel_image = (func_type_cuda_dll)GetProcAddress(cuda_hMod, "generate_voxel_image");
	if (generate_voxel_image == NULL) {
		sprintf_s(msg, "Function 'generate_voxel_image' not found, err: %d\r\n", GetLastError());
	}
	/*// unload cuda dll
if(!UnloadDll(&cuda_hMod)) {
	sprintf_s(msg, "Error unloading '%s', err: %d\r\n", cudaPath, GetLastError());
} else {
	sprintf_s(msg, "Successfully unloaded '%s'\r\n", cudaPath);
}*/

	return 0;
}

int create_bmp(int width, int height, char* rgb, char* bmp_array, int size);

void init_voxel_maps(void)
{
	//Leemos BMP del mapa de colores y de alturas
	//Mapa de colores 1024x1024 24 bit color a RGB bottom to top
	char filename_colores[] = "../maps/C1W.bmp";
	readBMP_RGB(filename_colores, &rgb_colores, &width_colores, &height_colores);

	//Mapa de alturas 1024x1024 8 bit height map bottom to top
	char filename_alturas[] = "../maps/D1.bmp";
	readBMP_HM(filename_alturas, &valor_alturas, &width_alturas, &height_alturas);
}

void obtain_voxel_bmp(int width, int height, BYTE* bmp_array, camera_t camera)
{
	///// CPU 
	voxel_space(camera, width, height, (char*)bmp_array, &width_colores, &height_colores, rgb_colores, &width_alturas, &height_alturas, valor_alturas);
}

void obtain_voxel_bmp_cuda(int width, int height, BYTE* bmp_array, camera_t camera)
{
	///// GPU 
	//int img_width, int img_height, int map_width, int map_height, camera_t camera, unsigned char* rgb_colormap, unsigned char* heightmap, unsigned char* rgb_result
	generate_voxel_image(width, height, 1024, 1024, camera, rgb_colores, valor_alturas, (unsigned char*)bmp_array);
}

void voxel_space(camera_t camera, int w, int h, char* rgb, int* w_colores, int* h_colores, unsigned char* rgb_colores, int* w_alturas, int* h_alturas, unsigned char* valor_alturas) {

	//Posición cámara en 1024x1024
	//float p[2] = { 512.0f, 512.0f };
	float p[2] = { camera.x, camera.y };

	//Altura cámara en bloques
	//float height = 150.0f;
	float height = camera.height;

	//Altura del horizonte con respecto a IM_HEIGHT
	//float horizon = (float)IM_HEIGHT / 2.0f;
	float horizon = camera.horizon;

	//Escalado de altura (cuanto más pequeño más exagerada la altura)
	float scale_height = 0.0015f;

	//Distancia máxima de renderizado en pixeles
	//float distance = 4000.0f;
	float distance = camera.distance;

	//Ángulo de la cámara en grados
	//float phi = 0.0f;
	//phi = phi * 180.0f / 3.1415f;
	float phi = camera.angle;
	//phi = phi * 180.0f / 3.1415f;
	////////////////////////////////////////////

	//Inicializar vector rgb a gris para el cielo
	for (int i = 0; i < IM_HEIGHT * IM_WIDTH * 3; i+=3)
	{
		rgb[i] = 148;
		rgb[i+1] = 209;
		rgb[i+2] = 239;
	}

	//Precalculamos los parámetros del ángulo de visión
	float sinphi = sin(phi);
	float cosphi = cos(phi);

	//Inicializamos el vector de visibilidad para cada columna
	int ybuffer[IM_WIDTH];
	for (int i = 0; i < IM_WIDTH; i++)
	{
		ybuffer[i] = (int)IM_HEIGHT;
	}

	//Generación de cada una de la líneas de delante a detrás
	float z = 1.0f;
	float dz = 1.00f;
	while (z <= distance )
	{
		//Cálculo de los extremos de las líneas para FOV 90º
		float pleft[2] = { (-cosphi * z - sinphi * z) + p[0],
							(sinphi * z - cosphi * z) + p[1] };
		float pright[2] = { (cosphi * z - sinphi * z) + p[0],
							(-sinphi * z - cosphi * z) + p[1] };

		//Segmentación líneas
		float dx = (pright[0] - pleft[0]) / (float)IM_WIDTH;
		float dy = (pright[1] - pleft[1]) / (float)IM_WIDTH;

		// Truncado

		//Para cada pixel de la línea en la pantalla
		for (int i = 0; i < IM_WIDTH; i++)
		{
			//Sacamos altura del pixel a dibujar
			//int offset_alturas = (int) (pleft[0] + 1024 * (int)pleft[1]);

			int offset_alturas = (((int)floor(pleft[1]) & 1023) * 1024) + ((int)floor(pleft[0]) & 1023);
			float height_on_screen = (((height - valor_alturas[offset_alturas]) / (z * scale_height)) + horizon);

			//Imprimimos colores hasta llegar al límite del buffer
			if (height_on_screen < 0) height_on_screen = 0;
			if (height_on_screen > IM_HEIGHT) height_on_screen = IM_HEIGHT;
			for (int j = (int)floor(height_on_screen); j < (int)floor(ybuffer[i]); j++)
			{
				int index_rgb = ((IM_WIDTH * j + i) * 3);
				rgb[index_rgb + 0] = rgb_colores[offset_alturas * 3 + 0];
				rgb[index_rgb + 1] = rgb_colores[offset_alturas * 3 + 1];
				rgb[index_rgb + 2] = rgb_colores[offset_alturas * 3 + 2];
			}
			

			//Actualizamos límite del buffer
			if (height_on_screen <= ybuffer[i] && height_on_screen > 0)
				ybuffer[i] = height_on_screen;

			//Avanzamos un paso
			pleft[0] += dx;
			pleft[1] += dy;

			

			//Wrapping

			//pleft[0] = fmodf(pleft[0], 1024.0f);
			//if (pleft[0] < 0.0f) pleft[0] += 1024.0f;
			//pleft[1] = fmodf(pleft[1], 1024.0f);
			//if (pleft[1] < 0.0f) pleft[1] += 1024.0f;
		}

		z += dz;
		//dz += 0.005; // Se pierde muhca resolucion
	}
}

struct BMPHeader
{
	char bfType[2];       /* "BM" */
	int bfSize;           /* Size of file in bytes */
	int bfReserved;       /* set to 0 */
	int bfOffBits;        /* Byte offset to actual bitmap data (= 54) */
	int biSize;           /* Size of BITMAPINFOHEADER, in bytes (= 40) */
	int biWidth;          /* Width of image, in pixels */
	int biHeight;         /* Height of images, in pixels */
	short biPlanes;       /* Number of planes in target device (set to 1) */
	short biBitCount;     /* Bits per pixel (24 in this case) */
	int biCompression;    /* Type of compression (0 if no compression) */
	int biSizeImage;      /* Image size, in bytes (0 if no compression) */
	int biXPelsPerMeter;  /* Resolution in pixels/meter of display device */
	int biYPelsPerMeter;  /* Resolution in pixels/meter of display device */
	int biClrUsed;        /* Number of colors in the color table (if 0, use
						  maximum allowed by biBitCount) */
	int biClrImportant;   /* Number of important colors.  If 0, all colors
						  are important */
};


int write_bmp(const char* filename, int width, int height, char* rgb)
{
	int i, j, ipos;
	int bytesPerLine;
	char* line;

	FILE* file;
	struct BMPHeader bmph;

	/* The length of each line must be a multiple of 4 bytes */

	bytesPerLine = (3 * (width + 1) / 4) * 4;

	strcpy_s(bmph.bfType, "BM");
	bmph.bfOffBits = 54;
	bmph.bfSize = bmph.bfOffBits + bytesPerLine * height;
	bmph.bfReserved = 0;
	bmph.biSize = 40;
	bmph.biWidth = width;
	bmph.biHeight = height;
	bmph.biPlanes = 1;
	bmph.biBitCount = 24;
	bmph.biCompression = 0;
	bmph.biSizeImage = bytesPerLine * height;
	bmph.biXPelsPerMeter = 0;
	bmph.biYPelsPerMeter = 0;
	bmph.biClrUsed = 0;
	bmph.biClrImportant = 0;

	fopen_s(&file, filename, "wb");
	if (file == NULL) return(0);

	fwrite(&bmph.bfType, 2, 1, file);
	fwrite(&bmph.bfSize, 4, 1, file);
	fwrite(&bmph.bfReserved, 4, 1, file);
	fwrite(&bmph.bfOffBits, 4, 1, file);
	fwrite(&bmph.biSize, 4, 1, file);
	fwrite(&bmph.biWidth, 4, 1, file);
	fwrite(&bmph.biHeight, 4, 1, file);
	fwrite(&bmph.biPlanes, 2, 1, file);
	fwrite(&bmph.biBitCount, 2, 1, file);
	fwrite(&bmph.biCompression, 4, 1, file);
	fwrite(&bmph.biSizeImage, 4, 1, file);
	fwrite(&bmph.biXPelsPerMeter, 4, 1, file);
	fwrite(&bmph.biYPelsPerMeter, 4, 1, file);
	fwrite(&bmph.biClrUsed, 4, 1, file);
	fwrite(&bmph.biClrImportant, 4, 1, file);

	line = (char*)malloc(bytesPerLine);

	for (i = height - 1; i >= 0; i--)
	{
		for (j = 0; j < width; j++)
		{
			ipos = 3 * (width * i + j);
			line[3 * j] = rgb[ipos + 2];
			line[3 * j + 1] = rgb[ipos + 1];
			line[3 * j + 2] = rgb[ipos];
		}
		fwrite(line, bytesPerLine, 1, file);
	}

	free(line);
	fclose(file);

	return(1);
}

void readBMP_RGB(char* filename, unsigned char** data_rgb, int* width_rgb, int* height_rgb)
{
	int i;
	FILE* f;
	fopen_s(&f, filename, "rb");
	char info[54];
	fread(info, sizeof(char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&info[18];
	int height = *(int*)&info[22];

	int size = 3 * width * height;
	*data_rgb = new unsigned char[size];// allocate 3 bytes per pixel

	fread(*data_rgb, sizeof(char), size, f); // read the rest of the data at once
	fclose(f);

	for (i = 0; i < size; i += 3)
	{
		unsigned char tmp = (*data_rgb)[i];
		(*data_rgb)[i] = (*data_rgb)[i + 2];
		(*data_rgb)[i + 2] = tmp;
	}

	width_rgb[0] = width;
	height_rgb[0] = height;
}

void readBMP_HM(char* filename, unsigned char** valor_alturas, int* width_alturas, int* height_alturas)
{
	FILE* f;
	fopen_s(&f, filename, "rb");
	char info[54];
	fread(info, sizeof(char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&info[18];
	int height = *(int*)&info[22];

	int size = 3 * width * height;
	unsigned char* valor_alturas_temp = new unsigned char[size];// allocate 3 bytes per pixel
	*valor_alturas = new unsigned char [width * height];
	fread(valor_alturas_temp, sizeof(char), size, f); // read the rest of the data at once

	fclose(f);

	for (int i = 0; i < (width * height); i ++)
	{
		(*valor_alturas)[i] = valor_alturas_temp[i * 3];
	}

	*width_alturas = width;
	*height_alturas = height;
}