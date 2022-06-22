
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
//#include "stdafx.h"
// Archivos de encabezado de Windows
#include <windows.h>
#include <stdio.h>
#include <math.h>

typedef struct camera_s
{
	float x;        // 512 // x position on the map
	float y;        // 800 // y position on the map
	float height;   // 78 // height of the camera
	float angle;    // 0 // direction of the camera
	float horizon;  // 100 // horizon position (look up and down)
	float distance; // 800   // distance of map
} camera_t;

__global__
void voxel_kernel(int* img_width, int* img_height, int* map_width, int* map_height, camera_t* camera, unsigned char* rgb_colormap, unsigned char* heightmap, unsigned char* rgb_result) {

	// calculate z index
	// z siempre va de 1 en 1  y si es < que distancia, hacemos el calculo
	int z = blockIdx.x * blockDim.x + threadIdx.x;
	if (z > camera->distance) return;

	/* NOTA!!
	Todo lo siguiente se podria calcular antes de llamar al kernel, es igual para todos los threads
	*/

	//Posición cámara en 1024x1024
	float p[2] = { camera->x, camera->y };

	//Altura cámara en bloques
	float height = camera->height;

	//Altura del horizonte con respecto a IM_HEIGHT
	//float horizon = (float)IM_HEIGHT / 2.0f;
	float horizon = camera->horizon;

	//Escalado de altura (cuanto más pequeño más exagerada la altura)
	float scale_height = 0.0015f;

	//Distancia máxima de renderizado en pixeles

	// solo se usa al principio para comparar con z
	//float distance = camera->distance;

	//Ángulo de la cámara en grados
	float phi = camera->angle;
	//Precalculamos los parámetros del ángulo de visión
	float sinphi = sin(phi);
	float cosphi = cos(phi);


	/* Esto ya es local de cada thread*/

	//Cálculo de los extremos de las líneas para FOV 90º
	float pleft[2] = { (-cosphi * z - sinphi * z) + p[0],
						(sinphi * z - cosphi * z) + p[1] };
	float pright[2] = { (cosphi * z - sinphi * z) + p[0],
						(-sinphi * z - cosphi * z) + p[1] };

	//Segmentación líneas
	float dx = (pright[0] - pleft[0]) / (float)*img_width;
	float dy = (pright[1] - pleft[1]) / (float)*img_width;

	for (int i = 0; i < *img_width; i++)
	{
		//Sacamos altura del pixel a dibujar
		//int offset_alturas = (int) (pleft[0] + 1024 * (int)pleft[1]);

		/* NOTA Sustituir 1023y 1024  por map_width*/
		// No detecta floor en cuda
		//int offset_alturas = (((int)floor(pleft[1]) & 1023) * 1024) + ((int)floor(pleft[0]) & 1023);
		int offset_alturas = (((int)(pleft[1]) & 1023) * 1024) + ((int)(pleft[0]) & 1023);
		float height_on_screen = (((height - map_height[offset_alturas]) / (z * scale_height)) + horizon);

		//Imprimimos colores hasta llegar al límite del buffer
		if (height_on_screen < 0) height_on_screen = 0;
		if (height_on_screen > *img_height) height_on_screen = *img_height;
		//for (int j = (int)floor(height_on_screen); j < (int)floor(*img_height); j++)
		for (int j = (int)(height_on_screen); j < (int)(*img_height); j++)
		{
			int index_rgb = ((*img_width * j + i) * 3);
			rgb_result[index_rgb + 0] = rgb_colormap[offset_alturas * 3 + 0];
			rgb_result[index_rgb + 1] = rgb_colormap[offset_alturas * 3 + 1];
			rgb_result[index_rgb + 2] = rgb_colormap[offset_alturas * 3 + 2];
		}

		//Avanzamos un paso
		pleft[0] += dx;
		pleft[1] += dy;
	}
}

extern "C" {

	/* This function is exported in DLL and calls cuda kernel*/
	int __declspec(dllexport) generate_voxel_image(int img_width, int img_height, int map_width, int map_height, camera_t camera, unsigned char* rgb_colormap, unsigned char* heightmap, unsigned char* rgb_result)
	{
		int map_size_rgb = map_width * map_height * 3;
		int map_size = map_width * map_height;
		int size_rgb = img_width * img_height * 3;

		// Allocate memory in device
		int* img_width_d;
		int* img_height_d;
		int* map_width_d;
		int* map_height_d;
		unsigned char* rgb_colormap_d;
		unsigned char* heightmap_d;
		unsigned char* rgb_result_d;
		camera_t* camera_d;


		cudaMalloc(&img_width_d, sizeof(int));
		cudaMalloc(&img_height_d, sizeof(int));
		cudaMalloc(&map_width_d, sizeof(int));
		cudaMalloc(&map_height_d, sizeof(int));
		cudaMalloc(&camera_d, sizeof(camera_t));

		cudaMalloc(&rgb_colormap_d, map_size_rgb * sizeof(char));
		cudaMalloc(&heightmap_d, map_size * sizeof(char));
		cudaMalloc(&rgb_result_d, size_rgb * sizeof(char));

		// Copy data from host to device
		//Inicializar vector rgb a azulito para el cielo
		for (int i = 0; i < img_width * img_height * 3; i += 3)
		{
			rgb_result[i] = 148;
			rgb_result[i + 1] = 209;
			rgb_result[i + 2] = 239;
		}


		cudaMemcpy(img_width_d, &img_width, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(img_height_d, &img_height, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(map_width_d, &map_width, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(map_height_d, &map_height,sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(camera_d, &camera, sizeof(camera_t), cudaMemcpyHostToDevice);

		cudaMemcpy(rgb_colormap_d, rgb_colormap, map_size_rgb * sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(heightmap_d, heightmap, map_size * sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(rgb_result_d, rgb_result, map_size * sizeof(char), cudaMemcpyHostToDevice);


		// Launch kernel
		voxel_kernel <<< 512 , 512 >>> (img_width_d, img_height_d, map_width_d, map_height_d, camera_d, rgb_colormap_d, heightmap_d, rgb_result_d);


		cudaMemcpy(rgb_result, rgb_result_d, size_rgb * sizeof(char), cudaMemcpyDeviceToHost);


		cudaFree(img_width_d);
		cudaFree(img_height_d);
		cudaFree(map_width_d);
		cudaFree(map_height_d);

		cudaFree(rgb_colormap_d);
		cudaFree(heightmap_d);
		cudaFree(rgb_result_d);

		return 0;
	}
}

BOOL APIENTRY DllMain(HMODULE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved
)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		break;
	case DLL_THREAD_ATTACH:
		break;
	case DLL_THREAD_DETACH:
		break;
	case DLL_PROCESS_DETACH:
		break;
	default:
		break;
	}

	return TRUE;
}