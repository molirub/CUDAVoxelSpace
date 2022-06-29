
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
void voxel_kernel(int* img_width, int* img_height, int* map_width, int* map_height,
	              float* pleftx, float* plefty, float* dx, float* dy, float* z, float* height, float* scale_height, float* horizon,
	              unsigned char* rgb_colormap, unsigned char* heightmap, unsigned char* rgb_result)
{

	// Sacamos i (columna actual
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= *img_width) return;
	float pleftx_i = *pleftx + *dx * i;
	float plefty_i = *plefty + *dy * i;

	//Sacamos altura del pixel a dibujar
	//int offset_alturas = (int) (pleft[0] + 1024 * (int)pleft[1]);

	/* NOTA Sustituir 1023y 1024  por map_width*/
	int offset_alturas = (((int)(floorf(plefty_i)) & 1023) * 1024) + ((int)(floorf(pleftx_i)) & 1023);
	float height_on_screen = (((*height - heightmap[offset_alturas]) / (*z * *scale_height)) + *horizon);

	//Imprimimos colores hasta llegar al límite del buffer
	if (height_on_screen < 0) height_on_screen = 0;
	if (height_on_screen > *img_height) height_on_screen = *img_height;
	//Pintamos linea vertical
	for (int j = (int)(floorf(height_on_screen)); j < (int)(floorf(*img_height)); j++)
	{
		int index_rgb = ((*img_width * j + i) * 3);
		rgb_result[index_rgb + 0] = rgb_colormap[offset_alturas * 3 + 0];
		rgb_result[index_rgb + 1] = rgb_colormap[offset_alturas * 3 + 1];
		rgb_result[index_rgb + 2] = rgb_colormap[offset_alturas * 3 + 2];
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

		float* pleftx_d;
		float* plefty_d;
		float* dx_d;
		float* dy_d;
		float* z_d;
		float* height_d;
		float* scale_height_d;
		float* horizon_d;
		


		cudaMalloc(&img_width_d, sizeof(int));
		cudaMalloc(&img_height_d, sizeof(int));
		cudaMalloc(&map_width_d, sizeof(int));
		cudaMalloc(&map_height_d, sizeof(int));

		cudaMalloc(&rgb_colormap_d, map_size_rgb * sizeof(char));
		cudaMalloc(&heightmap_d, map_size * sizeof(char));
		cudaMalloc(&rgb_result_d, size_rgb * sizeof(char));

		cudaMalloc(&pleftx_d, sizeof(float));
		cudaMalloc(&plefty_d, sizeof(float));
		cudaMalloc(&dx_d, sizeof(float));
		cudaMalloc(&dy_d, sizeof(float));
		cudaMalloc(&z_d, sizeof(float));
		cudaMalloc(&height_d, sizeof(float));
		cudaMalloc(&scale_height_d, sizeof(float));
		cudaMalloc(&horizon_d, sizeof(float));


		// Perform some previous calculations
		// Copy data from host to device
		//Inicializar vector rgb a azulito para el cielo
		for (int i = 0; i < img_width * img_height * 3; i += 3)
		{
			rgb_result[i] = 148;
			rgb_result[i + 1] = 209;
			rgb_result[i + 2] = 239;
		}
		float p[2] = { camera.x, camera.y };
		//Altura cámara en bloques
		float height = camera.height;
		//Altura del horizonte 
		float horizon = camera.horizon;
		//Escalado de altura (cuanto más pequeño más exagerada la altura)
		float scale_height = 0.0015f;
		//Distancia máxima de renderizado en pixeles
		float distance = camera.distance;
		//Ángulo de la cámara en grados
		float phi = camera.angle;
		//phi = phi * 180.0f / 3.1415f;

		//Precalculamos los parámetros del ángulo de visión
		float sinphi = sin(phi);
		float cosphi = cos(phi);

		//Generación de cada una de la líneas de delante a detrás
		float z = 1.0f;
		float dz = 1.00f;

		// Copiamos memoria a device antes del bucle
		cudaMemcpy(img_width_d, &img_width, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(img_height_d, &img_height, sizeof(int), cudaMemcpyHostToDevice);

		cudaMemcpy(map_width_d, &map_width, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(map_height_d, &map_height, sizeof(int), cudaMemcpyHostToDevice);

		cudaMemcpy(rgb_colormap_d, rgb_colormap, map_size_rgb * sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(heightmap_d, heightmap, map_size * sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(rgb_result_d, rgb_result, size_rgb * sizeof(char), cudaMemcpyHostToDevice);


		
		cudaMemcpy(height_d, &height, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(scale_height_d, &scale_height, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(horizon_d, &horizon, sizeof(float), cudaMemcpyHostToDevice);


		z = distance;
		while (z > 1)
		{

			//Cálculo de los extremos de las líneas para FOV 90º
			float pleft[2] = { (-cosphi * z - sinphi * z) + p[0],
								(sinphi * z - cosphi * z) + p[1] };
			float pright[2] = { (cosphi * z - sinphi * z) + p[0],
								(-sinphi * z - cosphi * z) + p[1] };

			//Segmentación líneas
			float dx = (pright[0] - pleft[0]) / (float)img_width;
			float dy = (pright[1] - pleft[1]) / (float)img_width;


			cudaMemcpy(pleftx_d, &pleft[0], sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(plefty_d, &pleft[1], sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(dx_d, &dx, sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(dy_d, &dy, sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(z_d, &z, sizeof(float), cudaMemcpyHostToDevice);

			// Launch kernel
			voxel_kernel <<< 128, 128 >>> (img_width_d, img_height_d, map_width_d, map_height_d, 
				                           pleftx_d, plefty_d, dx_d, dy_d, z_d, height_d, scale_height_d, horizon_d,
				                           rgb_colormap_d, heightmap_d, rgb_result_d);



			z -= dz;
			//dz += 0.005; // Se pierde muhca resolucion
		}

		// Copiamos del device al host
		cudaMemcpy(rgb_result, rgb_result_d, size_rgb * sizeof(char), cudaMemcpyDeviceToHost);


		cudaFree(img_width_d);
		cudaFree(img_height_d);
		cudaFree(map_width_d);
		cudaFree(map_height_d);

		cudaFree(pleftx_d);
		cudaFree(plefty_d);
		cudaFree(dx_d);
		cudaFree(dy_d);
		cudaFree(z_d);
		
		cudaFree(height_d);
		cudaFree(scale_height_d);
		cudaFree(horizon_d);

		cudaFree(rgb_colormap_d);
		cudaFree(heightmap_d);
		cudaFree(rgb_result_d);

		return 0;
	}
}

/*
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

		//Wrapping
		//pleft[0] = fmodf(pleft[0], 1024.0f);
		//if (pleft[0] < 0.0f) pleft[0] += 1024.0f;
		//pleft[1] = fmodf(pleft[1], 1024.0f);
		//if (pleft[1] < 0.0f) pleft[1] += 1024.0f;

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

*/

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