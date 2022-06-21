
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
//#include "stdafx.h"
// Archivos de encabezado de Windows
#include <windows.h>

#include <stdio.h>


__global__
void voxel_kernel(int n, float a, float* x, float* y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) y[i] = a * x[i] + y[i];
}

extern "C" {

	int __declspec(dllexport) generate_voxel_image(int N, char* msg) 
	{
		float* x, * y, * d_x, * d_y;
		x = (float*)malloc(N * sizeof(float));
		y = (float*)malloc(N * sizeof(float));

		cudaMalloc(&d_x, N * sizeof(float));
		cudaMalloc(&d_y, N * sizeof(float));

		for (int i = 0; i < N; i++) {
			x[i] = 1.0f;
			y[i] = 2.0f;
		}

		cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

		voxel_kernel << <(N + 255) / 256, 256 >> > (N, 2.0, d_x, d_y);

		cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

		float maxError = 0.0f;
		for (int i = 0; i < N; i++) {
			maxError = max(maxError, abs(y[i] - 4.0f));
		}
		sprintf_s(&msg[strlen(msg)], 999 - strlen(msg), "Max error: %f\r\n", maxError);

		free(x);
		free(y);
		cudaFree(d_x);
		cudaFree(d_y);

		return strlen(msg);
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