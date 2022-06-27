#include <Windows.h>
#include <gdiplus.h>
#include "generate_image.h"
#include <stdio.h>
#include <math.h> 



INT width = IM_WIDTH;
INT height = IM_HEIGHT;
BYTE* p_byte_array = new BYTE[width * 3 * height];
camera_t camera_values;


///////////////////////
/* Function definition*/
LRESULT CALLBACK window_process_messages(HWND hwnd, UINT msg, WPARAM param, LPARAM lparam);
void key_pressed_handler(WPARAM key);
void generate_image();
void draw(HWND hwnd);



/////////////////////////
/* Function implementation*/

/// <summary>
/// Windows main function
/// </summary>
/// <param name="current_instance"></param>
/// <param name="prev_instance"></param>
/// <param name="cmdline"></param>
/// <param name="cmd_count"></param>
/// <returns></returns>
int WINAPI WinMain(HINSTANCE current_instance, HINSTANCE prev_instance, PSTR cmdline, INT cmd_count)
{
	// First of all load bmp maps
	init_voxel_maps();

	// Initialize camera object
	camera_values.x = 512;
	camera_values.y = 512;
	camera_values.height = 100;
	camera_values.angle = 0;
	camera_values.horizon = IM_HEIGHT/2;
	camera_values.distance = 600;


	// Initialize GDI+
	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplustoken;
	Gdiplus::GdiplusStartup(&gdiplustoken, &gdiplusStartupInput, nullptr);

	// Create window class
	const wchar_t* CLASSNAME = L"ImageShowWindowClass";
	WNDCLASS wc{};
	wc.hInstance = current_instance;
	wc.lpszClassName = L"ImageShowWindowClass";
	wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)COLOR_WINDOW;
	wc.lpfnWndProc = window_process_messages;
	RegisterClass(&wc);

	// Create window
	CreateWindow(CLASSNAME, L"Image Show", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, width+15, height+40, nullptr, nullptr, nullptr, nullptr);

	// Window loop

	MSG msg{};

	while (GetMessage(&msg, nullptr, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	Gdiplus::GdiplusShutdown(gdiplustoken);
	return 0;
}


/// <summary>
/// Callback to process windows msg events 
/// </summary>
/// <param name="hwnd"></param>
/// <param name="msg"></param>
/// <param name="param"></param>
/// <param name="lparam"></param>
/// <returns></returns>
LRESULT CALLBACK window_process_messages(HWND hwnd, UINT msg, WPARAM param, LPARAM lparam)
{
	switch (msg)
	{
		case WM_PAINT:
			draw(hwnd);
			return 0;
		case WM_DESTROY:
			PostQuitMessage(0);
			return 0;
		case  WM_KEYDOWN :
			key_pressed_handler(param);
			//InvalidateRect(hwnd, 0, TRUE);
			RedrawWindow(hwnd, NULL, NULL, RDW_INVALIDATE);

			return 0;
		default:
			return DefWindowProc(hwnd, msg, param, lparam);
	}
}


/// <summary>
/// Function to draw the raw image on the window
/// </summary>
/// <param name="hdc"></param>
void draw(HWND hwnd)
{
	generate_image();

	PAINTSTRUCT ps;
	HDC hdc = BeginPaint(hwnd, &ps);
	
	Gdiplus::Graphics gf(hdc);
	INT stride = 3 * width;
	Gdiplus::Bitmap bmp_arr(width, height, stride, PixelFormat24bppRGB, (BYTE*)p_byte_array);

	gf.DrawImage(&bmp_arr, 0, 0);
	EndPaint(hwnd, &ps);
}

/// <summary>
/// Function that updates the image RGB values
/// </summary>
void generate_image()
{
	BYTE aux;
	obtain_voxel_bmp(width, height, p_byte_array, camera_values);
	for (int i = 0; i < width * height*3; i += 3)
	{
		aux = p_byte_array[i];
		p_byte_array[i] = p_byte_array[i + 2];
		p_byte_array[i+2]=aux;
	}
}

/// <summary>
/// Function that handles the kwy pressed event
/// </summary>
/// <param name="key"></param>
void key_pressed_handler(WPARAM key)
{
	switch (key)
	{
		// Rotate camera left
		case VK_LEFT:
		case 'A':
			camera_values.angle = camera_values.angle + 0.05;
			break;

		// Rotate camera right
		case VK_RIGHT:
		case 'D':
			camera_values.angle = camera_values.angle - 0.05;
			break;

		// Rotate camera up
		case 'Q':
			camera_values.horizon += 5;
			break;

		// Rotate camera down
		case 'E':
			camera_values.horizon -= 5;
			break;

		// Move forward
		case VK_UP:
		case 'W':
			camera_values.x -= sin(camera_values.angle) *1.8;
			camera_values.y -= cos(camera_values.angle) * 1.8;
			break;

		// Move backwards
		case VK_DOWN:
		case 'S':
			camera_values.x += sin(camera_values.angle) * 1.8;
			camera_values.y += cos(camera_values.angle) * 1.8;
			break;

		// Move up
		case 'R':
			camera_values.height = camera_values.height + 10;
			break;

		// Move down
		case 'F':
			camera_values.height = camera_values.height - 10;
			break;

		default:
			break;
	}
}
