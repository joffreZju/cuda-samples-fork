/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <cstdio>
#include <helper_cuda.h>
#include <helper_string.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <assert.h>
#include <windows.h>
#include <iostream>

#define MAX_DEPTH 16
#define INSERTION_SORT 32

////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
__device__ void selection_sort(unsigned int *data, int left, int right) {
  for (int i = left; i <= right; ++i) {
    unsigned min_val = data[i];
    int min_idx = i;

    // Find the smallest value in the range [left, right].
    for (int j = i + 1; j <= right; ++j) {
      unsigned val_j = data[j];

      if (val_j < min_val) {
        min_idx = j;
        min_val = val_j;
      }
    }

    // Swap the values.
    if (i != min_idx) {
      data[min_idx] = data[i];
      data[i] = min_val;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort(unsigned int *data, int left, int right,
                                     int depth) {
  // If we're too deep or there are few elements left, we use an insertion
  // sort...
  if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
    selection_sort(data, left, right);
    return;
  }

  unsigned int *lptr = data + left;
  unsigned int *rptr = data + right;
  unsigned int pivot = data[(left + right) / 2];

  // Do the partitioning.
  while (lptr <= rptr) {
    // Find the next left- and right-hand values to swap
    unsigned int lval = *lptr;
    unsigned int rval = *rptr;

    // Move the left pointer as long as the pointed element is smaller than the
    // pivot.
    while (lval < pivot) {
      lptr++;
      lval = *lptr;
    }

    // Move the right pointer as long as the pointed element is larger than the
    // pivot.
    while (rval > pivot) {
      rptr--;
      rval = *rptr;
    }

    // If the swap points are valid, do the swap!
    if (lptr <= rptr) {
      *lptr++ = rval;
      *rptr-- = lval;
    }
  }

  // Now the recursive part
  int nright = rptr - data;
  int nleft = lptr - data;

  // Launch a new block to sort the left part.
  if (left < (rptr - data)) {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cdp_simple_quicksort<<<1, 1, 0, s>>>(data, left, nright, depth + 1);
    cudaStreamDestroy(s);
  }

  // Launch a new block to sort the right part.
  if ((lptr - data) < right) {
    cudaStream_t s1;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cdp_simple_quicksort<<<1, 1, 0, s1>>>(data, nleft, right, depth + 1);
    cudaStreamDestroy(s1);
  }
}

BOOL ReadSection(const char* sectionName, BYTE** pData, DWORD* pSize) {
    HMODULE hModule = GetModuleHandle(NULL);
    if (hModule == NULL) {
        return FALSE;
    }
    // Get DOS header
    IMAGE_DOS_HEADER* dosHeader = reinterpret_cast<IMAGE_DOS_HEADER*>(hModule);
    // Get NT headers
    IMAGE_NT_HEADERS* ntHeaders = reinterpret_cast<IMAGE_NT_HEADERS*>((BYTE*)dosHeader + dosHeader->e_lfanew);

    // Get the first section header
    IMAGE_SECTION_HEADER* sections = IMAGE_FIRST_SECTION(ntHeaders);
    for (int i = 0; i < ntHeaders->FileHeader.NumberOfSections; ++i) {
        // Compare the name to find the desired section
        if (strncmp(reinterpret_cast<const char*>(sections[i].Name), sectionName, IMAGE_SIZEOF_SHORT_NAME) == 0) {
            *pData = reinterpret_cast<BYTE*>(hModule) + sections[i].VirtualAddress;
            *pSize = sections[i].Misc.VirtualSize;
            return TRUE;
        }
    }
    // Specified section not found
    return FALSE;
}

CUfunction qsort_kernel;

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void run_qsort(unsigned int *data, unsigned int nitems) {
  // Prepare CDP for the max depth 'MAX_DEPTH'.
    CUresult cuRes;
    /*
    const char* sectionName = ".nv_fatb"; // For example, the '.data' section of the PE
    BYTE* section_data = nullptr;
    DWORD section_size = 0;

    if (!ReadSection(sectionName, &section_data, &section_size)) {
        std::cerr << sectionName << " section not found!" << std::endl;
        exit(0);
    }
    //// Initialize
    //cuRes = cuInit(0);
    //assert(cuRes == CUDA_SUCCESS);

    //// Get number of devices supporting CUDA
    //int deviceCount = 0;
    //cuRes = cuDeviceGetCount(&deviceCount);
    //assert(cuRes == CUDA_SUCCESS);
    //if (deviceCount == 0) {
    //    //printf("There is no device supporting CUDA.\n");
    //    exit(0);
    //}

    //// Get handle for device 0
    //CUdevice cuDevice;
    //cuRes = cuDeviceGet(&cuDevice, 0);
    //assert(cuRes == CUDA_SUCCESS);

    //int major = 0, minor = 0;
    //char deviceName[256];
    //// get compute capabilities and the devicename
    //cuRes = cuDeviceComputeCapability(&major, &minor, cuDevice);
    //assert(cuRes == CUDA_SUCCESS);

    //cuRes = cuDeviceGetName(deviceName, 256, cuDevice);
    //assert(cuRes == CUDA_SUCCESS);

    //// Create context
    //CUcontext cuContext;
    //cuRes = cuCtxCreate(&cuContext, 0, cuDevice);
    //assert(cuRes == CUDA_SUCCESS);

    // Create module from binary file
    CUmodule cuModule;
    cuRes = cuModuleLoadData(&cuModule, section_data);
    assert(cuRes == CUDA_SUCCESS);

    // Get function handle from module
    CUfunction quick_sort_kernel;
    cuRes = cuModuleGetFunction(&quick_sort_kernel, cuModule, "_Z20cdp_simple_quicksortPjiii");
    assert(cuRes == CUDA_SUCCESS);

    cuuint32_t cnt = 0;
    cuRes = cuModuleGetFunctionCount(&cnt, cuModule);
    assert(cuRes == CUDA_SUCCESS);

    CUfunction funcs[10000];
    cuRes = cuModuleEnumerateFunctions(funcs, 10000, cuModule);
    assert(cuRes == CUDA_SUCCESS);
    */
  // Launch on device
  int left = 0;
  int right = nitems - 1;
  int depth = 0;
  std::cout << "Launching kernel on the GPU" << std::endl;
  void* arr[] = { &data, &left, &right, &depth };
  //cuRes = cuLaunchKernel(qsort_kernel, 1, 1, 1, 1, 1, 1, 0, 0, &arr[0], 0);
  //assert(cuRes == CUDA_SUCCESS);

  cuRes = cuLaunchKernel(qsort_kernel, 1, 1, 1, 1, 1, 1, 0, 0, &arr[0], 0);
  assert(cuRes == CUDA_SUCCESS);

  //cdp_simple_quicksort<<<1, 1>>>(data, left, right, 0);
  checkCudaErrors(cudaDeviceSynchronize());
}

////////////////////////////////////////////////////////////////////////////////
// Initialize data on the host.
////////////////////////////////////////////////////////////////////////////////
void initialize_data(unsigned int *dst, unsigned int nitems) {
  // Fixed seed for illustration
  srand(2047);

  // Fill dst with random values
  for (unsigned i = 0; i < nitems; i++) dst[i] = rand() % nitems;
}

////////////////////////////////////////////////////////////////////////////////
// Verify the results.
////////////////////////////////////////////////////////////////////////////////
void check_results(int n, unsigned int *results_d) {
  unsigned int *results_h = new unsigned[n];
  checkCudaErrors(cudaMemcpy(results_h, results_d, n * sizeof(unsigned),
                             cudaMemcpyDeviceToHost));

  for (int i = 1; i < n; ++i)
    if (results_h[i - 1] > results_h[i]) {
      std::cout << "Invalid item[" << i - 1 << "]: " << results_h[i - 1]
                << " greater than " << results_h[i] << std::endl;
      exit(EXIT_FAILURE);
    }

  for (int i = 0; i < n; i++)
      std::cout << "Sort Data [" << i << "]: " << results_h[i] << std::endl;
  std::cout << "OK" << std::endl;
  delete[] results_h;
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    typedef void* (*pfn_get_qsort_kernel)();
    HMODULE hr = LoadLibrary("C:/Users/sanze/Documents/mycode/d3d12-remote/build/benchmark/ecuda/Debug/ecuda64.dll");
    pfn_get_qsort_kernel func = (pfn_get_qsort_kernel)GetProcAddress(hr, "get_qsort_kernel");
    qsort_kernel = (CUfunction)func();

  int num_items = 128;
  bool verbose = false;

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "h")) {
    std::cerr << "Usage: " << argv[0]
              << " num_items=<num_items>\twhere num_items is the number of "
                 "items to sort"
              << std::endl;
    exit(EXIT_SUCCESS);
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "v")) {
    verbose = true;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "num_items")) {
    num_items = getCmdLineArgumentInt(argc, (const char **)argv, "num_items");

    if (num_items < 1) {
      std::cerr << "ERROR: num_items has to be greater than 1" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // Find/set device and get device properties
  //int device = -1;
  //cudaDeviceProp deviceProp;
  //device = findCudaDevice(argc, (const char **)argv);
  //checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));

  //if (!(deviceProp.major > 3 ||
  //      (deviceProp.major == 3 && deviceProp.minor >= 5))) {
  //  printf("GPU %d - %s  does not support CUDA Dynamic Parallelism\n Exiting.",
  //         device, deviceProp.name);
  //  exit(EXIT_WAIVED);
  //}

  // Create input data
  unsigned int *h_data = 0;
  unsigned int *d_data = 0;

  // Allocate CPU memory and initialize data.
  std::cout << "Initializing data:" << std::endl;
  h_data = (unsigned int *)malloc(num_items * sizeof(unsigned int));
  initialize_data(h_data, num_items);

  if (1) {
    for (int i = 0; i < num_items; i++)
      std::cout << "Data [" << i << "]: " << h_data[i] << std::endl;
  }

  // Allocate GPU memory.
  checkCudaErrors(
      cudaMalloc((void **)&d_data, num_items * sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(d_data, h_data, num_items * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));

  // Execute
  std::cout << "Running quicksort on " << num_items << " elements" << std::endl;
  run_qsort(d_data, num_items);

  // Check result
  std::cout << "Validating results: ";
  check_results(num_items, d_data);

  free(h_data);
  checkCudaErrors(cudaFree(d_data));

  exit(EXIT_SUCCESS);
}
