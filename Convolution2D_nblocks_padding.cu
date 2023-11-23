/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005 
#define MAX_BLOCK_WIDTH 32

float matrix_diff(float *m_cpu, float *m_gpu, int size) {
    int i, j;
    float diff, max = 0;

    for(i = 0; i < size; i++) {
        for(j = 0; j < size; j++) {
            diff = m_cpu[i * size + j] - m_gpu[i * size + j];
            if(ABS(diff) > max) {
                max = ABS(diff);
            }
        }
    }
    return(max);
}

__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter,
                                  int imageW, int imageH, int filterR) {
    int k;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    
    for (k = -filterR; k <= filterR; k++) {
        int d = column + k;

        sum += d_Src[(row + filterR) * (imageW + 2*filterR) + d + filterR] * d_Filter[filterR - k];          

        d_Dst[(row + filterR) * (imageW + 2*filterR) + filterR + column] = sum;
    }

}

__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter,
    			   int imageW, int imageH, int filterR) {
    int k;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    for (k = -filterR; k <= filterR; k++) {
        int d = row + k;
        
        sum += d_Src[(d + filterR) * (imageW + 2*filterR) + filterR + column] * d_Filter[filterR - k];
        
        d_Dst[row * imageW + column] = sum;
    }

}
 

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;
      int row = y, column = x;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        sum += h_Src[(row + filterR) * (imageW + 2*filterR) + d + filterR] * h_Filter[filterR - k];     

        h_Dst[(row + filterR) * (imageW + 2* filterR) + filterR + column] = sum;
      }
    }
  }
        
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int row = y, column = x;
        int d = row + k;

        sum += h_Src[(d + filterR) * (imageW + 2*filterR) + filterR + column] * h_Filter[filterR - k];
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }   
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    float
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU;

    float
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU;

    cudaEvent_t start, stop;


    int imageW;
    int imageH;
    //int error_check;
    float max_diff, milliseconds = 0.0;
    unsigned int i;
    struct timespec tv1, tv2;

	  printf("Enter filter radius : ");
	  scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)malloc((imageW + 2*filter_radius) * (imageH + 2*filter_radius) * sizeof(float));
    h_Buffer    = (float *)malloc((imageW + 2*filter_radius) * (imageH + 2*filter_radius) * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));

    cudaMalloc(&d_Filter, FILTER_LENGTH * sizeof(float));
    cudaMalloc(&d_Input, (imageW + 2*filter_radius) * (imageH + 2*filter_radius) * sizeof(float));
    cudaMalloc(&d_Buffer, (imageW + 2*filter_radius) * (imageH + 2*filter_radius) * sizeof(float));
    cudaMalloc(&d_OutputGPU, imageW * imageH * sizeof(float));

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < (imageW + 2*filter_radius) * (imageH + 2*filter_radius); i++) {
        if(i <  (imageW + 2*filter_radius) * filter_radius                                ||
           i >= (imageW + 2*filter_radius) * (imageW + 2 * filter_radius - filter_radius) ||
           i % (imageW + 2*filter_radius) <= filter_radius - 1                            ||
           i % (imageW + 2*filter_radius) >= (imageW + 2*filter_radius) - filter_radius)   {
            h_Input[i] = 0.0;
            h_Buffer[i] = 0.0;
        }
        else {
            h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
        }
        
    }


    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    printf ("CPU exec time = %7g sec\n\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));

    // Kommati pou ekteleitai sthn GPU
    printf("GPU computation...\n");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, h_Input, (imageW + 2*filter_radius) * (imageH + 2*filter_radius) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Buffer, h_Buffer, (imageW + 2*filter_radius) * (imageH + 2*filter_radius) * sizeof(float), cudaMemcpyHostToDevice);

    //dim3 dimGrid(1, 1);
    //dim3 dimBlock(imageW, imageH);

    dim3 dimGrid(imageW / MAX_BLOCK_WIDTH, imageH / MAX_BLOCK_WIDTH);
    dim3 dimBlock(MAX_BLOCK_WIDTH, MAX_BLOCK_WIDTH);

    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA Error: %s in %s, line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
        return(1);
    }

    cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Cuda exec time: %f sec\n\n", milliseconds / 1000);


    // Kanete h sugrish anamesa se GPU kai CPU kai` an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  

    max_diff = matrix_diff(h_OutputCPU, h_OutputGPU, imageW);
    printf("Max diff = %f\n", max_diff);


    // free all the allocated memory
    free(h_OutputCPU);
    free(h_OutputGPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    cudaFree(d_OutputGPU);
    cudaFree(d_Buffer);
    cudaFree(d_Input);
    cudaFree(d_Filter);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();


    return 0;
}
