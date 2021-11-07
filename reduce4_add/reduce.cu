#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <math.h>

#include "timing.h"

//#define UPB RAND_MAX 
#define UPB 2 
#define LEVELS 5
#define MAXDRET 102400
#define THRESH 2
#define W (sizeof(uint) * 8)

 
int shouldPrint = 0; //global variable

inline void check_cuda_errors(const char *filename, const int line_number)
{
    cudaThreadSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
        exit(-1);
    }
}

inline int nextPowerOfTwo(int x)
{
    return 1U << ( W - __builtin_clz(x - 1) );
}


//Pad the input data with zeros if input size is not power of 2.
float * GPUPrep(float *data, int n1, int *n2)
{
    *n2 = nextPowerOfTwo(n1);
    float *ret = (float *) malloc( *n2 * sizeof(float) );
    memset(ret, 0, *n2 * sizeof(float) );
    memcpy(ret, data, n1 * sizeof(float));

    return ret;
}



float * fillArray(int n, int upbound)
{
   int i;
   
   float *ret = (float *)malloc(sizeof(float) * n );

   /* Intializes random number generator */
   //seeds the random number generator used by the function rand.
   srand(time(NULL));

   /* generate n random numbers from 0 to unbound - 1 */
   for( i = 0 ; i < n ; i++ ) {
      ret[i] = rand() % upbound * 1.0f;
   }
   return ret;
}

void printArray(float *arr, int n){

   int i;

   for(i = 0; i < n; i ++)
      printf("%5.0f ", arr[i]);

   printf("\n");
}

float cpuReduce(float *h_in, int n)
{
    float s = 0;
    int i;
    for(i = 0; i < n; i ++)
    {
        s += h_in[i];
    }
    return s;
}


__global__ void reduce2(float *in, float *out, int n)
{
    extern __shared__ float sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
        {
            sdata[tid] += sdata[tid + s]; //sum number stored in low index
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// Check if an int is power of 2
int isPowerOfTwo (unsigned int x)
{
    while (((x & 1) == 0) && x > 1) /* While x is even and > 1 */
        x >>= 1;
    return (x == 1);
}


void usage()
{
   printf("Usage: ./progName blockWidth numElementsInput p \n");
   exit(-1);
}


//
void runCUDA( float *arr, int  n_old, int tile_width)
{    
   // set up host memory
   float *h_in, *h_out, *d_in, *d_out;
   int n = 0;
   h_in = GPUPrep(arr, n_old, &n); //Make input size power of 2.
   h_out = (float *)malloc(MAXDRET * sizeof(float));
   memset(h_out, 0, MAXDRET * sizeof(float));


   if( ! h_in || ! h_out )
   {
       printf("Error in host memory allocation!\n");
       exit(-1);
   }
   int num_block = ceil(n / (float)tile_width);
   printf("Num of blocks is %d\n", num_block);
   dim3 block(tile_width, 1, 1);
   dim3 grid(num_block, 1, 1);

   // allocate storage for the device
   cudaMalloc((void**)&d_in, sizeof(float) * n);
   cudaMalloc((void**)&d_out, sizeof(float) * MAXDRET);
   cudaMemset(d_out, 0, sizeof(float) * MAXDRET);

   // copy input to the device
   cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice);

   // time the kernel launches using CUDA events
   cudaEvent_t launch_begin, launch_end;
   cudaEventCreate(&launch_begin);
   cudaEventCreate(&launch_end);

   printf("The input array is:\n");
   //print out original array
   if(shouldPrint)
       printArray(h_in, n);


   int num_in = n, num_out = ceil((float)n / tile_width);
   float *temp;
  
   printf("Timing simple GPU implementation… \n");
   // record a CUDA event immediately before and after the kernel launch
   cudaEventRecord(launch_begin,0);
   while( 1 )
   {
       reduce2<<<grid, block, tile_width * sizeof(float)>>>(d_in, d_out, num_in);
       check_cuda_errors(__FILE__, __LINE__);
       cudaDeviceSynchronize();

       // if the number of local sum returned by kernel is greater than the threshold,
       // we do reduction on GPU for these returned local sums for another pass,
       // until, num_out < threshold
       if(num_out >= THRESH)
       {
           num_in = num_out;
           num_out = ceil((float)num_out / tile_width);
           grid.x = num_out; //change the grid dimension in x direction
           //Swap d_in and d_out, so that in the next iteration d_out is used as input and d_in is the output.
           temp = d_in;
           d_in = d_out;
           d_out = temp;
       }
       else
       {
           //copy the ouput of last lauch back to host,
           cudaMemcpy(h_out, d_out, sizeof(float) * num_out, cudaMemcpyDeviceToHost);
           break;
       }
    }//end of while

    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    // measure the time spent in the kernel
    float time = 0;
    cudaEventElapsedTime(&time, launch_begin, launch_end);

    printf(" done! GPU time cost in second: %f\n", time / 1000);
    printf("The output from device is:");
    //if(shouldPrint)
    printArray(h_out, num_out);

    // deallocate device memory
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_out);
    cudaEventDestroy(launch_begin);
    cudaEventDestroy(launch_end);
}


void runCPU( float * arr, int n )
{
    // time many multiplication calls and take the average time
    float average_cpu_time = 0;
    clock_t now, then;
    int num_cpu_test = 3;
    float sum = 0;

    printf("Timing CPU implementation…\n");
    for(int i = 0; i < num_cpu_test; ++i) //launch 3 times on CPU
    {
        // timing on CPU
       then = clock();
       sum = cpuReduce(arr, n);
       now = clock();

       // measure the time spent on CPU
       float time = 0;
       time = timeCost(then, now);

       average_cpu_time += time;
    }
    average_cpu_time /= num_cpu_test;
    printf(" done. CPU time cost in second: %f\n", average_cpu_time);

    //if (shouldPrint)
    printf("CPU finding sum number is %.1f\n", sum);
}



int main(int argc, char *argv[])
{
    // create a large workload so we can easily measure the
    // performance difference on CPU and GPU

    // to run this program: ./a.out blockWidth numElements p
    if(argc < 3 || argc > 4) {
       usage();
       return 1;
    } else  if(argc == 3){
          shouldPrint = 0;
    } else if(argv[3][0]=='p'){
          shouldPrint=1;
    } else {
          usage();
          return 1;
    }
  
    //
    int tile_width = atoi(argv[1]);
    if ( ! tile_width || ! isPowerOfTwo(tile_width) )
    {
        printf("Wrong argument passed in for blockWidth!\n");
        usage();
    }
    int n = atoi(argv[2]); //size of 1D input array
    if ( ! n || ceil(n / (float)tile_width) > 262144 )
    {
        printf("Size of input array goes beyond limit!\n");
        usage();
    }
   
    //generate input data from random generator
    float *data = fillArray(n, UPB);
   
    runCUDA( data, n, tile_width );
    runCPU( data, n );   

    //--------------------------------clean up-----------------------------------------------------
    free(data);

   return 0;
}

