#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h> 
//TODO: Implement dynamic block allocation on the GPU


int string_length(const char *str)
{
    int length = 0;
    while (str[length] != '\0') {
        length++;
    }
    return length;
}

__global__ void naiveSearch(char* sequence, char* pattern, int* seq_len, int* pat_len, int* matches)
{
    int sequence_len = *seq_len;
    int pattern_len = *pat_len;
    int tid = threadIdx.x;
    int chunk_size = (sequence_len )/blockDim.x; //Calculate chunk size for every thread
    int start = tid*chunk_size;
    int end = (tid+1)*chunk_size;
    int local_matches = 0;
    int j; // for iterations

    if (tid == blockDim.x - 1)
        end = sequence_len - pattern_len;


    for(int i = start; i < end; i++) // Iterate through the sequence
    {
        for(j = 0; j < pattern_len; j++)
        {
            if(sequence[i + j] != pattern[j])
                j=pattern_len+10;
        }
        if(j == pattern_len)
        {
            local_matches++; // Increment the local matches
        }
        
    }
    __syncthreads();
    atomicAdd(matches,local_matches); // Add the local matches to the global matches thread-safely
}


int main(int argc, char** argv)
{
    // Open the Sequence and remove any newlines
    FILE *input_file;
    //char ch;
    char* pattern = argv[1];
    int pattern_len = string_length(pattern);
    int file_size = 0;
    int seq_length;
    int thread_num = 1024;
    int matches=0; // Variable to store the integer on the host

    printf("\n\nParallel Version of Naive Pattern Matching Algorithm GPU Accelerated\n");
    // Open the initial sequence for reading

    // input_file = fopen("seq.txt", "r");
    // if (input_file == NULL)
    // {
    //     perror("Error opening the file\n");
    //     return EXIT_FAILURE;
    // }

    // // Open the output file for writing the sequence
    // output_file = fopen("seq_horizontal.txt", "w");
    // if (output_file == NULL)
    // {
    //     perror("Error opening the file\n");
    //     fclose(input_file);
    //     return EXIT_FAILURE;
    // }

    // // Read the sequence from input file and write to output file without newline characters
    // while ((ch = fgetc(input_file)) != EOF)
    // {
    //     if (ch != '\n') {
    //         fputc(ch, output_file);
    //     }
    // }

    // // Close the files
    // rewind(input_file);
    // fclose(input_file);
    // fclose(output_file);
    
    // Open the horizontal sequence for reading
    input_file = fopen("seq_horizontal.txt","r"); 

    //Get the sequence size
    fseek(input_file, 0, SEEK_END);
    file_size = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);

    // Allocate memory for the sequence
    char* sequence = (char *)malloc(file_size*sizeof(char)); 

    if (sequence == NULL) // allocation failed
    {
        perror("Memory allocation for the sequence failed\n");
        fclose(input_file);
        return EXIT_FAILURE;
    }
    else // allocation successful
    {
        fread(sequence, sizeof(char), file_size, input_file);
    }

    seq_length = (int)string_length(sequence);
    fclose(input_file);


    printf("\nSequence Length: %d\nPattern Length: %d\n", seq_length, pattern_len);
    char* gpu_sequence; // Pointer to store the sequence on the device
    char* gpu_pattern; // Pointer to store the pattern on the device
    int *gpu_sequence_length; // Pointer to store the sequence length
    int *gpu_pattern_length; // Pointer to store the pattern length
    int *gpu_matches; // Pointer to store the integer on the device


    cudaEvent_t start, stop; // Events for timing the process when executing on the GPU
    cudaEventCreate(&start); // Create events for timing the process
    cudaEventCreate(&stop);
    cudaEventRecord(start); // Start timer to calculate also the time of memory transfer
    cudaEventSynchronize(start); // Synchronize the event to start the timer
    // Allocate memory on the GPU
    cudaMalloc(&gpu_sequence, seq_length * sizeof(char));
    cudaMalloc(&gpu_pattern, pattern_len * sizeof(char));
    cudaMalloc(&gpu_matches, sizeof(int));
    cudaMalloc(&gpu_sequence_length, sizeof(int));
    cudaMalloc(&gpu_pattern_length, sizeof(int));


    
    // Copy the sequence and pattern to the GPU
    cudaMemcpy(gpu_sequence, sequence, seq_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_pattern, pattern, pattern_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_sequence_length, &seq_length, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_pattern_length, &pattern_len, sizeof(int), cudaMemcpyHostToDevice);
    
    // At the moment one block is utilized and 1024 threads - matches the Tesla V100 architecture
    // TODO: Implement dynamic block allocation on the GPU
    naiveSearch<<<1, thread_num>>>(gpu_sequence, gpu_pattern, gpu_sequence_length, gpu_pattern_length, gpu_matches);
    //print_gpu<<<1,1>>>(gpu_matches);
    cudaMemcpy(&matches, gpu_matches, sizeof(int), cudaMemcpyDeviceToHost);


    cudaEventRecord(stop); // End timer to calculate time taken for pattern matching and memory transfer
    cudaEventSynchronize(stop);

    float time_ms;

    cudaEventElapsedTime(&time_ms, start, stop); // Calculate the time taken

    //TODO: Hold position of matches in an array and print them out

    // Free the memory off the device and host
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(gpu_pattern);
    cudaFree(gpu_sequence);
    cudaFree(gpu_pattern_length);
    cudaFree(gpu_sequence_length);
    cudaFree(gpu_matches);

    free(sequence);
    printf("Matches Found: %d\n", matches); // Print the number of matches
    printf("Time taken for pattern matching %f seconds\n\n", time_ms/1000);
    return 0;
}