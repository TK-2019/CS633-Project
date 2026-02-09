#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "mpi.h"
#define MAX_FILE_NAME 256

#define GET_INDEX_3D(c1, c2, c3, d1, d2, d3) \
    ((c3) + (d3) * ((c2) + (d2) * (c1)))
#define GET_INDEX_4D(c1, c2, c3, c4, d1, d2, d3, d4) \
    ((c4) + (d4) * ((c3) + (d3) * ((c2) + (d2) * (c1))))

#define GET_INDEX_2D(c1, c2, d1, d2) \
    ((c2) + (d2) * (c1))

typedef struct{
    int x;
    int y;
    int z;
} coord;

coord get_coord(int pid, int px, int py, int pz)
{
    coord c;
    c.z = pid % pz;
    c.y = (pid / pz) % py;
    c.x = pid / (pz * py);
    return c;
}

int get_pid(coord c, int px, int py, int pz)
{
    return c.z + c.y * pz + c.x * pz * py;
}

int main(int argc, char *argv[])
{
    int myrank, size; 
    MPI_Status status;

    MPI_Init(&argc, &argv);


    MPI_Comm_rank(MPI_COMM_WORLD, &myrank) ;
    MPI_Comm_size(MPI_COMM_WORLD, &size);



    char input_file[MAX_FILE_NAME], output_file[MAX_FILE_NAME];
    strcpy(input_file, argv[1]);
    int px = atoi(argv[2]);
    int py = atoi(argv[3]);
    int pz = atoi(argv[4]);
    assert(px*py*pz == size);
    int nx = atoi(argv[5]);
    int ny = atoi(argv[6]);
    int nz = atoi(argv[7]);
    int nc = atoi(argv[8]);
    strcpy(output_file, argv[9]);
    int count = nx*ny*nz;
    float * local_data = malloc(nx/px*ny/py*nz/pz*nc*sizeof(float));
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before starting timer
    double t1 = MPI_Wtime();
    if(myrank == 0){
        // read the data from the binary file
        FILE * fp = fopen(input_file, "rb");
        if (fp == NULL) {
            fprintf(stderr, "Error opening file %s\n", input_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        float *rearranged_data = (float*)malloc(px * py * pz * (nx/px) * (ny/py) * (nz/pz) * nc * sizeof(float));
        if (rearranged_data == NULL) {
            fprintf(stderr, "Memory allocation failed for rearranged_data array\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        float value;
        for(int row = 0; row < count; ++row){
            for(int col = 0; col < nc; ++col){
                int x = row % nx;
                int y = (row / nx) % ny;
                int z = row / (nx*ny);
                int c = col;

                int px_index = x / (nx/px);
                int py_index = y / (ny/py);
                int pz_index = z / (nz/pz);
                int local_x = x % (nx/px);
                int local_y = y % (ny/py);
                int local_z = z % (nz/pz);
                int lnx = nx/px, lny = ny/py, lnz = nz/pz;
                int rearr_idx = c + 
                                nc * (local_z + 
                                    lnz * (local_y + 
                                        lny * (local_x + 
                                            lnx * (pz_index + 
                                                pz * (py_index + 
                                                    py * px_index)))));

                fread(&value, sizeof(float), 1, fp);
                rearranged_data[rearr_idx] = value;
            }
        }
        fclose(fp);
        // Now scatter so each process gets its own data
        MPI_Scatter(rearranged_data, nx/px*ny/py*nz/pz*nc, MPI_FLOAT, local_data, nx/px*ny/py*nz/pz*nc, MPI_FLOAT, 0, MPI_COMM_WORLD);
        // printf("scatter done\n");
        fflush(stdout);

        free(rearranged_data);
    }
    else{
        // receive the data
        MPI_Scatter(NULL, nx/px*ny/py*nz/pz*nc, MPI_FLOAT, local_data, nx/px*ny/py*nz/pz*nc, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    // printf("myrank :%d, local_data[0] :%f\n", myrank, local_data[0]);
    // fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before stopping timer and starting next phase
    double t2 = MPI_Wtime();
    // Now each process has its own local_data
    // We need to find the count of local minimas and maximas
    // we have to communicate the border values to the neighbours
    float * left = malloc(ny/py*nz/pz*nc*sizeof(float));
    float * right = malloc(ny/py*nz/pz*nc*sizeof(float));
    float * top = malloc(nx/px*nz/pz*nc*sizeof(float));
    float * bottom = malloc(nx/px*nz/pz*nc*sizeof(float));
    float * front = malloc(nx/px*ny/py*nc*sizeof(float));
    float * back = malloc(nx/px*ny/py*nc*sizeof(float));
    float * recv_left = malloc(ny/py*nz/pz*nc*sizeof(float));
    float * recv_right = malloc(ny/py*nz/pz*nc*sizeof(float));
    float * recv_top = malloc(nx/px*nz/pz*nc*sizeof(float));
    float * recv_bottom = malloc(nx/px*nz/pz*nc*sizeof(float));
    float * recv_front = malloc(nx/px*ny/py*nc*sizeof(float));
    float * recv_back = malloc(nx/px*ny/py*nc*sizeof(float));
    // populate left
    for(int y = 0; y < ny/py; ++y){
        for(int z = 0; z < nz/pz; ++z){
            for(int c = 0; c < nc; ++c){
                left[GET_INDEX_3D(y, z, c, ny/py, nz/pz, nc)] = 
                    local_data[GET_INDEX_4D(0, y, z, c, nx/px, ny/py, nz/pz, nc)];
            }
        }
    }
    // populate right
    for(int y = 0; y < ny/py; ++y){
        for(int z = 0; z < nz/pz; ++z){
            for(int c = 0; c < nc; ++c){
                right[GET_INDEX_3D(y, z, c, ny/py, nz/pz, nc)] = 
                    local_data[GET_INDEX_4D(nx/px-1, y, z, c, nx/px, ny/py, nz/pz, nc)];
            }
        }
    }
    // populate top
    for(int x = 0; x < nx/px; ++x){
        for(int z = 0; z < nz/pz; ++z){
            for(int c = 0; c < nc; ++c){
                top[GET_INDEX_3D(x, z, c, nx/px, nz/pz, nc)] = 
                    local_data[GET_INDEX_4D(x, 0, z, c, nx/px, ny/py, nz/pz, nc)];
            }
        }
    }
    // populate bottom
    for(int x = 0; x < nx/px; ++x){
        for(int z = 0; z < nz/pz; ++z){
            for(int c = 0; c < nc; ++c){
                bottom[GET_INDEX_3D(x, z, c, nx/px, nz/pz, nc)] = 
                    local_data[GET_INDEX_4D(x, ny/py-1, z, c, nx/px, ny/py, nz/pz, nc)];
            }
        }
    }
    // populate front
    for(int x = 0; x < nx/px; ++x){
        for(int y = 0; y < ny/py; ++y){
            for(int c = 0; c < nc; ++c){
                front[GET_INDEX_3D(x, y, c, nx/px, ny/py, nc)] = 
                    local_data[GET_INDEX_4D(x, y, 0, c, nx/px, ny/py, nz/pz, nc)];
            }
        }
    }
    // populate back
    for(int x = 0; x < nx/px; ++x){
        for(int y = 0; y < ny/py; ++y){
            for(int c = 0; c < nc; ++c){
                back[GET_INDEX_3D(x, y, c, nx/px, ny/py, nc)] = 
                    local_data[GET_INDEX_4D(x, y, nz/pz-1, c, nx/px, ny/py, nz/pz, nc)];
            }
        }
    }
    double t3 = MPI_Wtime();
    coord crd = get_coord(myrank, px, py, pz);
    fflush(stdout);
    // sending data
    MPI_Request send_requests[6], recv_requests[6];
    int send_count = 0, recv_count = 0;
    
    if(crd.x > 0){
        MPI_Isend(left, ny/py*nz/pz*nc, MPI_FLOAT, get_pid((coord){crd.x-1, crd.y, crd.z}, px, py, pz), 0, MPI_COMM_WORLD, &send_requests[send_count++]);
        MPI_Irecv(recv_left, ny/py*nz/pz*nc, MPI_FLOAT, get_pid((coord){crd.x-1, crd.y, crd.z}, px, py, pz), 0, MPI_COMM_WORLD, &recv_requests[recv_count++]);
    }
    if(crd.x < px-1){
        MPI_Isend(right, ny/py*nz/pz*nc, MPI_FLOAT, get_pid((coord){crd.x+1, crd.y, crd.z}, px, py, pz), 0, MPI_COMM_WORLD, &send_requests[send_count++]);
        MPI_Irecv(recv_right, ny/py*nz/pz*nc, MPI_FLOAT, get_pid((coord){crd.x+1, crd.y, crd.z}, px, py, pz), 0, MPI_COMM_WORLD, &recv_requests[recv_count++]);
    }
    if(crd.y > 0){
        MPI_Isend(top, nx/px*nz/pz*nc, MPI_FLOAT, get_pid((coord){crd.x, crd.y-1, crd.z}, px, py, pz), 0, MPI_COMM_WORLD, &send_requests[send_count++]);
        MPI_Irecv(recv_top, nx/px*nz/pz*nc, MPI_FLOAT, get_pid((coord){crd.x, crd.y-1, crd.z}, px, py, pz), 0, MPI_COMM_WORLD, &recv_requests[recv_count++]);
    }
    if(crd.y < py-1){
        MPI_Isend(bottom, nx/px*nz/pz*nc, MPI_FLOAT, get_pid((coord){crd.x, crd.y+1, crd.z}, px, py, pz), 0, MPI_COMM_WORLD, &send_requests[send_count++]);
        MPI_Irecv(recv_bottom, nx/px*nz/pz*nc, MPI_FLOAT, get_pid((coord){crd.x, crd.y+1, crd.z}, px, py, pz), 0, MPI_COMM_WORLD, &recv_requests[recv_count++]);
    }
    if(crd.z > 0){
        MPI_Isend(front, nx/px*ny/py*nc, MPI_FLOAT, get_pid((coord){crd.x, crd.y, crd.z-1}, px, py, pz), 0, MPI_COMM_WORLD, &send_requests[send_count++]);
        MPI_Irecv(recv_front, nx/px*ny/py*nc, MPI_FLOAT, get_pid((coord){crd.x, crd.y, crd.z-1}, px, py, pz), 0, MPI_COMM_WORLD, &recv_requests[recv_count++]);
    }
    if(crd.z < pz-1){
        MPI_Isend(back, nx/px*ny/py*nc, MPI_FLOAT, get_pid((coord){crd.x, crd.y, crd.z+1}, px, py, pz), 0, MPI_COMM_WORLD, &send_requests[send_count++]);
        MPI_Irecv(recv_back, nx/px*ny/py*nc, MPI_FLOAT, get_pid((coord){crd.x, crd.y, crd.z+1}, px, py, pz), 0, MPI_COMM_WORLD, &recv_requests[recv_count++]);
    }
    
    // Wait for all sends and receives to complete
    MPI_Waitall(send_count, send_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(recv_count, recv_requests, MPI_STATUSES_IGNORE);

    // Now we have the border values
    // Now we need to find the local minimas and maximas
    int * local_min_counts = malloc(nc * sizeof(int)), * local_max_counts = malloc(nc * sizeof(int));
    memset(local_min_counts, 0, nc * sizeof(int));
    memset(local_max_counts, 0, nc * sizeof(int));
    for(int c = 0; c < nc; ++c){
        for(int x = 0; x < nx/px; ++x){
            for(int y = 0; y < ny/py; ++y){
                for(int z = 0; z < nz/pz; ++z){
                    int is_min = 1;
                    int is_max = 1;
                    // add other conditions
                    float value = local_data[GET_INDEX_4D(x, y, z, c, nx/px, ny/py, nz/pz, nc)];
                    if(x > 0 && value >= local_data[GET_INDEX_4D(x-1, y, z, c, nx/px, ny/py, nz/pz, nc)]) is_min = 0;
                    else if(x == 0 && crd.x > 0 && value >= recv_left[GET_INDEX_3D(y, z, c, ny/py, nz/pz, nc)]) is_min = 0;
                    if(x < nx/px-1 && value >= local_data[GET_INDEX_4D(x+1, y, z, c, nx/px, ny/py, nz/pz, nc)]) is_min = 0;
                    else if(x == nx/px-1 && crd.x < px - 1 && value >= recv_right[GET_INDEX_3D(y, z, c, ny/py, nz/pz, nc)]) is_min = 0;
                    if(y > 0 && value >= local_data[GET_INDEX_4D(x, y-1, z, c, nx/px, ny/py, nz/pz, nc)]) is_min = 0;
                    else if(y == 0 && crd.y > 0 && value >= recv_top[GET_INDEX_3D(x, z, c, nx/px, nz/pz, nc)]) is_min = 0;
                    if(y < ny/py-1 && value >= local_data[GET_INDEX_4D(x, y+1, z, c, nx/px, ny/py, nz/pz, nc)]) is_min = 0;
                    else if(y == ny/py-1 && crd.y < py - 1 && value >= recv_bottom[GET_INDEX_3D(x, z, c, nx/px, nz/pz, nc)]) is_min = 0;
                    if(z > 0 && value >= local_data[GET_INDEX_4D(x, y, z-1, c, nx/px, ny/py, nz/pz, nc)]) is_min = 0;
                    else if(z == 0 && crd.z > 0 && value >= recv_front[GET_INDEX_3D(x, y, c, nx/px, ny/py, nc)]) is_min = 0;
                    if(z < nz/pz-1 && value >= local_data[GET_INDEX_4D(x, y, z+1, c, nx/px, ny/py, nz/pz, nc)]) is_min = 0;
                    else if(z == nz/pz-1 && crd.z < pz - 1 && value >= recv_back[GET_INDEX_3D(x, y, c, nx/px, ny/py, nc)]) is_min = 0;
                    if(is_min){
                        local_min_counts[c]++;
                    }
                    if(x > 0 && value <= local_data[GET_INDEX_4D(x-1, y, z, c, nx/px, ny/py, nz/pz, nc)]) is_max = 0;
                    else if(x == 0 && crd.x > 0 && value <= recv_left[GET_INDEX_3D(y, z, c, ny/py, nz/pz, nc)]) is_max = 0;
                    if(x < nx/px-1 && value <= local_data[GET_INDEX_4D(x+1, y, z, c, nx/px, ny/py, nz/pz, nc)]) is_max = 0;
                    else if(x == nx/px-1 && crd.x < px - 1 && value <= recv_right[GET_INDEX_3D(y, z, c, ny/py, nz/pz, nc)]) is_max = 0;
                    if(y > 0 && value <= local_data[GET_INDEX_4D(x, y-1, z, c, nx/px, ny/py, nz/pz, nc)]) is_max = 0;
                    else if(y == 0 && crd.y > 0 && value <= recv_top[GET_INDEX_3D(x, z, c, nx/px, nz/pz, nc)]) is_max = 0;
                    if(y < ny/py-1 && value <= local_data[GET_INDEX_4D(x, y+1, z, c, nx/px, ny/py, nz/pz, nc)]) is_max = 0;
                    else if(y == ny/py-1 && crd.y < py - 1 && value <= recv_bottom[GET_INDEX_3D(x, z, c, nx/px, nz/pz, nc)]) is_max = 0;
                    if(z > 0 && value <= local_data[GET_INDEX_4D(x, y, z-1, c, nx/px, ny/py, nz/pz, nc)]) is_max = 0;
                    else if(z == 0 && crd.z > 0 && value <= recv_front[GET_INDEX_3D(x, y, c, nx/px, ny/py, nc)]) is_max = 0;
                    if(z < nz/pz-1 && value <= local_data[GET_INDEX_4D(x, y, z+1, c, nx/px, ny/py, nz/pz, nc)]) is_max = 0;
                    else if(z == nz/pz-1 && crd.z < pz - 1 && value <= recv_back[GET_INDEX_3D(x, y, c, nx/px, ny/py, nc)]) is_max = 0;
                    if(is_max){
                        local_max_counts[c]++;
                    }
                }
            }
        }
    }
    float * global_mins = malloc(nc * sizeof(float)),  * global_maxs = malloc(nc * sizeof(float));
    for(int c = 0; c < nc; ++c){
        global_mins[c] = local_data[GET_INDEX_4D(0, 0, 0, c, nx/px, ny/py, nz/pz, nc)];
        global_maxs[c] = local_data[GET_INDEX_4D(0, 0, 0, c, nx/px, ny/py, nz/pz, nc)];
    }
    // Now we need to find the global minimas and maximas inside local_data
    for(int c = 0; c < nc; ++c){
        for(int x = 0; x < nx/px; ++x){
            for(int y = 0; y < ny/py; ++y){
                for(int z = 0; z < nz/pz; ++z){
                    if(local_data[GET_INDEX_4D(x, y, z, c, nx/px, ny/py, nz/pz, nc)] < global_mins[c]){
                        global_mins[c] = local_data[GET_INDEX_4D(x, y, z, c, nx/px, ny/py, nz/pz, nc)];
                    }
                    if(local_data[GET_INDEX_4D(x, y, z, c, nx/px, ny/py, nz/pz, nc)] > global_maxs[c]){
                        global_maxs[c] = local_data[GET_INDEX_4D(x, y, z, c, nx/px, ny/py, nz/pz, nc)];
                    }
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before stopping timer and starting next phase
    double t4 = MPI_Wtime();
    // Now we do mpi reduce so that the process 0 has the global max/min and sum of localmins/localmaxs
    int * total_local_minimas, * total_local_maximas;
    float * global_minimas, * global_maximas;
    if(myrank == 0){
        total_local_minimas = malloc(nc * sizeof(int));
        total_local_maximas = malloc(nc * sizeof(int));
        global_minimas = malloc(nc * sizeof(float));
        global_maximas = malloc(nc * sizeof(float));
    }
    MPI_Reduce(local_min_counts, total_local_minimas, nc, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max_counts, total_local_maximas, nc, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(global_mins, global_minimas, nc, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(global_maxs, global_maximas, nc, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before stopping timer
    double t5 = MPI_Wtime();
    double read_distribute_time = t2 - t1;
    double send_recv_time = t3 - t2;
    double compute_time = t4 - t3;
    double total_time = t5 - t1;
    double max_read_distribute_time, max_send_recv_time, max_compute_time, max_total_time;
    MPI_Reduce(&read_distribute_time, &max_read_distribute_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&send_recv_time, &max_send_recv_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &max_compute_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(myrank == 0){
        FILE * outfp = fopen(output_file, "w");
        if (outfp == NULL) {
            fprintf(stderr, "Error opening file %s\n", output_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for(int c = 0; c < nc; ++c){
            fprintf(outfp, "(%d, %d)", total_local_minimas[c], total_local_maximas[c]);
            if(c < nc - 1){
                fprintf(outfp, ", ");
            }
        }
        fprintf(outfp, "\n");
        for(int c = 0; c < nc; ++c){
            fprintf(outfp, "(%f, %f)", global_minimas[c], global_maximas[c]);
            if(c < nc - 1){
                fprintf(outfp, ", ");
            }
        }
        fprintf(outfp, "\n");
        fprintf(outfp, "%lf %lf %lf\n", max_read_distribute_time, max_send_recv_time + max_compute_time, max_total_time);
        // fprintf(outfp, "Time taken for read and distribute: %lf\n", max_read_distribute_time);
        // fprintf(outfp, "Time taken for send and receive: %lf\n", max_send_recv_time);
        // fprintf(outfp, "Time taken for compute: %lf\n", max_compute_time);
        // fprintf(outfp, "Time taken for total: %lf\n", max_total_time);
        fclose(outfp);
        free(total_local_minimas);
        free(total_local_maximas);
        free(global_minimas);
        free(global_maximas);
    }
    MPI_Finalize();
    return 0;
}
