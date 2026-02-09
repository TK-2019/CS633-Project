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
    double t1 = MPI_Wtime();
    float * local_data = malloc(nx/px*ny/py*nz/pz*nc*sizeof(float));
    MPI_File fh;
    MPI_Datatype filetype;
    MPI_Status read_status;

    int err = MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(err, error_string, &len);
        fprintf(stderr, "Rank %d: Failed to open binary file %s. MPI Error: %s\n", myrank, input_file, error_string);
        MPI_Abort(MPI_COMM_WORLD, err);
    }

    int num_local_blocks = ny/py * nz/pz;
    int* blocklengths = malloc(num_local_blocks * sizeof(int));
    MPI_Aint* displacements = malloc(num_local_blocks * sizeof(MPI_Aint));

    if (blocklengths == NULL || displacements == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate memory for filetype definition.\n", myrank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    coord crd = get_coord(myrank, px, py, pz);
    int xst = crd.x * nx/px;
    int yst = crd.y * ny/py;
    int zst = crd.z * nz/pz;
    int idx = 0;
    for(int lz = 0; lz < nz/pz; ++lz){
        for(int ly = 0; ly < ny/py; ++ly){
            int st_idx = (zst + lz) * nx * ny + (yst + ly) * nx + xst;
            displacements[idx] = st_idx * nc * sizeof(float);
            blocklengths[idx++] = nx/px * nc;

        }
    }
    assert(idx == num_local_blocks);
    MPI_Type_create_hindexed(num_local_blocks, blocklengths, displacements, MPI_FLOAT, &filetype);
    MPI_Type_commit(&filetype);
    free(blocklengths);
    free(displacements);
    err = MPI_File_set_view(fh, 0, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
     if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(err, error_string, &len);
        fprintf(stderr, "Rank %d: Failed during MPI_File_set_view. MPI Error: %s\n", myrank, error_string);
        MPI_Abort(MPI_COMM_WORLD, err);
    }

    err = MPI_File_read_all(fh, local_data, nx/px*ny/py*nz/pz*nc, MPI_FLOAT, &read_status);
     if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(err, error_string, &len);
        fprintf(stderr, "Rank %d: Failed during MPI_File_read_all. MPI Error: %s\n", myrank, error_string);
        MPI_Abort(MPI_COMM_WORLD, err);
    }
    
    MPI_Type_free(&filetype);
    MPI_File_close(&fh);
    float * local_data_copy = malloc(nx/px*ny/py*nz/pz*nc*sizeof(float));
    for(int x = 0; x < nx/px; ++x){
        for(int y = 0; y < ny/py; ++y){
            for(int z = 0; z < nz/pz; ++z){
                for(int c = 0; c < nc; ++c){
                    int ldata_index = c + x * nc + y * nx/px * nc + z * nx/px * ny/py * nc;
                    int ldata_copy_index = GET_INDEX_4D(x, y, z, c, nx/px, ny/py, nz/pz, nc);
                    local_data_copy[ldata_copy_index] = local_data[ldata_index];
                }
            }
        }
    }

    // swap local_data and local_data_copy
    float * temp = local_data;
    local_data = local_data_copy;
    local_data_copy = temp;
    free(local_data_copy);
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
    double t3 = MPI_Wtime();
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
    double t5 = MPI_Wtime();
    double read_distribute_time = t2 - t1;
    double transfer_time = t3 - t2;
    double compute_time = t4 - t3;
    double total_time = t5 - t1;
    double max_read_distribute_time, max_transfer_time, max_compute_time, max_total_time;
    MPI_Reduce(&read_distribute_time, &max_read_distribute_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&transfer_time, &max_transfer_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
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
            fprintf(outfp, "(%lf, %lf)", global_minimas[c], global_maximas[c]);
            if(c < nc - 1){
                fprintf(outfp, ", ");
            }
        }
        fprintf(outfp, "\n");
        fprintf(outfp, "%lf %lf %lf\n", max_read_distribute_time, max_transfer_time + max_compute_time, max_total_time);
        fclose(outfp);
        free(total_local_minimas);
        free(total_local_maximas);
        free(global_minimas);
        free(global_maximas);
    }
    MPI_Finalize();
    return 0;
}
