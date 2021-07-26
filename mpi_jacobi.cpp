/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>

/*
 * TODO: Implement your solutions here
 */

void distribute_vector(const int n, double *scatter_values, double **local_vector, MPI_Comm comm, int source_rank)
{

  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // As number of processor is n, we have to distribute to sqrt(n)
  const int NDIM = 2;
  int dim = std ::sqrt(size);
  int coords[NDIM];

  int sendRank;
  double *recArr;
  int bound;

  coords[1] = source_rank;
  int dest_ranks[dim - 1];
  for (int i = 1; i < dim; i++)
  {
    coords[0] = i;
    MPI_Cart_rank(comm, coords, &sendRank);
    dest_ranks[i - 1] = sendRank;
  }

  if (rank == source_rank)
  {

    int arrSize = n / dim;
    //In case of n is not divisble, then get the largest array size.
    int lastArrsize = n - (arrSize * (dim - 1));

    bound = lastArrsize;
    recArr = new double[bound];

    //Give maximum elements to source
    for (int i = 0; i < lastArrsize; i++)
    {
      recArr[i] = scatter_values[i];
    }

    // Send to all processor in same column of processor 0
    for (int i = 0; i < dim - 1; i++)
    {
      MPI_Send(&arrSize, 1, MPI_INT, dest_ranks[i], 111, MPI_COMM_WORLD);                                                //Send size of array first
      MPI_Send((scatter_values + arrSize * (i) + lastArrsize), arrSize, MPI_DOUBLE, dest_ranks[i], 111, MPI_COMM_WORLD); // Send array
    }
  }
  else
  {
    //Checking if recieve rank is in array
    //Check for better way to decide whether rank is in dest_rank or not
    for (int i = 0; i < dim - 1; i++)
    {
      if (dest_ranks[i] == rank)
      {
        MPI_Status stat;
        MPI_Recv(&bound, 1, MPI_INT, source_rank, 111, MPI_COMM_WORLD, &stat); //Receive the size first
        recArr = new double[bound];
        MPI_Recv(recArr, bound, MPI_DOUBLE, source_rank, 111, MPI_COMM_WORLD, &stat); //Recieve the array
      }
      else
      {
        continue;
      }
    }
  }

  *local_vector = recArr;
}

// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double *local_vector, double *output_vector, MPI_Comm comm)
{

  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // As number of processor is n, we have to distribute to sqrt(n)
  const int NDIM = 2;
  int dim = n;
  int root_p = std ::sqrt(size);
  int coords[NDIM];

  coords[1] = 0;
  int dest_ranks[root_p - 1];
  int sendRank;
  for (int i = 1; i < root_p; i++)
  {
    coords[0] = i;
    MPI_Cart_rank(comm, coords, &sendRank);
    dest_ranks[i - 1] = sendRank;
  }
  double *recArr;
  int arrSize = dim / root_p;
  int lastArrsize = dim - (arrSize * (root_p - 1));

  // 0 will recieve from all processors
  for (int i = 0; i < root_p - 1; i++)
  {

    if (dest_ranks[i] == rank)
    {
      MPI_Send(local_vector, arrSize, MPI_DOUBLE, 0, 111, MPI_COMM_WORLD);
      return;
    }
  }
  //Gather the elements at 0 processor
  if (rank == 0)
  {
    recArr = new double[dim];
    double *temp;
    for (int i = 0; i < lastArrsize; i++)
    {
      recArr[i] = local_vector[i];
    }
    for (int i = 0; i < root_p - 1; i++)
    {
      MPI_Status stat;
      temp = new double[arrSize];
      MPI_Recv(temp, arrSize, MPI_DOUBLE, dest_ranks[i], 111, MPI_COMM_WORLD, &stat);

      for (int j = 0; j < arrSize; j++)
      {
        recArr[lastArrsize + arrSize * (i) + j] = temp[j];
      }
    }
    //Setting the values in output vector
    for (int i = 0; i < dim; i++)
    {
      output_vector[i] = recArr[i];
    }
  }
}

void distribute_matrix(const int n, double *input_matrix, double **local_matrix, MPI_Comm comm)
{

  int dim = n;
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  int root_p = sqrt(size);
  int arrSize = dim / root_p;
  int lastArrsize = n - (arrSize * (root_p - 1));

  int sendRank;
  double *recArr;
  int bound;
  int coords[2];

  //Horizontal Destinations
  coords[0] = 0;
  int dest_ranks[root_p - 1];
  for (int i = 1; i < root_p; i++)
  {
    coords[1] = i;
    MPI_Cart_rank(comm, coords, &sendRank);
    dest_ranks[i - 1] = sendRank;
  }
  //Vertical Destinations
  coords[1] = 0;
  int vdest_ranks[dim - 1];
  for (int v = 1; v < root_p; v++)
  {
    coords[0] = v;
    MPI_Cart_rank(comm, coords, &sendRank);
    vdest_ranks[v - 1] = sendRank;
  }

  if (rank == 0)
  {

    recArr = new double[lastArrsize * dim];

    //Taking out the columns for 0 processor
    for (int j = 0; j < lastArrsize; j++)
    {
      for (int k = 0; k < dim; k++)
      {
        recArr[k * lastArrsize + j] = input_matrix[k * dim + j];
      }
    }

    //Distribute the columns to 1st row of processors
    int sbound = arrSize * dim;
    double *temp;

    for (int i = 1; i < root_p; i++)
    {
      temp = new double[sbound];
      //Taking out the rows for each processor
      for (int j = 0; j < arrSize; j++)
      {
        for (int k = 0; k < dim; k++)
        {
          temp[k * arrSize + j] = input_matrix[k * dim + (i - 1) * (arrSize) + lastArrsize + j];
        }
      }

      MPI_Send(&sbound, 1, MPI_INT, dest_ranks[i - 1], 111, MPI_COMM_WORLD);      //Send size of array first
      MPI_Send(temp, sbound, MPI_DOUBLE, dest_ranks[i - 1], 111, MPI_COMM_WORLD); // Send array
    }
    //Distribute the rows to 1st column of processors
    sbound = lastArrsize * arrSize;
    int begin = lastArrsize * lastArrsize;
    for (int i = 0; i < root_p - 1; i++)
    {
      MPI_Send(&sbound, 1, MPI_INT, vdest_ranks[i], 111, MPI_COMM_WORLD);                               //Send size of array first
      MPI_Send((recArr + begin + i * sbound), sbound, MPI_DOUBLE, vdest_ranks[i], 111, MPI_COMM_WORLD); // Send array
    }

    temp = new double[lastArrsize * lastArrsize];
    for (int ind = 0; ind < lastArrsize * lastArrsize; ind++)
    {
      temp[ind] = recArr[ind];
    }
    recArr = temp;
  }
  else
  {
    //Receive data from 0 for processor in 1st column
    for (int d = 0; d < root_p - 1; d++)
    {
      if (vdest_ranks[d] == rank)
      {
        MPI_Status stat;
        MPI_Recv(&bound, 1, MPI_INT, 0, 111, MPI_COMM_WORLD, &stat); //Receive the size first
        recArr = new double[bound];
        MPI_Recv(recArr, bound, MPI_DOUBLE, 0, 111, MPI_COMM_WORLD, &stat); //Recieve the array
      }
      else
        continue;
    }

    for (int d = 0; d < root_p - 1; d++)
    {
      //Get the vertical destination rank
      coords[1] = dest_ranks[d];
      for (int v = 1; v < root_p; v++)
      {
        coords[0] = v;
        MPI_Cart_rank(comm, coords, &sendRank);
        vdest_ranks[v - 1] = sendRank;
      }

      if (dest_ranks[d] == rank)
      {
        MPI_Status stat;
        MPI_Recv(&bound, 1, MPI_INT, 0, 111, MPI_COMM_WORLD, &stat); //Receive the size first
        recArr = new double[bound];
        MPI_Recv(recArr, bound, MPI_DOUBLE, 0, 111, MPI_COMM_WORLD, &stat); //Recieve the array
        //Send the sub-matrix to respective processors
        int sbound = arrSize * arrSize;
        int begin = lastArrsize * arrSize;
        for (int i = 0; i < root_p - 1; i++)
        {
          MPI_Send(&sbound, 1, MPI_INT, vdest_ranks[i], 111, MPI_COMM_WORLD);                               //Send size of array first
          MPI_Send((recArr + begin + i * sbound), sbound, MPI_DOUBLE, vdest_ranks[i], 111, MPI_COMM_WORLD); // Send array
        }

        double temp[begin];

        for (int ind = 0; ind < begin; ind++)
        {
          temp[ind] = recArr[ind];
        }
        recArr = new double[begin];
        //Setting local matrix values
        for (int i = 0; i < begin; i++)
        {
          recArr[i] = temp[i];
        }
      }
      else
      {
        //Receive the sub-matrix in each processor
        for (int e = 0; e < root_p - 1; e++)
        {
          if (vdest_ranks[e] == rank)
          {
            MPI_Status stat;
            MPI_Recv(&bound, 1, MPI_INT, dest_ranks[d], 111, MPI_COMM_WORLD, &stat); //Receive the size first
            recArr = new double[bound];
            MPI_Recv(recArr, bound, MPI_DOUBLE, dest_ranks[d], 111, MPI_COMM_WORLD, &stat); //Recieve the array
          }
          else
            continue;
        }
      }
    }
  }
  //Setting local matrix values
  *local_matrix = recArr;
}

void transpose_bcast_vector(const int n, double *col_vector, double *row_vector, MPI_Comm comm)
{
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  const int NDIM = 2;
  int dim = std::sqrt(size);
  int coords[NDIM];
  int col_size;
  int row_size;

  // Get the rank in cartesian coordinates
  MPI_Cart_coords(comm, rank, NDIM, coords);

  //create the size of data to receive
  col_size = int(floor(double(n) / double(dim)));
  row_size = col_size;

  if (coords[1] == 0)
  {
    col_size = n - col_size * (dim - 1);
  }
  if (coords[0] == 0)
  {
    row_size = n - row_size * (dim - 1);
  }

  //send data in first column to diagonals
  int target_coord[NDIM];
  target_coord[0] = coords[0];
  target_coord[1] = coords[0];

  // diagonal elements should receive from column 0;
  int diag_receive_coord[NDIM];
  diag_receive_coord[0] = coords[0];
  diag_receive_coord[1] = 0;

  int target_rank;
  int diag_receive_rank;
  MPI_Cart_rank(comm, target_coord, &target_rank);
  MPI_Cart_rank(comm, diag_receive_coord, &diag_receive_rank);

  if (coords[1] == 0)
  {
    if (coords[0] == 0)
    {
      for (int i = 0; i < col_size; i++)
      {
        row_vector[i] = col_vector[i];
      }
    }
    else
    {
      MPI_Send(col_vector, row_size, MPI_DOUBLE, target_rank, 111, MPI_COMM_WORLD);
    }
  }
  else if (coords[0] == coords[1])
  {
    MPI_Status stat;
    MPI_Recv(row_vector, row_size, MPI_DOUBLE, diag_receive_rank, 111, MPI_COMM_WORLD, &stat);
  }

  //Create Temporary Communicator with the columns as the color
  int color = coords[1];
  int column = coords[1];
  int color_source = column;

  //Broadcast along the columns of the processors
  int key = (coords[0] - coords[1]);
  key = (dim + key) % dim;

  MPI_Comm column_comm;
  MPI_Comm_split(comm, color, key, &column_comm);

  int msg_size;
  if (coords[0] == coords[1])
  {
    msg_size = row_size;
  }
  MPI_Bcast(&msg_size, 1, MPI_INT, 0, column_comm);
  MPI_Bcast(row_vector, col_size, MPI_DOUBLE, 0, column_comm);
  MPI_Comm_free(&column_comm);
}

void distributed_matrix_vector_mult(const int n, double *local_A, double *local_x, double *local_y, MPI_Comm comm)
{
  int rank, size, row_size, col_size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  const int NDIM = 2;
  int dim = sqrt(size);
  int coords[NDIM];
  MPI_Cart_coords(comm, rank, NDIM, coords);
  int idx_i = coords[0];
  int idx_j = coords[1];

  //create the size of data to receive
  col_size = int(floor(double(n) / double(dim)));
  row_size = col_size;

  if (idx_j == 0)
  {
    col_size = n - col_size * (dim - 1);
  }
  if (idx_i == 0)
  {
    row_size = n - row_size * (dim - 1);
  }

  double trans_x[col_size];

  //Distribute the x values
  MPI_Barrier(comm);
  transpose_bcast_vector(n, local_x, trans_x, comm);
  MPI_Barrier(comm);

  double small_sum = 0;
  double small_y[row_size];

  //Compute partial y values
  for (int i = 0; i < row_size; i++)
  {
    small_sum = 0;
    for (int j = 0; j < col_size; j++)
    {

      small_sum = small_sum + trans_x[j] * local_A[i * col_size + j];
    }
    small_y[i] = small_sum;
  }

  //Reduce the partial y values to the first column
  int color;
  color = coords[0];
  MPI_Comm row_comm;
  MPI_Comm_split(comm, color, rank, &row_comm);
  MPI_Reduce(small_y, local_y, row_size, MPI_DOUBLE, MPI_SUM, 0, row_comm);
  MPI_Comm_free(&row_comm);
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double *local_A, double *local_b, double *local_x,
                        MPI_Comm comm, int max_iter, double l2_termination)
{

  //The diagonal of A is the diagonal of the local A for the diagonal processors
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  //printf("rank: %d \n",rank);

  const int NDIM = 2;
  int dim = std::sqrt(size);
  int coords[NDIM];
  int col_size;
  int row_size;

  // Get the rank in cartesian coordinates
  MPI_Cart_coords(comm, rank, NDIM, coords);

  //Get the size of the row and columns
  col_size = int(floor(double(n) / double(dim)));
  row_size = col_size;

  if (coords[1] == 0)
  {
    col_size = n - col_size * (dim - 1);
  }
  if (coords[0] == 0)
  {
    row_size = n - row_size * (dim - 1);
  }
  //Make R as a copy of A
  double local_D[row_size];
  double R[(col_size * row_size)];
  for (int i = 0; i < row_size; i++)
  {
    for (int j = 0; j < col_size; j++)
    {
      R[i * col_size + j] = local_A[i * col_size + j];
    }
  }

  //Set diagonal entries of R to 0
  if (coords[0] == coords[1])
  {
    for (int i = 0; i < row_size; i++)
    {
      int j = i;
      local_D[i] = local_A[i * col_size + j];
      R[i * col_size + j] = 0;
    }
  }

  //Make row communicator and send diagonal entries to first column
  MPI_Comm row_comm;
  int color = coords[0];
  MPI_Comm_split(comm, color, rank, &row_comm);

  if (coords[0] == coords[1])
  {
    if (rank != 0)
    {
      MPI_Send(local_D, row_size, MPI_DOUBLE, 0, 111, row_comm);
    }
  }
  else if (coords[1] == 0)
  {
    MPI_Status stat;
    MPI_Recv(local_D, row_size, MPI_DOUBLE, coords[0], 111, row_comm, &stat);
  }

  //Free the row communicator
  MPI_Comm_free(&row_comm);

  //Make sure local_x is 0 for the first column of processors
  if (coords[1] == 0)
  {
    for (int i = 0; i < row_size; i++)
    {
      local_x[i] = 0;
    }
  }

  //Begin Jacobi Iteration

  for (int i = 0; i < max_iter; i++)
  {

    double local_w[row_size];

    distributed_matrix_vector_mult(n, R, local_x, local_w, comm);

    if (coords[1] == 0)
    {
      for (int j = 0; j < row_size; j++)
      {
        local_x[j] = (local_b[j] - local_w[j]) / local_D[j];
      }
    }

    distributed_matrix_vector_mult(n, local_A, local_x, local_w, comm);

    double l2norm;
    mpi_calculate_l2norm(n, local_w, local_b, &l2norm, comm);
    double l2vect[1];
    if (rank == 0)
    {
      l2vect[0] = l2norm;
    }
    //if(rank==0){
    //printf("################### i:%d, L2 norm: %f #############\n",i,l2vect[0]);
    //}

    MPI_Bcast(l2vect, 1, MPI_DOUBLE, 0, comm);
    MPI_Barrier(comm);

    if (l2vect[0] <= l2_termination)
    {
      break;
    }
  }
}

// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double *A,
                            double *x, double *y, MPI_Comm comm)
{

  // distribute the array onto local processors!
  double *local_A = NULL;
  double *local_x = NULL;
  distribute_matrix(n, &A[0], &local_A, comm);
  distribute_vector(n, &x[0], &local_x, comm);

  //For testing
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  int root_p = sqrt(size);
  int arrSize = n / root_p;
  int lastArrsize = n - (arrSize * (root_p - 1));

  double *local_y = new double[block_decompose_by_dim(n, comm, 0)];

  //Perform distributed matrix multiplication
  distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

  // gather results back to rank 0

  gather_vector(n, local_y, y, comm);
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double *A, double *b, double *x, MPI_Comm comm,
                int max_iter, double l2_termination)
{

  // distribute the array onto local processors!
  double *local_A = NULL;
  double *local_b = NULL;
  distribute_matrix(n, &A[0], &local_A, comm);
  distribute_vector(n, &b[0], &local_b, comm);

  // allocate local result space
  double *local_x = new double[block_decompose_by_dim(n, comm, 0)];

  // perform distributed jacobi iteration
  distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

  // gather results back to rank 0
  gather_vector(n, local_x, x, comm);
}
