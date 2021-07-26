/**
 * @file    utils.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements common utility/helper functions.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "utils.h"
#include <iostream>
#include <cmath>
#include <mpi.h>
/*********************************************************************
 *                 Implement your own functions here                 *
 *********************************************************************/

// ...
double calculate_l2norm(int n, double *y, double *yhat)
{
    double error = 0;
    for (int i = 0; i < n; i++)
    {
        //Incase we need L2 norm for error
        // error +=  std::pow((*(y+i) - *(yhat+i)),2);
        error += std::abs((*(y + i) - *(yhat + i)));
    }
    return error;
}
// Gathering L2 norm at 0 processor
void mpi_calculate_l2norm(int n, double *y, double *yhat, double * norm, MPI_Comm comm)
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

    // 0 will recieve local L2 norm from all processors
    for (int i = 0; i < root_p - 1; i++)
    {

        if (dest_ranks[i] == rank)
        {
            double error = 0;
            for (int j = 0; j < arrSize;j++)
            {
                error += std::abs((*(y + j) - *(yhat + j)));
            }
            MPI_Send(&error, 1, MPI_DOUBLE, 0, 111, MPI_COMM_WORLD);
            return;
        }
    }

    if (rank == 0)
    {
        double error = 0;
        for (int i = 0; i < lastArrsize; i++)
        {
            error += std::abs((*(y + i) - *(yhat + i)));
        }
        double temp;
        for (int i = 0; i < root_p - 1; i++)
        {
            MPI_Status stat;
            temp = 0;
            MPI_Recv(&temp, 1, MPI_DOUBLE, dest_ranks[i], 111, MPI_COMM_WORLD, &stat);

            error += temp;
        }
        // Set L2 norm in the reference
        *norm = error;
    }
}