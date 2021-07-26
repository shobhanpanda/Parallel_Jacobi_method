/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"
#include "utils.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
    // TODO
    double result[n];
    double temp;
    double curA ;
    for(int i=0;i<n;i++){
        temp = 0;
        for(int j=0;j<n;j++){
            curA = *((A+i*n) + j) ;
            // Inner * for dereferencing
            temp += curA * (*(x+j));
        }
        result[i] = temp;
    }
    // Setting the values in y vector due to pointers are passed-by-value
    for(int i=0;i<n;i++){
        *(y+i) = result[i];
    }
    return;
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
    // TODO
    double result[n];
    double temp;
    double curA ;
    for(int i=0;i<n;i++){
        temp = 0;
        for(int j=0;j<m;j++){
            curA = *((A+i*n) + j) ;
            // Inner * for dereferencing
            temp += curA * (*(x+j));
        }
        result[i] = temp;
    }
    for(int i=0;i<n;i++){
        *(y+i) = result[i];
    }
    return;
}

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
    // TODO
    int iter = 0;
    double error;
    double currA;
    double diag_inv[n][n];
    double off_diag[n][n];
    double x_temp[n];
    double Rx[n];
    double result[n];
    double temp[n*n];

    // Separating the diagonal and off-diagonal matrix
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            currA = *((A+i*n) + j);
            if(i==j){
                diag_inv[i][j] = 1/currA;
                off_diag[i][j] = 0;
            }else{
                diag_inv[i][j] = 0;
                off_diag[i][j] = currA;
            }
            
        }
    }
    
    // Jacobi Method Iterations
    while(iter < max_iter){

        // Calculating R*x
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                temp[i*n+j] = off_diag[i][j];
                // printf("temp [%d]",i+j,)
            }
        }

        matrix_vector_mult(n,temp,x, &Rx[0]);
        //

        //Calculating (b- R*x)
        for(int i=0;i<n;i++){
            x_temp[i] = *(b+i) - Rx[i]; 
        }
        //Calculating D^-1(b- R*x)
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                temp[i*n+j] = diag_inv[i][j];
            }
        }
        
        matrix_vector_mult(n, temp, x_temp, &result[0]);

        //Calculate L2 norm for error
        error = calculate_l2norm( n,x,result);
        if(l2_termination > error){
            for(int i=0;i<n;i++){
                *(x+i) = result[i];
            }
            return;
        }else{
            for(int i=0;i<n;i++){
                *(x+i) = result[i];
            }
        }
        iter++;
    }
}
