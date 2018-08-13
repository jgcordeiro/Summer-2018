// Run this file to test zcurid. See README.txt

#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"
#include "magma_v2.h"      // also includes cublas_v2.h
#include "magma_lapack.h"  // if you need BLAS & LAPACK

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

// Predefine functions
magma_int_t magma_zcurid(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipvt, magma_int_t *jpvt, magmaDoubleComplex *U,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info );
magma_int_t magma_zgeid(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *jpvt, magmaDoubleComplex *VT,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info );
magma_int_t magma_zge2id(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipvt, magma_int_t *jpvt,
    magmaDoubleComplex *WT, magmaDoubleComplex *VT,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info );
#ifdef __cplusplus
}
#endif

// The following functions are taken from example_v2.



// ------------------------------------------------------------
// Replace with your code to initialize the A matrix.
// This simply initializes it to random values.
// Note that A is stored column-wise, not row-wise.
//
// m   - number of rows,    m >= 0.
// n   - number of columns, n >= 0.
// A   - m-by-n array of size lda*n.
// lda - leading dimension of A, lda >= m.
//
// When lda > m, rows (m, ..., lda-1) below the bottom of the matrix are ignored.
// This is helpful for working with sub-matrices, and for aligning the top
// of columns to memory boundaries (or avoiding such alignment).
// Significantly better memory performance is achieved by having the outer loop
// over columns (j), and the inner loop over rows (i), than the reverse.
void zfill_matrix(
    magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda )
{
    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    
    magma_int_t i, j;
    for (j=0; j < n; ++j) {
        for (i=0; i < m; ++i) {
            A(i,j) = MAGMA_Z_MAKE( rand() / ((double) RAND_MAX),    // real part
                                   rand() / ((double) RAND_MAX) );  // imag part
        }
    }
    
    #undef A
}


// ------------------------------------------------------------
// Replace with your code to initialize the dA matrix on the GPU device.
// This simply leverages the CPU version above to initialize it to random values,
// and copies the matrix to the GPU.
void zfill_matrix_gpu(
    magma_int_t m, magma_int_t n, magmaDoubleComplex *dA, magma_int_t ldda,
    magma_queue_t queue )
{
    magmaDoubleComplex *A;
    magma_int_t lda = ldda;
    magma_zmalloc_cpu( &A, m*lda );
    if (A == NULL) {
        fprintf( stderr, "malloc failed\n" );
        return;
    }
    zfill_matrix( m, n, A, lda );
    magma_zsetmatrix( m, n, A, lda, dA, ldda, queue );
    magma_free_cpu( A );
}


// Test zcurid on a random matrix. The original matrix has dimensions M x N. The approximated matrix is K x K.
void cpu_interface( magma_int_t m, magma_int_t n, magma_int_t k )
{
    magmaDoubleComplex *A=NULL, *U=NULL, *work=NULL;
    magma_int_t *ipiv=NULL, *jpiv=NULL;
    double *rwork = NULL;
    magma_int_t lda  = n;
    magma_int_t info = 0;
    magma_int_t lwork = (2*k) + (32*n);

    printf( "Allocating memory...\n" );
    // magma_*malloc_cpu routines for CPU memory are type-safe and align to memory boundaries,
    // but you can use malloc or new if you prefer.
    magma_zmalloc_cpu( &A, lda*(m>n?m:n) );
    magma_zmalloc_cpu( &U, k*k );
    magma_zmalloc_cpu( &work, lwork );
    magma_dmalloc_cpu( &rwork, 5*k );
    magma_imalloc_cpu( &ipiv, k );
    magma_imalloc_cpu( &jpiv, k );
    if (A == NULL || U == NULL || ipiv == NULL || jpiv == NULL
        || work == NULL) {
        fprintf( stderr, "malloc failed\n" );
        goto cleanup;
    }
    
    printf( "Filling matrix...\n" );
    zfill_matrix( m, n, A, lda );
    
    printf( "Calling zcurid...\n" );
    magma_zcurid( m, n, k, A, lda, ipiv, jpiv, U, work, lwork, rwork, &info );

    if (info != 0) {
        fprintf( stderr, "magma_zcurid failed with info=%d\n", info );
    }
    
cleanup:
    magma_free_cpu( A );
    magma_free_cpu( U );
    magma_free_cpu( work );
    magma_free_cpu( ipiv );
    magma_free_cpu( jpiv );
    magma_free_cpu( rwork );
}

/*
TODO: Create a test for zcurid_gpu
void gpu_interface( magma_int_t n, magma_int_t nrhs )
{
    magmaDoubleComplex *dA=NULL, *dX=NULL;
    magma_int_t *ipiv=NULL;
    magma_int_t ldda = magma_roundup( n, 32 );  // round up to multiple of 32 for best GPU performance
    magma_int_t lddx = ldda;
    magma_int_t info = 0;
    magma_queue_t queue=NULL;
    
    // magma_*malloc routines for GPU memory are type-safe,
    // but you can use cudaMalloc if you prefer.
    magma_zmalloc( &dA, ldda*n );
    magma_zmalloc( &dX, lddx*nrhs );
    magma_imalloc_cpu( &ipiv, n );  // ipiv always on CPU
    if (dA == NULL || dX == NULL || ipiv == NULL) {
        fprintf( stderr, "malloc failed\n" );
        goto cleanup;
    }
    
    magma_int_t dev = 0;
    magma_queue_create( dev, &queue );
    
    // Replace these with your code to initialize A and X
    zfill_matrix_gpu( n, n, dA, ldda, queue );
    zfill_rhs_gpu( n, nrhs, dX, lddx, queue );
    
    magma_zgesv_gpu( n, 1, dA, ldda, ipiv, dX, ldda, &info );
    if (info != 0) {
        fprintf( stderr, "magma_zgesv_gpu failed with info=%d\n", info );
    }
    
    // TODO: use result in dX
    
cleanup:
    magma_queue_destroy( queue );
    magma_free( dA );
    magma_free( dX );
    magma_free_cpu( ipiv );
}
*/


// ------------------------------------------------------------
int main( int argc, char** argv )
{
    magma_init();
    
    magma_int_t m = 500, n = 1000, k = 50;
    
    printf( "using MAGMA CPU interface\n" );
    cpu_interface( m, n, k );

    /*
    printf( "using MAGMA GPU interface\n" );
    gpu_interface( n, nrhs );
    */

    magma_finalize();
    return 0;
}
