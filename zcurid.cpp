/*
	Based on MAGMA 2.4.0

	@date July 2018

	@precisions normal z -> c d s

	@author Jacob Cordeiro
*/

//#include "magma_internal.h"
#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"
#include "magma_v2.h"      // also includes cublas_v2.h
#include "magma_lapack.h"  // if you need BLAS & LAPACK

#define COMPLEX

/***************************************************************************//**
    Purpose
    -------
    magma_get_zgeid_nb computes the optimal block size for the function
    magma_zgeid, when applied to the M-by-N matrix A.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.
Further Details
    ---------------
    TODO

    @ingroup TODO
*******************************************************************************/

magma_int_t magma_get_zgeid_nb( magma_int_t m, magma_int_t n )
{
    return 32; // TODO: Test this
}

/***************************************************************************//**
    Purpose
    -------
    z_geid computes the interpolative decomposition of a complex
    M-by-N matrix A. The rank K ID is written

        A = A(:, JPVT) * conjugate-transpose(V)

    where JPVT is a set of K column indices and V is an N-by-K matrix.

    Note that the routine returns VT = V**H, not V.

    This algorithm is described in:
    Voronin, Sergey. Martinsson, Gunnar. Efficient algorithms
    for cur and interpolative matrix decomposition. 2016.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in]
    k       INTEGER
            The number of rows and columns of the matrix U. min(M,N) >= K >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, TODO.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A. LDA >= max(1,M).

    @param[in,out]
    jpvt    INTEGER array, dimension (K)
            On entry, if JPVT(J).ne.0, the J-th column of A is permuted
            to the front of A*P (a leading column); if JPVT(J)=0,
            the J-th column of A is a free column.
            On exit, if JPVT(J)=H, then the J-th column of C was the
            the H-th column of A.

    @param[out]
    V       COMPLEX_16 array, dimension (N,K)
            A is approximately equal to A(:, JPVT) * (V**H).

    @param[out]
    work   (workspace) COMPLEX_16 array on the GPU, dimension (MAX(1,LWORK))
            On exit, if INFO=0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            For [sd]geid, LWORK >= (N+1)*NB + 2*N;
            for [cz]geid, LWORK >= (N+1)*NB,
            where NB is the optimal blocksize.
    \n
            Note: unlike the CPU interface of this routine, the GPU interface
            does not support a workspace query.

*/
#ifdef COMPLEX
/**

    @param
    rwork   (workspace, for [cz]geid only) DOUBLE PRECISION array, dimension (2*N)

*/
#endif // COMPLEX
/**

    @param[out]
    info    INTEGER
      -     = 0: successful exit.
      -     < 0: if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ---------------
    TODO

    @ingroup TODO
*******************************************************************************/

extern "C" magma_int_t
magma_zgeid(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *jpvt, magmaDoubleComplex *V,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info )
{
#define  A(i, j) (A     + (i) + (j)*(lda ))
#define dA(i, j) (dwork + (i) + (j)*(ldda))
#define A2(i, j) (A2    + (i) + (j)*(lda ))
#define min(m, n) (m < n ? m : n)
#define max(m, n) (m > n ? m : n)

    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;
    const magma_int_t ione = 1;

    magmaDoubleComplex *dwork, *df;

    magma_int_t n_j, ldda, ldwork;
    magma_int_t i, j, jb, nb, sm, sn, fjb, nfxd;
    magma_int_t topbmn, lwkopt=0, lquery;
    magma_int_t minmn;
    // Workspace for zgetri
    magma_int_t lgetriwork = m * magma_get_zgetri_nb( m );

    magmaDoubleComplex *A2 = NULL;
    magma_zmalloc_cpu( &A2, lda*n );
    magmaDoubleComplex *S1 = NULL;
    magma_zmalloc_cpu( &S1, k*n );
    /*
    When created, S11 contains the transpose of S1 = [S11 S12].
    The first k*k elements are transposed in place to form S11.
    The last (n-k)*k elements are transposed and stored in S12.
    */
    magmaDoubleComplex *S11 = NULL;
    magma_zmalloc_cpu( &S11, n*k);
    magmaDoubleComplex *S12 = NULL;
    magma_zmalloc_cpu( &S12, k*(n-k));
    magmaDoubleComplex *T = NULL;
    magma_zmalloc_cpu( &T, k*(n-k));
    magmaDoubleComplex *IT = NULL;
    magma_zmalloc_cpu( &IT, k*n);
    magmaDoubleComplex *P = NULL;
    magma_zmalloc_cpu( &P, k*n);
    magma_int_t *ipvt = NULL;
    magma_imalloc_cpu( &ipvt, k + 1);
    magmaDoubleComplex *getriwork = NULL;
    magma_zmalloc_cpu( &getriwork, lgetriwork);
    magmaDoubleComplex *tau = NULL;
    magma_zmalloc_cpu( &tau, min(m,n));

    ldda = magma_roundup( m, 32 );
    nb = magma_get_zgeid_nb( m, n );

    // dwork and df are used for QR factorization
    ldwork = n*ldda + (n+1)*nb;
    if (MAGMA_SUCCESS != magma_zmalloc( &dwork, ldwork )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    df = dwork + n*ldda;

    *info = 0;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    }

    minmn = min(m,n);
    if (*info == 0) {
        if (minmn == 0) {
            lwkopt = 1;
        } else {
            lwkopt = (n + 1)*nb;
            #ifdef REAL
            lwkopt += 2*n;
            #endif
        }
        work[0] = magma_zmake_lwork( lwkopt );

        if (lwork < lwkopt && ! lquery) {
            *info = -8;
        }
    }

    if (k > minmn) {
        *info = -16;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        goto cleanup;
    } else if (lquery) {
        goto cleanup;
    }

    if (minmn == 0)
        goto cleanup;

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    /*
    Perform partial QR factorization with rank k. Adapted from zgeqp3.
    A is copied to a new matrix A2 so as not to change the original.
    */
    magmablas_zlacpy(MagmaFull, m, n, A, lda, A2, lda, queue);
    nfxd = 0; // TODO: Do we need nfxd?
    if (nfxd < minmn) {
        sm = m - nfxd;
        sn = n - nfxd;
        // sminmn = minmn - nfxd;
        
        if (nb < k) {
            j = nfxd;
            
            // Set the original matrix to the GPU
            magma_zsetmatrix_async( m, sn,
                                    A2(0,j), lda,
                                    dA(0,j), ldda, queue );
        }

        /* Initialize partial column norms. */
        for (j = nfxd; j < n; ++j) {
            rwork[j] = magma_cblas_dznrm2( sm, A2(nfxd,j), ione );
            rwork[n + j] = rwork[j];
        }

        j = nfxd;
        if (nb < k) {
            /* Use blocked code initially. */
            magma_queue_sync( queue );
            
            /* Compute factorization: while loop. */
            topbmn = k - nb;
            while (j < topbmn) {
                jb = min(nb, topbmn - j);
                
                /* Factorize JB columns among columns J:N. */
                n_j = n - j;

                if (j > nfxd) {
                    // Get panel to the CPU
                    magma_zgetmatrix( m-j, jb,
                                      dA(j,j), ldda,
                                      A2(j,j), lda, queue );
                    
                    // Get the rows
                    magma_zgetmatrix( jb, n_j - jb,
                                      dA(j,j + jb), ldda,
                                      A2(j,j + jb), lda, queue );
                }
                magma_zlaqps( m, n_j, j, jb, &fjb,
                              A2(0, j), lda,
                              dA(0, j), ldda,
                              &jpvt[j], &tau[j], &rwork[j], &rwork[n + j],
                              work,
                              &work[jb], n_j,
                              &df[jb],   n_j ); // TODO: Find out why jpvt isn't changing.

                j += fjb;  /* fjb is actual number of columns factored */
            }
        }
        /* Use unblocked code to factor the last or only block. */
        if (j < k) {
            n_j = n - j;
            if (j > nfxd) {
                magma_zgetmatrix( m-j, n_j,
                                  dA(j,j), ldda,
                                  A2(j,j), lda, queue );
            }
            lapackf77_zlaqp2(&m, &n_j, &j, A2(0, j), &lda, &jpvt[j],
                             &tau[j], &rwork[j], &rwork[n+j], work );
        }
    }

    // The upper triangle of A2 contains the upper trapezoidal matrix S1
    magmablas_zlacpy(MagmaUpper, m, n, A2, lda, S1, lda, queue);

    // Partition S1 = [S11 S12]
    magmablas_ztranspose(k, n, S1, k, S11, n, queue);
    magmablas_ztranspose(n - k, k, S11 + (k * n), n - k, S12, k, queue);
    magmablas_ztranspose_inplace(k, S11, k, queue);

    // Invert S11 in place; it is now K x M instead of M x K
    magma_zgetrf(m, k, S11, m, ipvt, info);
    magma_zgetri_gpu(k, S11, m, ipvt, getriwork, lgetriwork, info);

    // T = S11^-1 * S12
    magmablas_zgemm( MagmaNoTrans, MagmaNoTrans, k, n - k, k,
                     c_one, S11, k,
                     S12, k,
                     c_zero, T, k,
                     queue );

    // Concatenate I_k and T
    for (j = 0; j < k; j++)
    {
        for (i = 0; i < k; i++)
        {
            if (i == j)
            {
                IT[(j * n) + i] = c_one;
            }
            else
            {
                IT[(j * n) + i] = c_zero;
            }
        }

        magma_zcopy(n, &T[j * n], ione, &IT[(j * n) + k], ione, queue);
    }
    // TODO: This could be more efficient. Concatenate T**H vertically to I and take the adjoint of the result.

    // Convert jpvt to a matrix P
    magmablas_zlaset( MagmaFull, k, n, c_zero, c_zero, P, n, queue );
    for (j = 0; j < k; j++)
    {
        P[(j * n) + jpvt[j] - 1] = c_one; // jpvt is indexed from 1
    }

    // V = P * ([I T]**H)
    magma_zgemm( MagmaNoTrans, MagmaConjTrans, k, k, n,
                 c_one, P, k,
                 IT, n,
                 c_zero, V, k,
                 queue );


cleanup:
    magma_free_cpu( A2 );
    magma_free( dwork );
    magma_free_cpu( S1 );
    magma_free_cpu( S11 );
    magma_free_cpu( S12 );
    magma_free_cpu( T );
    magma_free_cpu( IT );
    magma_free_cpu( P );
    magma_free_cpu( getriwork );
    magma_free_cpu( tau );
    magma_queue_destroy( queue );

    return *info;
}

/***************************************************************************//**
    Purpose
    -------
    z_ge2id computes the two sided interpolative decomposition of a complex
    M-by-N matrix A. The rank K two sided ID is written

        A = W * A(IPVT, JPVT) * conjugate-transpose(V)

    where IPVT is a set of K row indices, JPVT is a set of K column indices,
    W is an M-by-K matrix, and V is an N-by-K matrix.

    Note that the routine returns VT = V**H, not V.

    This algorithm is described in:
    Voronin, Sergey. Martinsson, Gunnar. Efficient algorithms
    for cur and interpolative matrix decomposition. 2016.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in]
    k       INTEGER
            The number of rows and columns of the matrix U. min(M,N) >= K >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, TODO.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A. LDA >= max(1,M).

    @param[in,out]
    ipvt    INTEGER array, dimension (K)
            On entry, if IPVT(I).ne.0, the I-th row of A is permuted
            to the front of A*P (a leading row); if IPVT(J)=0,
            the I-th row of A is a free row.
            On exit, if IPVT(J)=H, then the I-th row of C was the
            the H-th row of A.

    @param[in,out]
    jpvt    INTEGER array, dimension (K)
            On entry, if JPVT(J).ne.0, the J-th column of A is permuted
            to the front of A*P (a leading column); if JPVT(J)=0,
            the J-th column of A is a free column.
            On exit, if JPVT(J)=KH, then the J-th column of C was the
            the H-th column of A.

    @param[out]
    WT      COMPLEX_16 array, dimension (K,M)

    @param[out]
    VT      COMPLEX_16 array, dimension (N,K)

    @param[out]
    work   (workspace) COMPLEX_16 array on the GPU, dimension (MAX(1,LWORK))
            On exit, if INFO=0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            For [sd]geqp3, LWORK >= TODO;
            for [cz]geqp3, LWORK >= TODO,
            where NB is the optimal blocksize.
    \n
            Note: unlike the CPU interface of this routine, the GPU interface
            does not support a workspace query.

*/
#ifdef COMPLEX
/**

    @param
    rwork   (workspace, for [cz]geqp3 only) DOUBLE PRECISION array, dimension (2*MAX(M,N))

*/
#endif // COMPLEX
/**

    @param[out]
    info    INTEGER
      -     = 0: successful exit.
      -     < 0: if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ---------------
    TODO

    @ingroup TODO
*******************************************************************************/

extern "C" magma_int_t
magma_zge2id(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipvt, magma_int_t *jpvt,
    magmaDoubleComplex *W, magmaDoubleComplex *V,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info )
{
#define  A(i, j) (A     + (i) + (j)*(lda ))
#define dA(i, j) (dwork + (i) + (j)*(ldda))
#define min(m, n) (m < n ? m : n)
#define max(m, n) (m > n ? m : n)

    // const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    // const magmaDoubleComplex c_one = MAGMA_Z_ONE;
    const magma_int_t ione = 1;

    magmaDoubleComplex *CT = NULL;
    magma_zmalloc_cpu( &CT, k*m );

    magma_int_t j, nfxd, minmn;
    magma_int_t /*lwkopt=0, */lquery;

    minmn = min(m,n);

    *info = 0;
    lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    }

    if (k > minmn) {
        *info = -16; // TODO: Is this right?
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        goto cleanup;
    } else if (lquery) {
        goto cleanup;
    }

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    if (minmn == 0)
        goto cleanup;

    // Use ID to factor A = A[:,jpvt]V* = CV*
    magma_zgeid( m, n, k, A, lda, jpvt, V,
        work, lwork,
        #ifdef COMPLEX
        rwork,
        #endif
        info );

    /* Generate CT = C**H in a new matrix (copy jpvt columns of A to rows of CT)
     * Note jpvt uses 1-based indices for historical compatibility. */
    nfxd = 0;
    for (j = 0; j < n; ++j) {
        if (jpvt[j] != 0) {
            if (j != nfxd) {
                magma_zcopy(m, A(j, 0), n, CT + m * nfxd, ione, queue);
                jpvt[j]    = jpvt[nfxd];
                jpvt[nfxd] = j + 1;
            }
            else {
                jpvt[j] = j + 1;
            }
            ++nfxd;
        }
        else {
            jpvt[j] = j + 1;
        }
    }

    // Use ID to factor C**H = C[:,ipvt]W* = A[jpvt,ipvt]W*
    magma_zgeid( k, m, k, CT, k, ipvt, W,
        work, lwork,
        #ifdef COMPLEX
        rwork,
        #endif
        info );

cleanup:
    magma_free_cpu( CT );
    magma_queue_destroy( queue );

    return *info;
}

/***************************************************************************//**
    Purpose
    -------
    zgecur computes the CUR-factorization of a complex M-by-N
    matrix A using CUR-ID. The rank K CUR-factorization is written

        A = C * U * R

    where C is an M-by-K subset of the columns of A, U is a K-by-K
    matrix, and R is a K-by-N subset of the rows of A.

    using a rank k CUR-ID algorithm.

    This algorithm is described in:
    Voronin, Sergey. Martinsson, Gunnar. Efficient algorithms
    for cur and interpolative matrix decomposition. 2016.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in]
    k       INTEGER
            The number of rows and columns of the matrix U. min(M,N) >= K >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the first k rows of A contain the elements of A(:, JPVT).

    @param[in]
    lda     INTEGER
            The leading dimension of the array A. LDA >= max(1,M).

    @param[in,out]
    ipvt    INTEGER array, dimension (K)
            On entry, if IPVT(I).ne.0, the I-th row of A is permuted
            to the front of R (a leading row); if IPVT(I)=0,
            the I-th row of A is a free row.
            On exit, if IPVT(I)=H, then the I-th row of R was the
            the H-th row of A.

    @param[in,out]
    jpvt    INTEGER array, dimension (K)
            On entry, if JPVT(J).ne.0, the J-th column of A is permuted
            to the front of C (a leading column); if JPVT(J)=0,
            the J-th column of A is a free column.
            On exit, if JPVT(J)=H, then the J-th column of C was the
            the H-th column of A.

    @param[out]
    U       COMPLEX_16 array, dimension (K,K)
            A is approximately equal to A(:, JPVT) * U * A(IPVT, :).

    @param[out]
    work   (workspace) COMPLEX_16 array on the GPU, dimension (MAX(1,LWORK))
            On exit, if INFO=0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            For [sd]geqp3, LWORK >= TODO;
            for [cz]geqp3, LWORK >= TODO,
            where NB is the optimal blocksize.
    \n
            Note: unlike the CPU interface of this routine, the GPU interface
            does not support a workspace query.

*/
#ifdef COMPLEX
/**

    @param
    rwork   (workspace, for [cz]geqp3 only) DOUBLE PRECISION array, dimension MAX(5*K, 2*MAX(M,N))
    TODO?

*/
#endif // COMPLEX
/**

    @param[out]
    info    INTEGER
      -     = 0: successful exit.
      -     < 0: if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ---------------
    TODO

    @ingroup TODO
*******************************************************************************/

extern "C" magma_int_t
magma_zcurid(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipvt, magma_int_t *jpvt, magmaDoubleComplex *U,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info )
{
#define  A(i, j) (A     + (i) + (j)*(lda ))
#define dA(i, j) (dwork + (i) + (j)*(ldda))
#define min(m, n) (m < n ? m : n)
#define max(m, n) (m > n ? m : n)

    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;
    const magma_int_t ione = 1;

    magmaDoubleComplex *W = NULL;
    magma_zmalloc_cpu( &W, m*k );
    magmaDoubleComplex *V = NULL;
    magma_zmalloc_cpu( &V, n*k );
    double *s = NULL;
    magma_dmalloc_cpu( &s, k );
    magmaDoubleComplex *sinv = NULL;
    magma_zmalloc_cpu( &sinv, n*k );
    magmaDoubleComplex *RU = NULL;
    magma_zmalloc_cpu( &RU, k*k );
    magmaDoubleComplex *RVT = NULL;
    magma_zmalloc_cpu( &RVT, n*n );
    magmaDoubleComplex *Rinv = NULL;
    magma_zmalloc_cpu( &Rinv, n*k );

    magma_int_t j, nfxd, /*nb, */minmn;
    magma_int_t /*lwkopt=0, */lquery;

    *info = 0;
    lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    }

    minmn = min(m,n);

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        goto cleanup;
    } else if (lquery) {
        goto cleanup;
    }

    if (minmn == 0)
        goto cleanup;

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    /*
    TODO: Add something like this?
    #ifdef REAL
    double *rwork = max(5*k, max(2*max(m,n)));
    #endif
    */

    // Use 2-sided ID to factor A = W A[I,J] V*
    magma_zge2id(
        m, n, k, A, lda, ipvt, jpvt, W, V, work, lwork,
        #ifdef COMPLEX
        rwork,
        #endif
        info );

    /* Generate R in the first k rows of A
     * Note jpvt uses 1-based indices for historical compatibility. */
    nfxd = 0;
    for (j = 0; j < n; ++j) {
        if (jpvt[j] != 0) {
            if (j != nfxd) {
                blasf77_zswap(&m, A(0, j), &ione, A(0, nfxd), &ione);
                jpvt[j]    = jpvt[nfxd];
                jpvt[nfxd] = j + 1;
            }
            else {
                jpvt[j] = j + 1;
            }
            ++nfxd;
        }
        else {
            jpvt[j] = j + 1;
        }
    }

    // Compute pseudoinverse of R
    // See https://math.stackexchange.com/questions/458404/how-can-we-compute-pseudoinverse-for-any-matrix
    
    // Find the SVD of R: R = RU * diag(s) * RVT
    // Optimal lwork size is 2*k + max(2*k*nb, n*nb)
    // The rwork size is 5*k.
    magma_zgesvd( MagmaAllVec, MagmaAllVec, k, n,
                  A, k, s, RU, k, RVT, n,
                  work, lwork,
                  #ifdef COMPLEX
                  rwork,
                  #endif
                  info );

    for (j = 0; j < k; j++)
    {
        sinv[j * (1 + k)] = MAGMA_Z_MAKE(1.0 / s[j], 0.0);
    }

    magmablas_zgemm( MagmaConjTrans, MagmaNoTrans, n, n, k,
                   c_one, RU, n, sinv, n,
                   c_zero, work, n, queue );

    magmablas_zgemm( MagmaNoTrans, MagmaConjTrans, n, k, k,
                   c_one, work, n, RVT, k,
                   c_zero, Rinv, k, queue );

    // U = V*Rinv (Rinv is the pseudoinverse of R)
    magmablas_zgemm( MagmaConjTrans, MagmaNoTrans, k, n, k,
                   c_one, V, k, Rinv, k,
                   c_zero, U, k, queue );

cleanup:
    magma_free_cpu( W );
    magma_free_cpu( V );
    magma_free_cpu( s );
    magma_free_cpu( sinv );
    magma_free_cpu( RU );
    magma_free_cpu( RVT );
    magma_free_cpu( Rinv );
    magma_queue_destroy( queue );

    return *info;
}

// TODO: GPU version, c/d/s versions, randomized version, sparse version