To run this code, install MAGMA 2.4.0 and copy all files to the "example" folder.

Three files are included:
	zcurid.cpp defines functions for ID, two-sided ID, and CUR decomposition.
	z_curid_test.cpp implements the new functions and tests CUR on a random matrix.
	Makefile is modified from the "example" makefile and defines "make z_curid_test."

Run z_curid_test to test the provided functions: magma_zgeid, magma_zge2id, and magma_zcurid.