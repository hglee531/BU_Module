#ifdef __INTEL_MKL
INCLUDE 'mkl_pardiso.f90'
#endif

MODULE CSRMATRIX

#ifdef __INTEL_MKL
USE MKL_PARDISO
#endif

#ifdef __PGI
USE CUSPARSE
#endif

IMPLICIT NONE

TYPE CSR_FLOAT
  REAL(4), POINTER :: csrVal(:)
  INTEGER, POINTER :: csrRowPtr(:), csrColIdx(:)
  !--- Parallel Generation Variables
  REAL(4), POINTER :: bufVal(:, :)
  INTEGER, POINTER :: bufRowPtr(:, :), bufColIdx(:, :)
#ifdef __PGI
  REAL(4), POINTER, DEVICE :: d_csrVal(:)
  INTEGER, POINTER, DEVICE :: d_csrRowPtr(:), d_csrColIdx(:)
  TYPE(cusparseSolveAnalysisInfo) :: info, infoLU(2)
  TYPE(cusparseMatDescr) :: descr, descrLU(2)
#endif
  INTEGER :: nr, nc, nnz, nnzOmp(64), nThread
  LOGICAL :: lAlloc = .FALSE., lFinalized = .FALSE., lDevice = .FALSE., lParallel = .FALSE.
#ifdef __INTEL_MKL
  !--- PARDISO Variables
  TYPE(MKL_PARDISO_HANDLE) :: pardisoPtr(64)
  INTEGER :: pardisoParam(64)
  INTEGER, POINTER :: pardisoPermute(:)
#endif
END TYPE

TYPE CSR_FLOAT_COMPLEX
  COMPLEX(4), POINTER :: csrVal(:)
  INTEGER, POINTER :: csrRowPtr(:), csrColIdx(:)
  !--- Parallel Generation Variables
  REAL(4), POINTER :: bufVal(:, :)
  INTEGER, POINTER :: bufRowPtr(:, :), bufColIdx(:, :)
#ifdef __PGI
  REAL(4), POINTER, DEVICE :: d_csrVal(:)
  INTEGER, POINTER, DEVICE :: d_csrRowPtr(:), d_csrColIdx(:)
  TYPE(cusparseSolveAnalysisInfo) :: info, infoLU(2)
  TYPE(cusparseMatDescr) :: descr, descrLU(2)
#endif
  INTEGER :: nr, nc, nnz, nnzOmp(64), nThread
  LOGICAL :: lAlloc = .FALSE., lFinalized = .FALSE., lDevice = .FALSE., lParallel = .FALSE.
#ifdef __INTEL_MKL
  !--- PARDISO Variables
  TYPE(MKL_PARDISO_HANDLE) :: pardisoPtr(64)
  INTEGER :: pardisoParam(64)
  INTEGER, POINTER :: pardisoPermute(:)
#endif
END TYPE

TYPE CSR_DOUBLE
  REAL(8), POINTER :: csrVal(:)
  INTEGER, POINTER :: csrRowPtr(:), csrColIdx(:)
  !--- Parallel Generation Variables
  REAL(8), POINTER :: bufVal(:, :)
  INTEGER, POINTER :: bufRowPtr(:, :), bufColIdx(:, :)
#ifdef __PGI
  REAL(8), POINTER, DEVICE :: d_csrVal(:)
  INTEGER, POINTER, DEVICE :: d_csrRowPtr(:), d_csrColIdx(:)
  TYPE(cusparseSolveAnalysisInfo) :: info, infoLU(2)
  TYPE(cusparseMatDescr) :: descr, descrLU(2)
#endif
  INTEGER :: nr, nc, nnz, nnzOmp(64), nThread
  LOGICAL :: lAlloc = .FALSE., lFinalized = .FALSE., lDevice = .FALSE., lParallel = .FALSE.
#ifdef __INTEL_MKL
  !--- PARDISO Variables
  TYPE(MKL_PARDISO_HANDLE) :: pardisoPtr(64)
  INTEGER :: pardisoParam(64)
  INTEGER, POINTER :: pardisoPermute(:)
#endif
END TYPE

TYPE CSR_DOUBLE_COMPLEX
  COMPLEX(8), POINTER :: csrVal(:)
  INTEGER, POINTER :: csrRowPtr(:), csrColIdx(:)
  !--- Parallel Generation Variables
  COMPLEX(8), POINTER :: bufVal(:, :)
  INTEGER, POINTER :: bufRowPtr(:, :), bufColIdx(:, :)
#ifdef __PGI
  COMPLEX(8), POINTER, DEVICE :: d_csrVal(:)
  INTEGER, POINTER, DEVICE :: d_csrRowPtr(:), d_csrColIdx(:)
  TYPE(cusparseSolveAnalysisInfo) :: info, infoLU(2)
  TYPE(cusparseMatDescr) :: descr, descrLU(2)
#endif
  INTEGER :: nr, nc, nnz, nnzOmp(64), nThread
  LOGICAL :: lAlloc = .FALSE., lFinalized = .FALSE., lDevice = .FALSE., lParallel = .FALSE.
#ifdef __INTEL_MKL
  !--- PARDISO Variables
  TYPE(MKL_PARDISO_HANDLE) :: pardisoPtr(64)
  INTEGER :: pardisoParam(64)
  INTEGER, POINTER :: pardisoPermute(:)
#endif
END TYPE

TYPE CSR_MIXED
  REAL(8), POINTER :: csrVal(:)
  INTEGER, POINTER :: csrRowPtr(:), csrColIdx(:)
  !--- Parallel Generation Variables
  REAL(8), POINTER :: bufVal(:, :)
  INTEGER, POINTER :: bufRowPtr(:, :), bufColIdx(:, :)
#ifdef __PGI
  REAL(4), POINTER, DEVICE :: d_csrVal(:)
  REAL(8), POINTER, DEVICE :: d_csrVal8(:)
  INTEGER, POINTER, DEVICE :: d_csrRowPtr(:), d_csrColIdx(:)
  TYPE(cusparseSolveAnalysisInfo) :: info, infoLU(2)
  TYPE(cusparseMatDescr) :: descr, descrLU(2)
#endif
  INTEGER :: nr, nc, nnz, nnzOmp(64), nThread
  LOGICAL :: lAlloc = .FALSE., lFinalized = .FALSE., lDevice = .FALSE., lParallel = .FALSE.
#ifdef __INTEL_MKL
  !--- PARDISO Variables
  TYPE(MKL_PARDISO_HANDLE) :: pardisoPtr(64)
  INTEGER :: pardisoParam(64)
  INTEGER, POINTER :: pardisoPermute(:)
#endif
END TYPE

TYPE CSR_MIXED_COMPLEX
  COMPLEX(8), POINTER :: csrVal(:)
  INTEGER, POINTER :: csrRowPtr(:), csrColIdx(:)
  !--- Parallel Generation Variables
  COMPLEX(8), POINTER :: bufVal(:, :)
  INTEGER, POINTER :: bufRowPtr(:, :), bufColIdx(:, :)
#ifdef __PGI
  COMPLEX(4), POINTER, DEVICE :: d_csrVal(:)
  COMPLEX(8), POINTER, DEVICE :: d_csrVal8(:)
  INTEGER, POINTER, DEVICE :: d_csrRowPtr(:), d_csrColIdx(:)
  TYPE(cusparseSolveAnalysisInfo) :: info, infoLU(2)
  TYPE(cusparseMatDescr) :: descr, descrLU(2)
#endif
  INTEGER :: nr, nc, nnz, nnzOmp(64), nThread
  LOGICAL :: lAlloc = .FALSE., lFinalized = .FALSE., lDevice = .FALSE., lParallel = .FALSE.
#ifdef __INTEL_MKL
  !--- PARDISO Variables
  TYPE(MKL_PARDISO_HANDLE) :: pardisoPtr(64)
  INTEGER :: pardisoParam(64)
  INTEGER, POINTER :: pardisoPermute(:)
#endif
END TYPE

INTERFACE createCsr
  MODULE PROCEDURE createCsrFloat
  MODULE PROCEDURE createCsrDouble
  MODULE PROCEDURE createCsrMixed
  MODULE PROCEDURE createCsrFloatZ
  MODULE PROCEDURE createCsrDoubleZ
  MODULE PROCEDURE createCsrMixedZ
  MODULE PROCEDURE createCsrFloatParallel
  MODULE PROCEDURE createCsrDoubleParallel
  MODULE PROCEDURE createCsrMixedParallel
END INTERFACE

INTERFACE destroyCsr
  MODULE PROCEDURE destroyCsrFloat
  MODULE PROCEDURE destroyCsrDouble
  MODULE PROCEDURE destroyCsrMixed
  MODULE PROCEDURE destroyCsrFloatZ
  MODULE PROCEDURE destroyCsrDoubleZ
  MODULE PROCEDURE destroyCsrMixedZ
END INTERFACE

INTERFACE pushCsr
  MODULE PROCEDURE pushCsrFloat4
  MODULE PROCEDURE pushCsrFloat8
  MODULE PROCEDURE pushCsrDouble4
  MODULE PROCEDURE pushCsrDouble8
  MODULE PROCEDURE pushCsrMixed4
  MODULE PROCEDURE pushCsrMixed8
  
  MODULE PROCEDURE pushCsrFloatZ4
  MODULE PROCEDURE pushCsrFloatZ8
  MODULE PROCEDURE pushCsrDoubleZ4
  MODULE PROCEDURE pushCsrDoubleZ8
  MODULE PROCEDURE pushCsrMixedZ4
  MODULE PROCEDURE pushCsrMixedZ8
  
  MODULE PROCEDURE pushCsrFloatZ4Z
  MODULE PROCEDURE pushCsrFloatZ8Z
  MODULE PROCEDURE pushCsrDoubleZ4Z
  MODULE PROCEDURE pushCsrDoubleZ8Z
  MODULE PROCEDURE pushCsrMixedZ4Z
  MODULE PROCEDURE pushCsrMixedZ8Z
  
  MODULE PROCEDURE pushCsrParallelFloat4
  MODULE PROCEDURE pushCsrParallelFloat8
  MODULE PROCEDURE pushCsrParallelDouble4
  MODULE PROCEDURE pushCsrParallelDouble8
  MODULE PROCEDURE pushCsrParallelMixed4
  MODULE PROCEDURE pushCsrParallelMixed8
END INTERFACE

INTERFACE finalizeCsr
  MODULE PROCEDURE finalizeCsrFloat
  MODULE PROCEDURE finalizeCsrDouble
  MODULE PROCEDURE finalizeCsrMixed
  MODULE PROCEDURE finalizeCsrFloatZ
  MODULE PROCEDURE finalizeCsrDoubleZ
  MODULE PROCEDURE finalizeCsrMixedZ
END INTERFACE

INTERFACE finalizeSortCsr
  MODULE PROCEDURE finalizeSortCsrFloat
  MODULE PROCEDURE finalizeSortCsrDouble
  MODULE PROCEDURE finalizeSortCsrMixed
  !MODULE PROCEDURE finalizeSortCsrFloatZ
  !MODULE PROCEDURE finalizeSortCsrDoubleZ
  !MODULE PROCEDURE finalizeSortCsrMixedZ
END INTERFACE

INTERFACE printCsr
  MODULE PROCEDURE printCsrFloat
  MODULE PROCEDURE printCsrDouble
  MODULE PROCEDURE printCsrMixed
END INTERFACE

INTERFACE ASSIGNMENT (=)
  MODULE PROCEDURE copyCsrFloat
  MODULE PROCEDURE copyCsrDouble
  MODULE PROCEDURE copyCsrMixed
  MODULE PROCEDURE copyCsrFloatZ
  MODULE PROCEDURE copyCsrDoubleZ
  MODULE PROCEDURE copyCsrMixedZ
  MODULE PROCEDURE copyCsrFloat2Double
  MODULE PROCEDURE copyCsrFloat2Mixed
  MODULE PROCEDURE copyCsrDouble2Float
  MODULE PROCEDURE copyCsrDouble2Mixed
  MODULE PROCEDURE copyCsrMixed2Float
  MODULE PROCEDURE copyCsrMixed2Double
  ! REAL to COMPLEX
  MODULE PROCEDURE copyCsrFloat2FloatZ
  MODULE PROCEDURE copyCsrFloat2DoubleZ
  MODULE PROCEDURE copyCsrFloat2MixedZ
  MODULE PROCEDURE copyCsrDouble2FloatZ
  MODULE PROCEDURE copyCsrDouble2DoubleZ
  MODULE PROCEDURE copyCsrDouble2MixedZ
  MODULE PROCEDURE copyCsrMixed2FloatZ
  MODULE PROCEDURE copyCsrMixed2DoubleZ
  MODULE PROCEDURE copyCsrMixed2MixedZ
  ! Complex to Complex
  MODULE PROCEDURE copyCsrFloatZ2DoubleZ
  MODULE PROCEDURE copyCsrFloatZ2MixedZ
  MODULE PROCEDURE copyCsrDoubleZ2FloatZ
  MODULE PROCEDURE copyCsrDoubleZ2MixedZ
  MODULE PROCEDURE copyCsrMixedZ2FloatZ
  MODULE PROCEDURE copyCsrMixedZ2DoubleZ
END INTERFACE

INTERFACE OPERATOR (-)
  MODULE PROCEDURE subCsrCsrFloat
  MODULE PROCEDURE subCsrCsrDouble
END INTERFACE

PRIVATE :: allocCsrParallelBufferFloat, allocCsrParallelBufferDouble, allocCsrParallelBufferMixed
PRIVATE :: collapseCsrParallelBufferFloat, collapseCsrParallelBufferDouble, collapseCsrParallelBufferMixed

CONTAINS

SUBROUTINE createCsrFloat(csrFloat, nnz, nr, nc)

IMPLICIT NONE

TYPE(CSR_FLOAT) :: csrFloat
INTEGER :: nnz, nr, nc
INTEGER :: ierr

IF (csrFloat%lAlloc) CALL destroyCsr(csrFloat)

csrFloat%nnz = 0
csrFloat%nr = nr
csrFloat%nc = nc

ALLOCATE(csrFloat%csrVal(nnz))
ALLOCATE(csrFloat%csrColIdx(nnz))
ALLOCATE(csrFloat%csrRowPtr(nr + 1))
csrFloat%csrVal = 0.0
csrFloat%csrColIdx = 0
csrFloat%csrRowPtr = 0

#ifdef __INTEL_MKL
ALLOCATE(csrFloat%pardisoPermute(nr))
#endif

#ifdef __PGI
ierr = cusparseCreateMatDescr(csrFloat%descr)
ierr = cusparseCreateMatDescr(csrFloat%descrLU(1))
ierr = cusparseCreateMatDescr(csrFloat%descrLU(2))
#endif

csrFloat%lAlloc = .TRUE.

END SUBROUTINE

SUBROUTINE createCsrDouble(csrDouble, nnz, nr, nc)

IMPLICIT NONE

TYPE(CSR_DOUBLE) :: csrDouble
INTEGER :: nnz, nr, nc
INTEGER :: ierr

IF (csrDouble%lAlloc) CALL destroyCsr(csrDouble)

csrDouble%nnz = 0
csrDouble%nr = nr
csrDouble%nc = nc

ALLOCATE(csrDouble%csrVal(nnz))
ALLOCATE(csrDouble%csrColIdx(nnz))
ALLOCATE(csrDouble%csrRowPtr(nr + 1))
csrDouble%csrVal = 0.0
csrDouble%csrColIdx = 0
csrDouble%csrRowPtr = 0

#ifdef __INTEL_MKL
ALLOCATE(csrDouble%pardisoPermute(nr))
#endif

#ifdef __PGI
ierr = cusparseCreateMatDescr(csrDouble%descr)
ierr = cusparseCreateMatDescr(csrDouble%descrLU(1))
ierr = cusparseCreateMatDescr(csrDouble%descrLU(2))
#endif

csrDouble%lAlloc = .TRUE.

END SUBROUTINE

SUBROUTINE createCsrMixed(csrMixed, nnz, nr, nc)

IMPLICIT NONE

TYPE(CSR_MIXED) :: csrMixed
INTEGER :: nnz, nr, nc
INTEGER :: ierr

IF (csrMixed%lAlloc) CALL destroyCsr(csrMixed)

csrMixed%nnz = 0
csrMixed%nr = nr
csrMixed%nc = nc

ALLOCATE(csrMixed%csrVal(nnz))
ALLOCATE(csrMixed%csrColIdx(nnz))
ALLOCATE(csrMixed%csrRowPtr(nr + 1))
csrMixed%csrVal = 0.0
csrMixed%csrColIdx = 0
csrMixed%csrRowPtr = 0

#ifdef __INTEL_MKL
ALLOCATE(csrMixed%pardisoPermute(nr))
#endif

#ifdef __PGI
ierr = cusparseCreateMatDescr(csrMixed%descr)
ierr = cusparseCreateMatDescr(csrMixed%descrLU(1))
ierr = cusparseCreateMatDescr(csrMixed%descrLU(2))
#endif

csrMixed%lAlloc = .TRUE.

END SUBROUTINE

SUBROUTINE createCsrFloatZ(csrFloatZ, nnz, nr, nc)

IMPLICIT NONE

TYPE(CSR_FLOAT_COMPLEX) :: csrFloatZ
INTEGER :: nnz, nr, nc
INTEGER :: ierr

IF (csrFloatZ%lAlloc) CALL destroyCsr(csrFloatZ)

csrFloatZ%nnz = 0
csrFloatZ%nr = nr
csrFloatZ%nc = nc

ALLOCATE(csrFloatZ%csrVal(nnz))
ALLOCATE(csrFloatZ%csrColIdx(nnz))
ALLOCATE(csrFloatZ%csrRowPtr(nr + 1))
csrFloatZ%csrVal = 0.0
csrFloatZ%csrColIdx = 0
csrFloatZ%csrRowPtr = 0

#ifdef __INTEL_MKL
ALLOCATE(csrFloatZ%pardisoPermute(nr))
#endif

#ifdef __PGI
ierr = cusparseCreateMatDescr(csrFloatZ%descr)
ierr = cusparseCreateMatDescr(csrFloatZ%descrLU(1))
ierr = cusparseCreateMatDescr(csrFloatZ%descrLU(2))
#endif

csrFloatZ%lAlloc = .TRUE.

END SUBROUTINE

SUBROUTINE createCsrDoubleZ(csrDoubleZ, nnz, nr, nc)

IMPLICIT NONE

TYPE(CSR_DOUBLE_COMPLEX) :: csrDoubleZ
INTEGER :: nnz, nr, nc
INTEGER :: ierr

IF (csrDoubleZ%lAlloc) CALL destroyCsr(csrDoubleZ)

csrDoubleZ%nnz = 0
csrDoubleZ%nr = nr
csrDoubleZ%nc = nc

ALLOCATE(csrDoubleZ%csrVal(nnz))
ALLOCATE(csrDoubleZ%csrColIdx(nnz))
ALLOCATE(csrDoubleZ%csrRowPtr(nr + 1))
csrDoubleZ%csrVal = 0.0
csrDoubleZ%csrColIdx = 0
csrDoubleZ%csrRowPtr = 0

#ifdef __INTEL_MKL
ALLOCATE(csrDoubleZ%pardisoPermute(nr))
#endif

#ifdef __PGI
ierr = cusparseCreateMatDescr(csrDoubleZ%descr)
ierr = cusparseCreateMatDescr(csrDoubleZ%descrLU(1))
ierr = cusparseCreateMatDescr(csrDoubleZ%descrLU(2))
#endif

csrDoubleZ%lAlloc = .TRUE.

END SUBROUTINE

SUBROUTINE createCsrMixedZ(csrMixedZ, nnz, nr, nc)

IMPLICIT NONE

TYPE(CSR_MIXED_COMPLEX) :: csrMixedZ
INTEGER :: nnz, nr, nc
INTEGER :: ierr

IF (csrMixedZ%lAlloc) CALL destroyCsr(csrMixedZ)

csrMixedZ%nnz = 0
csrMixedZ%nr = nr
csrMixedZ%nc = nc

ALLOCATE(csrMixedZ%csrVal(nnz))
ALLOCATE(csrMixedZ%csrColIdx(nnz))
ALLOCATE(csrMixedZ%csrRowPtr(nr + 1))
csrMixedZ%csrVal = 0.0
csrMixedZ%csrColIdx = 0
csrMixedZ%csrRowPtr = 0

#ifdef __INTEL_MKL
ALLOCATE(csrMixedZ%pardisoPermute(nr))
#endif

#ifdef __PGI
ierr = cusparseCreateMatDescr(csrMixedZ%descr)
ierr = cusparseCreateMatDescr(csrMixedZ%descrLU(1))
ierr = cusparseCreateMatDescr(csrMixedZ%descrLU(2))
#endif

csrMixedZ%lAlloc = .TRUE.

END SUBROUTINE

SUBROUTINE createCsrFloatParallel(csrFloat, nnz, nr, nc, nThread)

IMPLICIT NONE

TYPE(CSR_FLOAT) :: csrFloat
INTEGER :: nnz, nr, nc, nThread

CALL createCsr(csrFloat, nnz, nr, nc)
CALL allocCsrParallelBufferFloat(csrFloat, nnz, nThread)

END SUBROUTINE

SUBROUTINE createCsrDoubleParallel(csrDouble, nnz, nr, nc, nThread)

IMPLICIT NONE

TYPE(CSR_DOUBLE) :: csrDouble
INTEGER :: nnz, nr, nc, nThread

CALL createCsr(csrDouble, nnz, nr, nc)
CALL allocCsrParallelBufferDouble(csrDouble, nnz, nThread)

END SUBROUTINE

SUBROUTINE createCsrMixedParallel(csrMixed, nnz, nr, nc, nThread)

IMPLICIT NONE

TYPE(CSR_MIXED) :: csrMixed
INTEGER :: nnz, nr, nc, nThread

CALL createCsr(csrMixed, nnz, nr, nc)
CALL allocCsrParallelBufferMixed(csrMixed, nnz, nThread)

END SUBROUTINE

SUBROUTINE destroyCsrFloat(csrFloat)

IMPLICIT NONE

TYPE(CSR_FLOAT) :: csrFloat
INTEGER :: ierr

IF (.NOT. csrFloat%lAlloc) RETURN

DEALLOCATE(csrFloat%csrVal)
DEALLOCATE(csrFloat%csrColIdx)
DEALLOCATE(csrFloat%csrRowPtr)

#ifdef __PGI
IF (csrFloat%lDevice) THEN
  DEALLOCATE(csrFloat%d_csrVal)
  DEALLOCATE(csrFloat%d_csrColIdx)
  DEALLOCATE(csrFloat%d_csrRowPtr)
ENDIF

ierr = cusparseDestroyMatDescr(csrFloat%descr)
ierr = cusparseDestroyMatDescr(csrFloat%descrLU(1))
ierr = cusparseDestroyMatDescr(csrFloat%descrLU(2))
#endif

#ifdef __INTEL_MKL
DEALLOCATE(csrFloat%pardisoPermute)
#endif

csrFloat%lAlloc = .FALSE.
csrFloat%lFinalized = .FALSE.
csrFloat%lDevice = .FALSE.

END SUBROUTINE

SUBROUTINE destroyCsrDouble(csrDouble)

IMPLICIT NONE

TYPE(CSR_DOUBLE) :: csrDouble
INTEGER :: ierr

IF (.NOT. csrDouble%lAlloc) RETURN

DEALLOCATE(csrDouble%csrVal)
DEALLOCATE(csrDouble%csrColIdx)
DEALLOCATE(csrDouble%csrRowPtr)

#ifdef __PGI
IF (csrDouble%lDevice) THEN
  DEALLOCATE(csrDouble%d_csrVal)
  DEALLOCATE(csrDouble%d_csrColIdx)
  DEALLOCATE(csrDouble%d_csrRowPtr)
ENDIF

ierr = cusparseDestroyMatDescr(csrDouble%descr)
ierr = cusparseDestroyMatDescr(csrDouble%descrLU(1))
ierr = cusparseDestroyMatDescr(csrDouble%descrLU(2))
#endif

#ifdef __INTEL_MKL
DEALLOCATE(csrDouble%pardisoPermute)
#endif

csrDouble%lAlloc = .FALSE.
csrDouble%lFinalized = .FALSE.
csrDouble%lDevice = .FALSE.

END SUBROUTINE

SUBROUTINE destroyCsrMixed(csrMixed)

IMPLICIT NONE

TYPE(CSR_MIXED) :: csrMixed
INTEGER :: ierr

IF (.NOT. csrMixed%lAlloc) RETURN

DEALLOCATE(csrMixed%csrVal)
DEALLOCATE(csrMixed%csrColIdx)
DEALLOCATE(csrMixed%csrRowPtr)

#ifdef __PGI
IF (csrMixed%lDevice) THEN
  DEALLOCATE(csrMixed%d_csrVal)
  DEALLOCATE(csrMixed%d_csrVal8)
  DEALLOCATE(csrMixed%d_csrColIdx)
  DEALLOCATE(csrMixed%d_csrRowPtr)
ENDIF

ierr = cusparseDestroyMatDescr(csrMixed%descr)
ierr = cusparseDestroyMatDescr(csrMixed%descrLU(1))
ierr = cusparseDestroyMatDescr(csrMixed%descrLU(2))
#endif

#ifdef __INTEL_MKL
DEALLOCATE(csrMixed%pardisoPermute)
#endif

csrMixed%lAlloc = .FALSE.
csrMixed%lFinalized = .FALSE.
csrMixed%lDevice = .FALSE.

END SUBROUTINE

SUBROUTINE destroyCsrFloatZ(csrFloatZ)

IMPLICIT NONE

TYPE(CSR_FLOAT_COMPLEX) :: csrFloatZ
INTEGER :: ierr

IF (.NOT. csrFloatZ%lAlloc) RETURN

DEALLOCATE(csrFloatZ%csrVal)
DEALLOCATE(csrFloatZ%csrColIdx)
DEALLOCATE(csrFloatZ%csrRowPtr)

#ifdef __PGI
IF (csrFloatZ%lDevice) THEN
  DEALLOCATE(csrFloatZ%d_csrVal)
  DEALLOCATE(csrFloatZ%d_csrColIdx)
  DEALLOCATE(csrFloatZ%d_csrRowPtr)
ENDIF

ierr = cusparseDestroyMatDescr(csrFloatZ%descr)
ierr = cusparseDestroyMatDescr(csrFloatZ%descrLU(1))
ierr = cusparseDestroyMatDescr(csrFloatZ%descrLU(2))
#endif

#ifdef __INTEL_MKL
DEALLOCATE(csrFloatZ%pardisoPermute)
#endif

csrFloatZ%lAlloc = .FALSE.
csrFloatZ%lFinalized = .FALSE.
csrFloatZ%lDevice = .FALSE.

END SUBROUTINE

SUBROUTINE destroyCsrDoubleZ(csrDoubleZ)

IMPLICIT NONE

TYPE(CSR_DOUBLE_COMPLEX) :: csrDoubleZ
INTEGER :: ierr

IF (.NOT. csrDoubleZ%lAlloc) RETURN

DEALLOCATE(csrDoubleZ%csrVal)
DEALLOCATE(csrDoubleZ%csrColIdx)
DEALLOCATE(csrDoubleZ%csrRowPtr)

#ifdef __PGI
IF (csrDoubleZ%lDevice) THEN
  DEALLOCATE(csrDoubleZ%d_csrVal)
  DEALLOCATE(csrDoubleZ%d_csrColIdx)
  DEALLOCATE(csrDoubleZ%d_csrRowPtr)
ENDIF

ierr = cusparseDestroyMatDescr(csrDoubleZ%descr)
ierr = cusparseDestroyMatDescr(csrDoubleZ%descrLU(1))
ierr = cusparseDestroyMatDescr(csrDoubleZ%descrLU(2))
#endif

#ifdef __INTEL_MKL
DEALLOCATE(csrDoubleZ%pardisoPermute)
#endif

csrDoubleZ%lAlloc = .FALSE.
csrDoubleZ%lFinalized = .FALSE.
csrDoubleZ%lDevice = .FALSE.

END SUBROUTINE

SUBROUTINE destroyCsrMixedZ(csrMixedZ)

IMPLICIT NONE

TYPE(CSR_MIXED_COMPLEX) :: csrMixedZ
INTEGER :: ierr

IF (.NOT. csrMixedZ%lAlloc) RETURN

DEALLOCATE(csrMixedZ%csrVal)
DEALLOCATE(csrMixedZ%csrColIdx)
DEALLOCATE(csrMixedZ%csrRowPtr)

#ifdef __PGI
IF (csrMixedZ%lDevice) THEN
  DEALLOCATE(csrMixedZ%d_csrVal)
  DEALLOCATE(csrMixedZ%d_csrVal8)
  DEALLOCATE(csrMixedZ%d_csrColIdx)
  DEALLOCATE(csrMixedZ%d_csrRowPtr)
ENDIF

ierr = cusparseDestroyMatDescr(csrMixedZ%descr)
ierr = cusparseDestroyMatDescr(csrMixedZ%descrLU(1))
ierr = cusparseDestroyMatDescr(csrMixedZ%descrLU(2))
#endif

#ifdef __INTEL_MKL
DEALLOCATE(csrMixedZ%pardisoPermute)
#endif

csrMixedZ%lAlloc = .FALSE.
csrMixedZ%lFinalized = .FALSE.
csrMixedZ%lDevice = .FALSE.

END SUBROUTINE

SUBROUTINE allocCsrParallelBufferFloat(csrFloat, nnz, nThread)

IMPLICIT NONE

TYPE(CSR_FLOAT) :: csrFloat
INTEGER :: nr, nc, nnz, nThread

IF (csrFloat%lParallel) RETURN

nr = csrFloat%nr
nc = csrFloat%nc

csrFloat%nnzOmp = 0

ALLOCATE(csrFloat%bufVal(nnz, nThread))
ALLOCATE(csrFloat%bufColIdx(nnz, nThread))
ALLOCATE(csrFloat%bufRowPtr(nr, nThread))
csrFloat%bufVal = 0.0
csrFloat%bufColIdx = 0
csrFloat%bufRowPtr = 0

csrFloat%nThread = nThread
csrFloat%lParallel = .TRUE.

END SUBROUTINE

SUBROUTINE allocCsrParallelBufferDouble(csrDouble, nnz, nThread)

IMPLICIT NONE

TYPE(CSR_DOUBLE) :: csrDouble
INTEGER :: nr, nc, nnz, nThread

IF (csrDouble%lParallel) RETURN

nr = csrDouble%nr
nc = csrDouble%nc

csrDouble%nnzOmp = 0

ALLOCATE(csrDouble%bufVal(nnz, nThread))
ALLOCATE(csrDouble%bufColIdx(nnz, nThread))
ALLOCATE(csrDouble%bufRowPtr(nr, nThread))
csrDouble%bufVal = 0.0
csrDouble%bufColIdx = 0
csrDouble%bufRowPtr = 0

csrDouble%nThread = nThread
csrDouble%lParallel = .TRUE.

END SUBROUTINE

SUBROUTINE allocCsrParallelBufferMixed(csrMixed, nnz, nThread)

IMPLICIT NONE

TYPE(CSR_MIXED) :: csrMixed
INTEGER :: nr, nc, nnz, nThread

IF (csrMixed%lParallel) RETURN

nr = csrMixed%nr
nc = csrMixed%nc

csrMixed%nnzOmp = 0

ALLOCATE(csrMixed%bufVal(nnz, nThread))
ALLOCATE(csrMixed%bufColIdx(nnz, nThread))
ALLOCATE(csrMixed%bufRowPtr(nr, nThread))
csrMixed%bufVal = 0.0
csrMixed%bufColIdx = 0
csrMixed%bufRowPtr = 0

csrMixed%nThread = nThread
csrMixed%lParallel = .TRUE.

END SUBROUTINE

SUBROUTINE collapseCsrParallelBufferFloat(csrFloat)

IMPLICIT NONE

TYPE(CSR_FLOAT) :: csrFloat
INTEGER :: nr, nc, nnz, nThread
INTEGER :: i, ir, ic, inz, inzOmp(64), tid
INTEGER, POINTER :: rowList(:)

IF (.NOT. csrFloat%lParallel) RETURN

nr = csrFloat%nr
nc = csrFloat%nc
nThread = csrFloat%nThread

ALLOCATE(rowList(nr))

!$OMP PARALLEL
!$OMP DO SCHEDULE(GUIDED)
DO ir = 1, nr
  DO tid = 1, nThread
    IF (csrFloat%bufRowPtr(ir, tid) .NE. 0) THEN
      csrFloat%csrRowPtr(ir) = csrFloat%bufRowPtr(ir, tid)
      rowList(ir) = tid
      EXIT
    ENDIF
  ENDDO
ENDDO
!$OMP END DO
!$OMP END PARALLEL

inz = 0; inzOmp = 0
DO ir = 1, nr
  tid = rowList(ir)
  nnz = csrFloat%bufRowPtr(ir, tid)
  DO i = 1, nnz
    inz = inz + 1
    inzOmp(tid) = inzOmp(tid) + 1
    csrFloat%csrVal(inz) = csrFloat%bufVal(inzOmp(tid), tid)
    csrFloat%csrColIdx(inz) = csrFloat%bufColIdx(inzOmp(tid), tid)
  ENDDO
  csrFloat%nnz = csrFloat%nnz + nnz
ENDDO

DEALLOCATE(rowList)

DEALLOCATE(csrFloat%bufVal)
DEALLOCATE(csrFloat%bufColIdx)
DEALLOCATE(csrFloat%bufRowPtr)

csrFloat%lParallel = .FALSE.

END SUBROUTINE

SUBROUTINE collapseCsrParallelBufferDouble(csrDouble)

IMPLICIT NONE

TYPE(CSR_DOUBLE) :: csrDouble
INTEGER :: nr, nc, nnz, nThread
INTEGER :: i, ir, ic, inz, inzOmp(64), tid
INTEGER, POINTER :: rowList(:)

IF (.NOT. csrDouble%lParallel) RETURN

nr = csrDouble%nr
nc = csrDouble%nc
nThread = csrDouble%nThread

ALLOCATE(rowList(nr))

!$OMP PARALLEL
!$OMP DO SCHEDULE(GUIDED)
DO ir = 1, nr
  DO tid = 1, nThread
    IF (csrDouble%bufRowPtr(ir, tid) .NE. 0) THEN
      csrDouble%csrRowPtr(ir) = csrDouble%bufRowPtr(ir, tid)
      rowList(ir) = tid
      EXIT
    ENDIF
  ENDDO
ENDDO
!$OMP END DO
!$OMP END PARALLEL

inz = 0; inzOmp = 0
DO ir = 1, nr
  tid = rowList(ir)
  nnz = csrDouble%bufRowPtr(ir, tid)
  DO i = 1, nnz
    inz = inz + 1
    inzOmp(tid) = inzOmp(tid) + 1
    csrDouble%csrVal(inz) = csrDouble%bufVal(inzOmp(tid), tid)
    csrDouble%csrColIdx(inz) = csrDouble%bufColIdx(inzOmp(tid), tid)
  ENDDO
  csrDouble%nnz = csrDouble%nnz + nnz
ENDDO

DEALLOCATE(rowList)

DEALLOCATE(csrDouble%bufVal)
DEALLOCATE(csrDouble%bufColIdx)
DEALLOCATE(csrDouble%bufRowPtr)

csrDouble%lParallel = .FALSE.

END SUBROUTINE

SUBROUTINE collapseCsrParallelBufferMixed(csrMixed)

IMPLICIT NONE

TYPE(CSR_MIXED) :: csrMixed
INTEGER :: nr, nc, nnz, nThread
INTEGER :: i, ir, ic, inz, inzOmp(64), tid
INTEGER, POINTER :: rowList(:)

IF (.NOT. csrMixed%lParallel) RETURN

nr = csrMixed%nr
nc = csrMixed%nc
nThread = csrMixed%nThread

ALLOCATE(rowList(nr))

!$OMP PARALLEL
!$OMP DO SCHEDULE(GUIDED)
DO ir = 1, nr
  DO tid = 1, nThread
    IF (csrMixed%bufRowPtr(ir, tid) .NE. 0) THEN
      csrMixed%csrRowPtr(ir) = csrMixed%bufRowPtr(ir, tid)
      rowList(ir) = tid
      EXIT
    ENDIF
  ENDDO
ENDDO
!$OMP END DO
!$OMP END PARALLEL

inz = 0; inzOmp = 0
DO ir = 1, nr
  tid = rowList(ir)
  nnz = csrMixed%bufRowPtr(ir, tid)
  DO i = 1, nnz
    inz = inz + 1
    inzOmp(tid) = inzOmp(tid) + 1
    csrMixed%csrVal(inz) = csrMixed%bufVal(inzOmp(tid), tid)
    csrMixed%csrColIdx(inz) = csrMixed%bufColIdx(inzOmp(tid), tid)
  ENDDO
  csrMixed%nnz = csrMixed%nnz + nnz
ENDDO

DEALLOCATE(rowList)

DEALLOCATE(csrMixed%bufVal)
DEALLOCATE(csrMixed%bufColIdx)
DEALLOCATE(csrMixed%bufRowPtr)

csrMixed%lParallel = .FALSE.

END SUBROUTINE

SUBROUTINE pushCsrFloat4(csrFloat, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_FLOAT) :: csrFloat
REAL(4) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrFloat%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrFloat%nc)        RETURN

csrFloat%nnz = csrFloat%nnz + 1
csrFloat%csrVal(csrFloat%nnz) = val
csrFloat%csrColIdx(csrFloat%nnz) = ic
csrFloat%csrRowPtr(ir) = csrFloat%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrFloat8(csrFloat, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_FLOAT) :: csrFloat
REAL(8) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0D-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrFloat%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrFloat%nc)        RETURN

csrFloat%nnz = csrFloat%nnz + 1
csrFloat%csrVal(csrFloat%nnz) = val
csrFloat%csrColIdx(csrFloat%nnz) = ic
csrFloat%csrRowPtr(ir) = csrFloat%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrDouble4(csrDouble, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_DOUBLE) :: csrDouble
REAL(4) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrDouble%nr)       RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrDouble%nc)       RETURN

csrDouble%nnz = csrDouble%nnz + 1
csrDouble%csrVal(csrDouble%nnz) = val
csrDouble%csrColIdx(csrDouble%nnz) = ic
csrDouble%csrRowPtr(ir) = csrDouble%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrDouble8(csrDouble, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_DOUBLE) :: csrDouble
REAL(8) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0D-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrDouble%nr)       RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrDouble%nc)       RETURN

csrDouble%nnz = csrDouble%nnz + 1
csrDouble%csrVal(csrDouble%nnz) = val
csrDouble%csrColIdx(csrDouble%nnz) = ic
csrDouble%csrRowPtr(ir) = csrDouble%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrMixed4(csrMixed, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_MIXED) :: csrMixed
REAL(4) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrMixed%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrMixed%nc)        RETURN

csrMixed%nnz = csrMixed%nnz + 1
csrMixed%csrVal(csrMixed%nnz) = val
csrMixed%csrColIdx(csrMixed%nnz) = ic
csrMixed%csrRowPtr(ir) = csrMixed%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrMixed8(csrMixed, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_MIXED) :: csrMixed
REAL(8) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0D-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrMixed%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrMixed%nc)        RETURN

csrMixed%nnz = csrMixed%nnz + 1
csrMixed%csrVal(csrMixed%nnz) = val
csrMixed%csrColIdx(csrMixed%nnz) = ic
csrMixed%csrRowPtr(ir) = csrMixed%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrFloatZ4(csrFloatZ, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_FLOAT_COMPLEX) :: csrFloatZ
REAL(4) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrFloatZ%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrFloatZ%nc)        RETURN

csrFloatZ%nnz = csrFloatZ%nnz + 1
csrFloatZ%csrVal(csrFloatZ%nnz) = val
csrFloatZ%csrColIdx(csrFloatZ%nnz) = ic
csrFloatZ%csrRowPtr(ir) = csrFloatZ%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrFloatZ8(csrFloatZ, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_FLOAT_COMPLEX) :: csrFloatZ
REAL(8) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0D-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrFloatZ%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrFloatZ%nc)        RETURN

csrFloatZ%nnz = csrFloatZ%nnz + 1
csrFloatZ%csrVal(csrFloatZ%nnz) = val
csrFloatZ%csrColIdx(csrFloatZ%nnz) = ic
csrFloatZ%csrRowPtr(ir) = csrFloatZ%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrDoubleZ4(csrDoubleZ, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_DOUBLE_COMPLEX) :: csrDoubleZ
REAL(4) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrDoubleZ%nr)       RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrDoubleZ%nc)       RETURN

csrDoubleZ%nnz = csrDoubleZ%nnz + 1
csrDoubleZ%csrVal(csrDoubleZ%nnz) = val
csrDoubleZ%csrColIdx(csrDoubleZ%nnz) = ic
csrDoubleZ%csrRowPtr(ir) = csrDoubleZ%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrDoubleZ8(csrDoubleZ, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_DOUBLE_COMPLEX) :: csrDoubleZ
REAL(8) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrDoubleZ%nr)       RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrDoubleZ%nc)       RETURN

csrDoubleZ%nnz = csrDoubleZ%nnz + 1
csrDoubleZ%csrVal(csrDoubleZ%nnz) = val
csrDoubleZ%csrColIdx(csrDoubleZ%nnz) = ic
csrDoubleZ%csrRowPtr(ir) = csrDoubleZ%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrMixedZ4(csrMixedZ, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_MIXED_COMPLEX) :: csrMixedZ
REAL(4) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrMixedZ%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrMixedZ%nc)        RETURN

csrMixedZ%nnz = csrMixedZ%nnz + 1
csrMixedZ%csrVal(csrMixedZ%nnz) = val
csrMixedZ%csrColIdx(csrMixedZ%nnz) = ic
csrMixedZ%csrRowPtr(ir) = csrMixedZ%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrMixedZ8(csrMixedZ, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_MIXED_COMPLEX) :: csrMixedZ
REAL(8) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrMixedZ%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrMixedZ%nc)        RETURN

csrMixedZ%nnz = csrMixedZ%nnz + 1
csrMixedZ%csrVal(csrMixedZ%nnz) = val
csrMixedZ%csrColIdx(csrMixedZ%nnz) = ic
csrMixedZ%csrRowPtr(ir) = csrMixedZ%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrFloatZ4Z(csrFloatZ, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_FLOAT_COMPLEX) :: csrFloatZ
COMPLEX(4) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrFloatZ%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrFloatZ%nc)        RETURN

csrFloatZ%nnz = csrFloatZ%nnz + 1
csrFloatZ%csrVal(csrFloatZ%nnz) = val
csrFloatZ%csrColIdx(csrFloatZ%nnz) = ic
csrFloatZ%csrRowPtr(ir) = csrFloatZ%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrFloatZ8Z(csrFloatZ, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_FLOAT_COMPLEX) :: csrFloatZ
COMPLEX(8) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0D-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrFloatZ%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrFloatZ%nc)        RETURN

csrFloatZ%nnz = csrFloatZ%nnz + 1
csrFloatZ%csrVal(csrFloatZ%nnz) = val
csrFloatZ%csrColIdx(csrFloatZ%nnz) = ic
csrFloatZ%csrRowPtr(ir) = csrFloatZ%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrDoubleZ4Z(csrDoubleZ, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_DOUBLE_COMPLEX) :: csrDoubleZ
COMPLEX(4) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrDoubleZ%nr)       RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrDoubleZ%nc)       RETURN

csrDoubleZ%nnz = csrDoubleZ%nnz + 1
csrDoubleZ%csrVal(csrDoubleZ%nnz) = val
csrDoubleZ%csrColIdx(csrDoubleZ%nnz) = ic
csrDoubleZ%csrRowPtr(ir) = csrDoubleZ%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrDoubleZ8Z(csrDoubleZ, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_DOUBLE_COMPLEX) :: csrDoubleZ
COMPLEX(8) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrDoubleZ%nr)       RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrDoubleZ%nc)       RETURN

csrDoubleZ%nnz = csrDoubleZ%nnz + 1
csrDoubleZ%csrVal(csrDoubleZ%nnz) = val
csrDoubleZ%csrColIdx(csrDoubleZ%nnz) = ic
csrDoubleZ%csrRowPtr(ir) = csrDoubleZ%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrMixedZ4Z(csrMixedZ, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_MIXED_COMPLEX) :: csrMixedZ
COMPLEX(4) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrMixedZ%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrMixedZ%nc)        RETURN

csrMixedZ%nnz = csrMixedZ%nnz + 1
csrMixedZ%csrVal(csrMixedZ%nnz) = val
csrMixedZ%csrColIdx(csrMixedZ%nnz) = ic
csrMixedZ%csrRowPtr(ir) = csrMixedZ%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrMixedZ8Z(csrMixedZ, val, ir, ic)

IMPLICIT NONE

TYPE(CSR_MIXED_COMPLEX) :: csrMixedZ
COMPLEX(8) :: val
INTEGER :: ir, ic

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrMixedZ%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrMixedZ%nc)        RETURN

csrMixedZ%nnz = csrMixedZ%nnz + 1
csrMixedZ%csrVal(csrMixedZ%nnz) = val
csrMixedZ%csrColIdx(csrMixedZ%nnz) = ic
csrMixedZ%csrRowPtr(ir) = csrMixedZ%csrRowPtr(ir) + 1

END SUBROUTINE

SUBROUTINE pushCsrParallelFloat4(csrFloat, val, ir, ic, thread)

IMPLICIT NONE

TYPE(CSR_FLOAT) :: csrFloat
REAL(4) :: val
INTEGER :: ir, ic
INTEGER :: thread

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrFloat%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrFloat%nc)        RETURN

csrFloat%nnzOmp(thread) = csrFloat%nnzOmp(thread) + 1
csrFloat%bufVal(csrFloat%nnzOmp(thread), thread) = val
csrFloat%bufColIdx(csrFloat%nnzOmp(thread), thread) = ic
csrFloat%bufRowPtr(ir, thread) = csrFloat%bufRowPtr(ir, thread) + 1

END SUBROUTINE

SUBROUTINE pushCsrParallelFloat8(csrFloat, val, ir, ic, thread)

IMPLICIT NONE

TYPE(CSR_FLOAT) :: csrFloat
REAL(8) :: val
INTEGER :: ir, ic
INTEGER :: thread

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrFloat%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrFloat%nc)        RETURN

csrFloat%nnzOmp(thread) = csrFloat%nnzOmp(thread) + 1
csrFloat%bufVal(csrFloat%nnzOmp(thread), thread) = val
csrFloat%bufColIdx(csrFloat%nnzOmp(thread), thread) = ic
csrFloat%bufRowPtr(ir, thread) = csrFloat%bufRowPtr(ir, thread) + 1

END SUBROUTINE

SUBROUTINE pushCsrParallelDouble4(csrDouble, val, ir, ic, thread)

IMPLICIT NONE

TYPE(CSR_DOUBLE) :: csrDouble
REAL(4) :: val
INTEGER :: ir, ic
INTEGER :: thread

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrDouble%nr)       RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrDouble%nc)       RETURN

csrDouble%nnzOmp(thread) = csrDouble%nnzOmp(thread) + 1
csrDouble%bufVal(csrDouble%nnzOmp(thread), thread) = val
csrDouble%bufColIdx(csrDouble%nnzOmp(thread), thread) = ic
csrDouble%bufRowPtr(ir, thread) = csrDouble%bufRowPtr(ir, thread) + 1

END SUBROUTINE

SUBROUTINE pushCsrParallelDouble8(csrDouble, val, ir, ic, thread)

IMPLICIT NONE

TYPE(CSR_DOUBLE) :: csrDouble
REAL(8) :: val
INTEGER :: ir, ic
INTEGER :: thread

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrDouble%nr)       RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrDouble%nc)       RETURN

csrDouble%nnzOmp(thread) = csrDouble%nnzOmp(thread) + 1
csrDouble%bufVal(csrDouble%nnzOmp(thread), thread) = val
csrDouble%bufColIdx(csrDouble%nnzOmp(thread), thread) = ic
csrDouble%bufRowPtr(ir, thread) = csrDouble%bufRowPtr(ir, thread) + 1

END SUBROUTINE

SUBROUTINE pushCsrParallelMixed4(csrMixed, val, ir, ic, thread)

IMPLICIT NONE

TYPE(CSR_MIXED) :: csrMixed
REAL(4) :: val
INTEGER :: ir, ic
INTEGER :: thread

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrMixed%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrMixed%nc)        RETURN

csrMixed%nnzOmp(thread) = csrMixed%nnzOmp(thread) + 1
csrMixed%bufVal(csrMixed%nnzOmp(thread), thread) = val
csrMixed%bufColIdx(csrMixed%nnzOmp(thread), thread) = ic
csrMixed%bufRowPtr(ir, thread) = csrMixed%bufRowPtr(ir, thread) + 1

END SUBROUTINE

SUBROUTINE pushCsrParallelMixed8(csrMixed, val, ir, ic, thread)

IMPLICIT NONE

TYPE(CSR_MIXED) :: csrMixed
REAL(8) :: val
INTEGER :: ir, ic
INTEGER :: thread

IF (abs(val) .LE. 1.0E-10)      RETURN
IF (ir .LE. 0)                  RETURN
IF (ir .GT. csrMixed%nr)        RETURN
IF (ic .LE. 0)                  RETURN
IF (ic .GT. csrMixed%nc)        RETURN

csrMixed%nnzOmp(thread) = csrMixed%nnzOmp(thread) + 1
csrMixed%bufVal(csrMixed%nnzOmp(thread), thread) = val
csrMixed%bufColIdx(csrMixed%nnzOmp(thread), thread) = ic
csrMixed%bufRowPtr(ir, thread) = csrMixed%bufRowPtr(ir, thread) + 1

END SUBROUTINE

SUBROUTINE finalizeCsrFloat(csrFloat, lDevice)

IMPLICIT NONE

TYPE(CSR_FLOAT) :: csrFloat
LOGICAL :: lDevice
REAL(4), POINTER :: csrVal(:)
INTEGER, POINTER :: csrColIdx(:)
INTEGER, ALLOCATABLE :: csrRowPtr(:)
INTEGER :: i, ierr

IF (csrFloat%lParallel) CALL collapseCsrParallelBufferFloat(csrFloat)

IF (.NOT. csrFloat%lFinalized) THEN

  csrVal => csrFloat%csrVal
  csrColIdx => csrFloat%csrColIdx

  ALLOCATE(csrFloat%csrVal(csrFloat%nnz))
  ALLOCATE(csrFloat%csrColIdx(csrFloat%nnz))

  DO i = 1, csrFloat%nnz
    csrFloat%csrVal(i) = csrVal(i)
    csrFloat%csrColIdx(i) = csrColIdx(i)
  ENDDO

  DEALLOCATE(csrVal)
  DEALLOCATE(csrColIdx)

  ALLOCATE(csrRowPtr(csrFloat%nr + 1))

  csrRowPtr = csrFloat%csrRowPtr
  csrFloat%csrRowPtr(1) = 1
  DO i = 2, csrFloat%nr + 1
    csrFloat%csrRowPtr(i) = csrFloat%csrRowPtr(i - 1) + csrRowPtr(i - 1)
  ENDDO

  DEALLOCATE(csrRowPtr)

  csrFloat%lFinalized = .TRUE.

ENDIF

#ifdef __PGI
IF (lDevice) THEN
  IF (.NOT. csrFloat%lDevice) THEN
    ALLOCATE(csrFloat%d_csrVal(csrFloat%nnz)); csrFloat%d_csrVal = csrFloat%csrVal
    ALLOCATE(csrFloat%d_csrColIdx(csrFloat%nnz)); csrFloat%d_csrColIdx = csrFloat%csrColIdx
    ALLOCATE(csrFloat%d_csrRowPtr(csrFloat%nr + 1)); csrFloat%d_csrRowPtr = csrFloat%csrRowPtr
    csrFloat%lDevice = .TRUE.
  ENDIF
  ierr = cusparseSetMatIndexBase(csrFloat%descr, CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatIndexBase(csrFloat%descrLU(1), CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatIndexBase(csrFloat%descrLU(2), CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatType(csrFloat%descr, CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatType(csrFloat%descrLU(1), CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatType(csrFloat%descrLU(2), CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatFillMode(csrFloat%descrLU(1), CUSPARSE_FILL_MODE_LOWER)
  ierr = cusparseSetMatFillMode(csrFloat%descrLU(2), CUSPARSE_FILL_MODE_UPPER)
  ierr = cusparseSetMatDiagType(csrFloat%descrLU(1), CUSPARSE_DIAG_TYPE_NON_UNIT)
  ierr = cusparseSetMatDiagType(csrFloat%descrLU(2), CUSPARSE_DIAG_TYPE_NON_UNIT)
ENDIF
#endif

END SUBROUTINE

SUBROUTINE finalizeCsrDouble(csrDouble, lDevice)

IMPLICIT NONE

TYPE(CSR_DOUBLE) :: csrDouble
LOGICAL :: lDevice
REAL(8), POINTER :: csrVal(:)
INTEGER, POINTER :: csrColIdx(:)
INTEGER, ALLOCATABLE :: csrRowPtr(:)
INTEGER :: i, ierr

IF (csrDouble%lParallel) CALL collapseCsrParallelBufferDouble(csrDouble)

IF (.NOT. csrDouble%lFinalized) THEN

  csrVal => csrDouble%csrVal
  csrColIdx => csrDouble%csrColIdx

  ALLOCATE(csrDouble%csrVal(csrDouble%nnz))
  ALLOCATE(csrDouble%csrColIdx(csrDouble%nnz))

  DO i = 1, csrDouble%nnz
    csrDouble%csrVal(i) = csrVal(i)
    csrDouble%csrColIdx(i) = csrColIdx(i)
  ENDDO

  DEALLOCATE(csrVal)
  DEALLOCATE(csrColIdx)

  ALLOCATE(csrRowPtr(csrDouble%nr + 1))

  csrRowPtr = csrDouble%csrRowPtr
  csrDouble%csrRowPtr(1) = 1
  DO i = 2, csrDouble%nr + 1
    csrDouble%csrRowPtr(i) = csrDouble%csrRowPtr(i - 1) + csrRowPtr(i - 1)
  ENDDO

  DEALLOCATE(csrRowPtr)

  csrDouble%lFinalized = .TRUE.

ENDIF

#ifdef __PGI
IF (lDevice) THEN
  IF (.NOT. csrDouble%lDevice) THEN
    ALLOCATE(csrDouble%d_csrVal(csrDouble%nnz)); csrDouble%d_csrVal = csrDouble%csrVal
    ALLOCATE(csrDouble%d_csrColIdx(csrDouble%nnz)); csrDouble%d_csrColIdx = csrDouble%csrColIdx
    ALLOCATE(csrDouble%d_csrRowPtr(csrDouble%nr + 1)); csrDouble%d_csrRowPtr = csrDouble%csrRowPtr
    csrDouble%lDevice = .TRUE.
  ENDIF
  ierr = cusparseSetMatIndexBase(csrDouble%descr, CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatIndexBase(csrDouble%descrLU(1), CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatIndexBase(csrDouble%descrLU(2), CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatType(csrDouble%descr, CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatType(csrDouble%descrLU(1), CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatType(csrDouble%descrLU(2), CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatFillMode(csrDouble%descrLU(1), CUSPARSE_FILL_MODE_LOWER)
  ierr = cusparseSetMatFillMode(csrDouble%descrLU(2), CUSPARSE_FILL_MODE_UPPER)
  ierr = cusparseSetMatDiagType(csrDouble%descrLU(1), CUSPARSE_DIAG_TYPE_NON_UNIT)
  ierr = cusparseSetMatDiagType(csrDouble%descrLU(2), CUSPARSE_DIAG_TYPE_NON_UNIT)
ENDIF
#endif

END SUBROUTINE

SUBROUTINE finalizeCsrMixed(csrMixed, lDevice)

IMPLICIT NONE

TYPE(CSR_MIXED) :: csrMixed
LOGICAL :: lDevice
REAL(8), POINTER :: csrVal(:)
INTEGER, POINTER :: csrColIdx(:)
INTEGER, ALLOCATABLE :: csrRowPtr(:)
INTEGER :: i, ierr

IF (csrMixed%lParallel) CALL collapseCsrParallelBufferMixed(csrMixed)

IF (.NOT. csrMixed%lFinalized) THEN

  csrVal => csrMixed%csrVal
  csrColIdx => csrMixed%csrColIdx

  ALLOCATE(csrMixed%csrVal(csrMixed%nnz))
  ALLOCATE(csrMixed%csrColIdx(csrMixed%nnz))

  DO i = 1, csrMixed%nnz
    csrMixed%csrVal(i) = csrVal(i)
    csrMixed%csrColIdx(i) = csrColIdx(i)
  ENDDO

  DEALLOCATE(csrVal)
  DEALLOCATE(csrColIdx)

  ALLOCATE(csrRowPtr(csrMixed%nr + 1))

  csrRowPtr = csrMixed%csrRowPtr
  csrMixed%csrRowPtr(1) = 1
  DO i = 2, csrMixed%nr + 1
    csrMixed%csrRowPtr(i) = csrMixed%csrRowPtr(i - 1) + csrRowPtr(i - 1)
  ENDDO

  DEALLOCATE(csrRowPtr)

  csrMixed%lFinalized = .TRUE.

ENDIF

#ifdef __PGI
IF (lDevice) THEN
  IF (.NOT. csrMixed%lDevice) THEN
    ALLOCATE(csrMixed%d_csrVal(csrMixed%nnz)); csrMixed%d_csrVal = csrMixed%csrVal
    ALLOCATE(csrMixed%d_csrVal8(csrMixed%nnz)); csrMixed%d_csrVal8 = csrMixed%csrVal
    ALLOCATE(csrMixed%d_csrColIdx(csrMixed%nnz)); csrMixed%d_csrColIdx = csrMixed%csrColIdx
    ALLOCATE(csrMixed%d_csrRowPtr(csrMixed%nr + 1)); csrMixed%d_csrRowPtr = csrMixed%csrRowPtr
    csrMixed%lDevice = .TRUE.
  ENDIF
  ierr = cusparseSetMatIndexBase(csrMixed%descr, CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatIndexBase(csrMixed%descrLU(1), CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatIndexBase(csrMixed%descrLU(2), CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatType(csrMixed%descr, CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatType(csrMixed%descrLU(1), CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatType(csrMixed%descrLU(2), CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatFillMode(csrMixed%descrLU(1), CUSPARSE_FILL_MODE_LOWER)
  ierr = cusparseSetMatFillMode(csrMixed%descrLU(2), CUSPARSE_FILL_MODE_UPPER)
  ierr = cusparseSetMatDiagType(csrMixed%descrLU(1), CUSPARSE_DIAG_TYPE_NON_UNIT)
  ierr = cusparseSetMatDiagType(csrMixed%descrLU(2), CUSPARSE_DIAG_TYPE_NON_UNIT)
ENDIF
#endif

END SUBROUTINE

SUBROUTINE finalizeSortCsrFloat(csrFloat, lDevice)

IMPLICIT NONE

TYPE(CSR_FLOAT) :: csrFloat
LOGICAL :: lDevice
REAL(4), ALLOCATABLE :: rowVal(:, :)
REAL(4) :: val
INTEGER, ALLOCATABLE :: rowCol(:, :)
INTEGER :: maxRowEntry
INTEGER :: idx, ic, ir, i, j

IF (csrFloat%lFinalized) RETURN
IF (csrFloat%lParallel) CALL collapseCsrParallelBufferFloat(csrFloat)

maxRowEntry = maxval(csrFloat%csrRowPtr)

ALLOCATE(rowVal(maxRowEntry, csrFloat%nr))
ALLOCATE(rowCol(maxRowEntry, csrFloat%nr))

idx = 0

DO ir = 1, csrFloat%nr
  DO ic = 1, csrFloat%csrRowPtr(ir)
    idx = idx + 1
    rowVal(ic, ir) = csrFloat%csrVal(idx)
    rowCol(ic, ir) = csrFloat%csrColIdx(idx)
  ENDDO
ENDDO

!$OMP PARALLEL PRIVATE(val, ic)
!$OMP DO SCHEDULE(GUIDED)
DO ir = 1, csrFloat%nr
  DO j = csrFloat%csrRowPtr(ir), 1, -1
    DO i = 1, j - 1
      IF (rowCol(i, ir) .GT. rowCol(i + 1, ir)) THEN
        val = rowVal(i, ir)
        rowVal(i, ir) = rowVal(i + 1, ir); rowVal(i + 1, ir) = val
        ic = rowCol(i, ir)
        rowCol(i, ir) = rowCol(i + 1, ir); rowCol(i + 1, ir) = ic
      ENDIF
    ENDDO
  ENDDO
ENDDO
!$OMP END DO
!$OMP END PARALLEL

idx = 0

DO ir = 1, csrFloat%nr
  DO ic = 1, csrFloat%csrRowPtr(ir)
    idx = idx + 1
    csrFloat%csrVal(idx) = rowVal(ic, ir)
    csrFloat%csrColIdx(idx) = rowCol(ic, ir)
  ENDDO
ENDDO

CALL finalizeCsrFloat(csrFloat, lDevice)

DEALLOCATE(rowVal)
DEALLOCATE(rowCol)

END SUBROUTINE

SUBROUTINE finalizeSortCsrDouble(csrDouble, lDevice)

IMPLICIT NONE

TYPE(CSR_DOUBLE) :: csrDouble
LOGICAL :: lDevice
REAL(8), ALLOCATABLE :: rowVal(:, :)
REAL(8) :: val
INTEGER, ALLOCATABLE :: rowCol(:, :)
INTEGER :: maxRowEntry
INTEGER :: idx, ic, ir, i, j

IF (csrDouble%lFinalized) RETURN
IF (csrDouble%lParallel) CALL collapseCsrParallelBufferDouble(csrDouble)

maxRowEntry = maxval(csrDouble%csrRowPtr)

ALLOCATE(rowVal(maxRowEntry, csrDouble%nr))
ALLOCATE(rowCol(maxRowEntry, csrDouble%nr))

idx = 0

DO ir = 1, csrDouble%nr
  DO ic = 1, csrDouble%csrRowPtr(ir)
    idx = idx + 1
    rowVal(ic, ir) = csrDouble%csrVal(idx)
    rowCol(ic, ir) = csrDouble%csrColIdx(idx)
  ENDDO
ENDDO

!$OMP PARALLEL PRIVATE(val, ic)
!$OMP DO SCHEDULE(GUIDED)
DO ir = 1, csrDouble%nr
  DO j = csrDouble%csrRowPtr(ir), 1, -1
    DO i = 1, j - 1
      IF (rowCol(i, ir) .GT. rowCol(i + 1, ir)) THEN
        val = rowVal(i, ir)
        rowVal(i, ir) = rowVal(i + 1, ir); rowVal(i + 1, ir) = val
        ic = rowCol(i, ir)
        rowCol(i, ir) = rowCol(i + 1, ir); rowCol(i + 1, ir) = ic
      ENDIF
    ENDDO
  ENDDO
ENDDO
!$OMP END DO
!$OMP END PARALLEL

idx = 0

DO ir = 1, csrDouble%nr
  DO ic = 1, csrDouble%csrRowPtr(ir)
    idx = idx + 1
    csrDouble%csrVal(idx) = rowVal(ic, ir)
    csrDouble%csrColIdx(idx) = rowCol(ic, ir)
  ENDDO
ENDDO

CALL finalizeCsrDouble(csrDouble, lDevice)

DEALLOCATE(rowVal)
DEALLOCATE(rowCol)

END SUBROUTINE

SUBROUTINE finalizeSortCsrMixed(csrMixed, lDevice)

IMPLICIT NONE

TYPE(CSR_MIXED) :: csrMixed
LOGICAL :: lDevice
REAL(8), ALLOCATABLE :: rowVal(:, :)
REAL(8) :: val
INTEGER, ALLOCATABLE :: rowCol(:, :)
INTEGER :: maxRowEntry
INTEGER :: idx, ic, ir, i, j

IF (csrMixed%lFinalized) RETURN
IF (csrMixed%lParallel) CALL collapseCsrParallelBufferMixed(csrMixed)

maxRowEntry = maxval(csrMixed%csrRowPtr)

ALLOCATE(rowVal(maxRowEntry, csrMixed%nr))
ALLOCATE(rowCol(maxRowEntry, csrMixed%nr))

idx = 0

DO ir = 1, csrMixed%nr
  DO ic = 1, csrMixed%csrRowPtr(ir)
    idx = idx + 1
    rowVal(ic, ir) = csrMixed%csrVal(idx)
    rowCol(ic, ir) = csrMixed%csrColIdx(idx)
  ENDDO
ENDDO

!$OMP PARALLEL PRIVATE(val, ic)
!$OMP DO SCHEDULE(GUIDED)
DO ir = 1, csrMixed%nr
  DO j = csrMixed%csrRowPtr(ir), 1, -1
    DO i = 1, j - 1
      IF (rowCol(i, ir) .GT. rowCol(i + 1, ir)) THEN
        val = rowVal(i, ir)
        rowVal(i, ir) = rowVal(i + 1, ir); rowVal(i + 1, ir) = val
        ic = rowCol(i, ir)
        rowCol(i, ir) = rowCol(i + 1, ir); rowCol(i + 1, ir) = ic
      ENDIF
    ENDDO
  ENDDO
ENDDO
!$OMP END DO
!$OMP END PARALLEL

idx = 0

DO ir = 1, csrMixed%nr
  DO ic = 1, csrMixed%csrRowPtr(ir)
    idx = idx + 1
    csrMixed%csrVal(idx) = rowVal(ic, ir)
    csrMixed%csrColIdx(idx) = rowCol(ic, ir)
  ENDDO
ENDDO

CALL finalizeCsrMixed(csrMixed, lDevice)

DEALLOCATE(rowVal)
DEALLOCATE(rowCol)

END SUBROUTINE

SUBROUTINE finalizeCsrFloatZ(csrFloatZ, lDevice)

IMPLICIT NONE

TYPE(CSR_FLOAT_COMPLEX) :: csrFloatZ
LOGICAL :: lDevice
COMPLEX(4), POINTER :: csrVal(:)
INTEGER, POINTER :: csrColIdx(:)
INTEGER, ALLOCATABLE :: csrRowPtr(:)
INTEGER :: i, ierr

!IF (csrFloatZ%lParallel) CALL collapseCsrParallelBufferFloat(csrFloatZ)

IF (.NOT. csrFloatZ%lFinalized) THEN

  csrVal => csrFloatZ%csrVal
  csrColIdx => csrFloatZ%csrColIdx

  ALLOCATE(csrFloatZ%csrVal(csrFloatZ%nnz))
  ALLOCATE(csrFloatZ%csrColIdx(csrFloatZ%nnz))

  DO i = 1, csrFloatZ%nnz
    csrFloatZ%csrVal(i) = csrVal(i)
    csrFloatZ%csrColIdx(i) = csrColIdx(i)
  ENDDO

  DEALLOCATE(csrVal)
  DEALLOCATE(csrColIdx)

  ALLOCATE(csrRowPtr(csrFloatZ%nr + 1))

  csrRowPtr = csrFloatZ%csrRowPtr
  csrFloatZ%csrRowPtr(1) = 1
  DO i = 2, csrFloatZ%nr + 1
    csrFloatZ%csrRowPtr(i) = csrFloatZ%csrRowPtr(i - 1) + csrRowPtr(i - 1)
  ENDDO

  DEALLOCATE(csrRowPtr)

  csrFloatZ%lFinalized = .TRUE.

ENDIF

#ifdef __PGI
IF (lDevice) THEN
  IF (.NOT. csrFloatZ%lDevice) THEN
    ALLOCATE(csrFloatZ%d_csrVal(csrFloatZ%nnz)); csrFloatZ%d_csrVal = csrFloatZ%csrVal
    ALLOCATE(csrFloatZ%d_csrColIdx(csrFloatZ%nnz)); csrFloatZ%d_csrColIdx = csrFloatZ%csrColIdx
    ALLOCATE(csrFloatZ%d_csrRowPtr(csrFloatZ%nr + 1)); csrFloatZ%d_csrRowPtr = csrFloatZ%csrRowPtr
    csrFloatZ%lDevice = .TRUE.
  ENDIF
  ierr = cusparseSetMatIndexBase(csrFloatZ%descr, CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatIndexBase(csrFloatZ%descrLU(1), CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatIndexBase(csrFloatZ%descrLU(2), CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatType(csrFloatZ%descr, CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatType(csrFloatZ%descrLU(1), CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatType(csrFloatZ%descrLU(2), CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatFillMode(csrFloatZ%descrLU(1), CUSPARSE_FILL_MODE_LOWER)
  ierr = cusparseSetMatFillMode(csrFloatZ%descrLU(2), CUSPARSE_FILL_MODE_UPPER)
  ierr = cusparseSetMatDiagType(csrFloatZ%descrLU(1), CUSPARSE_DIAG_TYPE_NON_UNIT)
  ierr = cusparseSetMatDiagType(csrFloatZ%descrLU(2), CUSPARSE_DIAG_TYPE_NON_UNIT)
ENDIF
#endif

END SUBROUTINE

SUBROUTINE finalizeCsrDoubleZ(csrDoubleZ, lDevice)

IMPLICIT NONE

TYPE(CSR_DOUBLE_COMPLEX) :: csrDoubleZ
LOGICAL :: lDevice
COMPLEX(8), POINTER :: csrVal(:)
INTEGER, POINTER :: csrColIdx(:)
INTEGER, ALLOCATABLE :: csrRowPtr(:)
INTEGER :: i, ierr

!IF (csrDoubleZ%lParallel) CALL collapseCsrParallelBufferDouble(csrDoubleZ)

IF (.NOT. csrDoubleZ%lFinalized) THEN

  csrVal => csrDoubleZ%csrVal
  csrColIdx => csrDoubleZ%csrColIdx

  ALLOCATE(csrDoubleZ%csrVal(csrDoubleZ%nnz))
  ALLOCATE(csrDoubleZ%csrColIdx(csrDoubleZ%nnz))

  DO i = 1, csrDoubleZ%nnz
    csrDoubleZ%csrVal(i) = csrVal(i)
    csrDoubleZ%csrColIdx(i) = csrColIdx(i)
  ENDDO

  DEALLOCATE(csrVal)
  DEALLOCATE(csrColIdx)

  ALLOCATE(csrRowPtr(csrDoubleZ%nr + 1))

  csrRowPtr = csrDoubleZ%csrRowPtr
  csrDoubleZ%csrRowPtr(1) = 1
  DO i = 2, csrDoubleZ%nr + 1
    csrDoubleZ%csrRowPtr(i) = csrDoubleZ%csrRowPtr(i - 1) + csrRowPtr(i - 1)
  ENDDO

  DEALLOCATE(csrRowPtr)

  csrDoubleZ%lFinalized = .TRUE.

ENDIF

#ifdef __PGI
IF (lDevice) THEN
  IF (.NOT. csrDoubleZ%lDevice) THEN
    ALLOCATE(csrDoubleZ%d_csrVal(csrDoubleZ%nnz)); csrDoubleZ%d_csrVal = csrDoubleZ%csrVal
    ALLOCATE(csrDoubleZ%d_csrColIdx(csrDoubleZ%nnz)); csrDoubleZ%d_csrColIdx = csrDoubleZ%csrColIdx
    ALLOCATE(csrDoubleZ%d_csrRowPtr(csrDoubleZ%nr + 1)); csrDoubleZ%d_csrRowPtr = csrDoubleZ%csrRowPtr
    csrDoubleZ%lDevice = .TRUE.
  ENDIF
  ierr = cusparseSetMatIndexBase(csrDoubleZ%descr, CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatIndexBase(csrDoubleZ%descrLU(1), CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatIndexBase(csrDoubleZ%descrLU(2), CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatType(csrDoubleZ%descr, CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatType(csrDoubleZ%descrLU(1), CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatType(csrDoubleZ%descrLU(2), CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatFillMode(csrDoubleZ%descrLU(1), CUSPARSE_FILL_MODE_LOWER)
  ierr = cusparseSetMatFillMode(csrDoubleZ%descrLU(2), CUSPARSE_FILL_MODE_UPPER)
  ierr = cusparseSetMatDiagType(csrDoubleZ%descrLU(1), CUSPARSE_DIAG_TYPE_NON_UNIT)
  ierr = cusparseSetMatDiagType(csrDoubleZ%descrLU(2), CUSPARSE_DIAG_TYPE_NON_UNIT)
ENDIF
#endif

END SUBROUTINE

SUBROUTINE finalizeCsrMixedZ(csrMixedZ, lDevice)

IMPLICIT NONE

TYPE(CSR_MIXED_COMPLEX) :: csrMixedZ
LOGICAL :: lDevice
COMPLEX(8), POINTER :: csrVal(:)
INTEGER, POINTER :: csrColIdx(:)
INTEGER, ALLOCATABLE :: csrRowPtr(:)
INTEGER :: i, ierr

!IF (csrMixedZ%lParallel) CALL collapseCsrParallelBufferMixed(csrMixedZ)

IF (.NOT. csrMixedZ%lFinalized) THEN

  csrVal => csrMixedZ%csrVal
  csrColIdx => csrMixedZ%csrColIdx

  ALLOCATE(csrMixedZ%csrVal(csrMixedZ%nnz))
  ALLOCATE(csrMixedZ%csrColIdx(csrMixedZ%nnz))

  DO i = 1, csrMixedZ%nnz
    csrMixedZ%csrVal(i) = csrVal(i)
    csrMixedZ%csrColIdx(i) = csrColIdx(i)
  ENDDO

  DEALLOCATE(csrVal)
  DEALLOCATE(csrColIdx)

  ALLOCATE(csrRowPtr(csrMixedZ%nr + 1))

  csrRowPtr = csrMixedZ%csrRowPtr
  csrMixedZ%csrRowPtr(1) = 1
  DO i = 2, csrMixedZ%nr + 1
    csrMixedZ%csrRowPtr(i) = csrMixedZ%csrRowPtr(i - 1) + csrRowPtr(i - 1)
  ENDDO

  DEALLOCATE(csrRowPtr)

  csrMixedZ%lFinalized = .TRUE.

ENDIF

#ifdef __PGI
IF (lDevice) THEN
  IF (.NOT. csrMixedZ%lDevice) THEN
    ALLOCATE(csrMixedZ%d_csrVal(csrMixedZ%nnz)); csrMixedZ%d_csrVal = csrMixedZ%csrVal
    ALLOCATE(csrMixedZ%d_csrVal8(csrMixedZ%nnz)); csrMixedZ%d_csrVal8 = csrMixedZ%csrVal
    ALLOCATE(csrMixedZ%d_csrColIdx(csrMixedZ%nnz)); csrMixedZ%d_csrColIdx = csrMixedZ%csrColIdx
    ALLOCATE(csrMixedZ%d_csrRowPtr(csrMixedZ%nr + 1)); csrMixedZ%d_csrRowPtr = csrMixedZ%csrRowPtr
    csrMixedZ%lDevice = .TRUE.
  ENDIF
  ierr = cusparseSetMatIndexBase(csrMixedZ%descr, CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatIndexBase(csrMixedZ%descrLU(1), CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatIndexBase(csrMixedZ%descrLU(2), CUSPARSE_INDEX_BASE_ONE)
  ierr = cusparseSetMatType(csrMixedZ%descr, CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatType(csrMixedZ%descrLU(1), CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatType(csrMixedZ%descrLU(2), CUSPARSE_MATRIX_TYPE_GENERAL)
  ierr = cusparseSetMatFillMode(csrMixedZ%descrLU(1), CUSPARSE_FILL_MODE_LOWER)
  ierr = cusparseSetMatFillMode(csrMixedZ%descrLU(2), CUSPARSE_FILL_MODE_UPPER)
  ierr = cusparseSetMatDiagType(csrMixedZ%descrLU(1), CUSPARSE_DIAG_TYPE_NON_UNIT)
  ierr = cusparseSetMatDiagType(csrMixedZ%descrLU(2), CUSPARSE_DIAG_TYPE_NON_UNIT)
ENDIF
#endif

END SUBROUTINE
!
!SUBROUTINE finalizeSortCsrFloatZ(csrFloatZ, lDevice)
!
!IMPLICIT NONE
!
!TYPE(CSR_FLOAT_COMPLEX) :: csrFloatZ
!LOGICAL :: lDevice
!COMPLEX(4), ALLOCATABLE :: rowVal(:, :)
!COMPLEX(4) :: val
!INTEGER, ALLOCATABLE :: rowCol(:, :)
!INTEGER :: maxRowEntry
!INTEGER :: idx, ic, ir, i, j
!
!IF (csrFloatZ%lFinalized) RETURN
!!IF (csrFloatZ%lParallel) CALL collapseCsrParallelBufferFloat(csrFloatZ)
!
!maxRowEntry = maxval(csrFloatZ%csrRowPtr)
!
!ALLOCATE(rowVal(maxRowEntry, csrFloatZ%nr))
!ALLOCATE(rowCol(maxRowEntry, csrFloatZ%nr))
!
!idx = 0
!
!DO ir = 1, csrFloatZ%nr
!  DO ic = 1, csrFloatZ%csrRowPtr(ir)
!    idx = idx + 1
!    rowVal(ic, ir) = csrFloatZ%csrVal(idx)
!    rowCol(ic, ir) = csrFloatZ%csrColIdx(idx)
!  ENDDO
!ENDDO
!
!!$OMP PARALLEL PRIVATE(val, ic)
!!$OMP DO SCHEDULE(GUIDED)
!DO ir = 1, csrFloatZ%nr
!  DO j = csrFloatZ%csrRowPtr(ir), 1, -1
!    DO i = 1, j - 1
!      IF (rowCol(i, ir) .GT. rowCol(i + 1, ir)) THEN
!        val = rowVal(i, ir)
!        rowVal(i, ir) = rowVal(i + 1, ir); rowVal(i + 1, ir) = val
!        ic = rowCol(i, ir)
!        rowCol(i, ir) = rowCol(i + 1, ir); rowCol(i + 1, ir) = ic
!      ENDIF
!    ENDDO
!  ENDDO
!ENDDO
!!$OMP END DO
!!$OMP END PARALLEL
!
!idx = 0
!
!DO ir = 1, csrFloatZ%nr
!  DO ic = 1, csrFloatZ%csrRowPtr(ir)
!    idx = idx + 1
!    csrFloatZ%csrVal(idx) = rowVal(ic, ir)
!    csrFloatZ%csrColIdx(idx) = rowCol(ic, ir)
!  ENDDO
!ENDDO
!
!CALL finalizecsrFloatZ(csrFloatZ, lDevice)
!
!DEALLOCATE(rowVal)
!DEALLOCATE(rowCol)
!
!END SUBROUTINE
!
!SUBROUTINE finalizeSortcsrDoubleZ(csrDoubleZ, lDevice)
!
!IMPLICIT NONE
!
!TYPE(CSR_DOUBLE) :: csrDoubleZ
!LOGICAL :: lDevice
!COMPLEX(8), ALLOCATABLE :: rowVal(:, :)
!COMPLEX(8) :: val
!INTEGER, ALLOCATABLE :: rowCol(:, :)
!INTEGER :: maxRowEntry
!INTEGER :: idx, ic, ir, i, j
!
!IF (csrDoubleZ%lFinalized) RETURN
!!IF (csrDoubleZ%lParallel) CALL collapseCsrParallelBufferDouble(csrDoubleZ)
!
!maxRowEntry = maxval(csrDoubleZ%csrRowPtr)
!
!ALLOCATE(rowVal(maxRowEntry, csrDoubleZ%nr))
!ALLOCATE(rowCol(maxRowEntry, csrDoubleZ%nr))
!
!idx = 0
!
!DO ir = 1, csrDoubleZ%nr
!  DO ic = 1, csrDoubleZ%csrRowPtr(ir)
!    idx = idx + 1
!    rowVal(ic, ir) = csrDoubleZ%csrVal(idx)
!    rowCol(ic, ir) = csrDoubleZ%csrColIdx(idx)
!  ENDDO
!ENDDO
!
!!$OMP PARALLEL PRIVATE(val, ic)
!!$OMP DO SCHEDULE(GUIDED)
!DO ir = 1, csrDoubleZ%nr
!  DO j = csrDoubleZ%csrRowPtr(ir), 1, -1
!    DO i = 1, j - 1
!      IF (rowCol(i, ir) .GT. rowCol(i + 1, ir)) THEN
!        val = rowVal(i, ir)
!        rowVal(i, ir) = rowVal(i + 1, ir); rowVal(i + 1, ir) = val
!        ic = rowCol(i, ir)
!        rowCol(i, ir) = rowCol(i + 1, ir); rowCol(i + 1, ir) = ic
!      ENDIF
!    ENDDO
!  ENDDO
!ENDDO
!!$OMP END DO
!!$OMP END PARALLEL
!
!idx = 0
!
!DO ir = 1, csrDoubleZ%nr
!  DO ic = 1, csrDoubleZ%csrRowPtr(ir)
!    idx = idx + 1
!    csrDoubleZ%csrVal(idx) = rowVal(ic, ir)
!    csrDoubleZ%csrColIdx(idx) = rowCol(ic, ir)
!  ENDDO
!ENDDO
!
!CALL finalizecsrDoubleZ(csrDoubleZ, lDevice)
!
!DEALLOCATE(rowVal)
!DEALLOCATE(rowCol)
!
!END SUBROUTINE
!
!SUBROUTINE finalizeSortcsrMixedZ(csrMixedZ, lDevice)
!
!IMPLICIT NONE
!
!TYPE(CSR_MIXED) :: csrMixedZ
!LOGICAL :: lDevice
!COMPLEX(8), ALLOCATABLE :: rowVal(:, :)
!COMPLEX(8) :: val
!INTEGER, ALLOCATABLE :: rowCol(:, :)
!INTEGER :: maxRowEntry
!INTEGER :: idx, ic, ir, i, j
!
!IF (csrMixedZ%lFinalized) RETURN
!!IF (csrMixedZ%lParallel) CALL collapseCsrParallelBufferMixed(csrMixedZ)
!
!maxRowEntry = maxval(csrMixedZ%csrRowPtr)
!
!ALLOCATE(rowVal(maxRowEntry, csrMixedZ%nr))
!ALLOCATE(rowCol(maxRowEntry, csrMixedZ%nr))
!
!idx = 0
!
!DO ir = 1, csrMixedZ%nr
!  DO ic = 1, csrMixedZ%csrRowPtr(ir)
!    idx = idx + 1
!    rowVal(ic, ir) = csrMixedZ%csrVal(idx)
!    rowCol(ic, ir) = csrMixedZ%csrColIdx(idx)
!  ENDDO
!ENDDO
!
!!$OMP PARALLEL PRIVATE(val, ic)
!!$OMP DO SCHEDULE(GUIDED)
!DO ir = 1, csrMixedZ%nr
!  DO j = csrMixedZ%csrRowPtr(ir), 1, -1
!    DO i = 1, j - 1
!      IF (rowCol(i, ir) .GT. rowCol(i + 1, ir)) THEN
!        val = rowVal(i, ir)
!        rowVal(i, ir) = rowVal(i + 1, ir); rowVal(i + 1, ir) = val
!        ic = rowCol(i, ir)
!        rowCol(i, ir) = rowCol(i + 1, ir); rowCol(i + 1, ir) = ic
!      ENDIF
!    ENDDO
!  ENDDO
!ENDDO
!!$OMP END DO
!!$OMP END PARALLEL
!
!idx = 0
!
!DO ir = 1, csrMixedZ%nr
!  DO ic = 1, csrMixedZ%csrRowPtr(ir)
!    idx = idx + 1
!    csrMixedZ%csrVal(idx) = rowVal(ic, ir)
!    csrMixedZ%csrColIdx(idx) = rowCol(ic, ir)
!  ENDDO
!ENDDO
!
!CALL finalizecsrMixedZ(csrMixedZ, lDevice)
!
!DEALLOCATE(rowVal)
!DEALLOCATE(rowCol)
!
!END SUBROUTINE

SUBROUTINE printCsrFloat(csrFloat, filename, io)

IMPLICIT NONE

TYPE(CSR_FLOAT) :: csrFloat
CHARACTER(*) :: filename
INTEGER :: io
INTEGER :: i

OPEN(io, FILE = filename)
WRITE(io, *), csrFloat%nr, csrFloat%nc, csrFloat%nnz
DO i = 1, csrFloat%nnz
  IF (i .LE. csrFloat%nr + 1) THEN
    WRITE(io, *), csrFloat%csrVal(i), csrFloat%csrColIdx(i), csrFloat%csrRowPtr(i)
  ELSE
    WRITE(io, *), csrFloat%csrVal(i), csrFloat%csrColIdx(i), 0
  ENDIF
ENDDO
CLOSE(io)

END SUBROUTINE

SUBROUTINE printCsrDouble(csrDouble, filename, io)

IMPLICIT NONE

TYPE(CSR_DOUBLE) :: csrDouble
CHARACTER(*) :: filename
INTEGER :: io
INTEGER :: i

OPEN(io, FILE = filename)
WRITE(io, *), csrDouble%nr, csrDouble%nc, csrDouble%nnz
DO i = 1, csrDouble%nnz
  IF (i .LE. csrDouble%nr + 1) THEN
    WRITE(io, *), csrDouble%csrVal(i), csrDouble%csrColIdx(i), csrDouble%csrRowPtr(i)
  ELSE
    WRITE(io, *), csrDouble%csrVal(i), csrDouble%csrColIdx(i), 0
  ENDIF
ENDDO
CLOSE(io)

END SUBROUTINE

SUBROUTINE printCsrMixed(csrMixed, filename, io)

IMPLICIT NONE

TYPE(CSR_MIXED) :: csrMixed
CHARACTER(*) :: filename
INTEGER :: io
INTEGER :: i

OPEN(io, FILE = filename)
WRITE(io, *), csrMixed%nr, csrMixed%nc, csrMixed%nnz
DO i = 1, csrMixed%nnz
  IF (i .LE. csrMixed%nr + 1) THEN
    WRITE(io, *), csrMixed%csrVal(i), csrMixed%csrColIdx(i), csrMixed%csrRowPtr(i)
  ELSE
    WRITE(io, *), csrMixed%csrVal(i), csrMixed%csrColIdx(i), 0
  ENDIF
ENDDO
CLOSE(io)

END SUBROUTINE

SUBROUTINE copyCsrFloat(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_FLOAT), INTENT(OUT) :: outputCsr
TYPE(CSR_FLOAT), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrDouble(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_DOUBLE), INTENT(OUT) :: outputCsr
TYPE(CSR_DOUBLE), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrMixed(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_MIXED), INTENT(OUT) :: outputCsr
TYPE(CSR_MIXED), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrFloatZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_FLOAT_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_FLOAT_COMPLEX), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrDoubleZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_DOUBLE_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_DOUBLE_COMPLEX), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrMixedZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_MIXED_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_MIXED_COMPLEX), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrFloat2Double(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_DOUBLE), INTENT(OUT) :: outputCsr
TYPE(CSR_FLOAT), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrFloat2Mixed(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_MIXED), INTENT(OUT) :: outputCsr
TYPE(CSR_FLOAT), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrDouble2Float(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_FLOAT), INTENT(OUT) :: outputCsr
TYPE(CSR_DOUBLE), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrDouble2Mixed(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_MIXED), INTENT(OUT) :: outputCsr
TYPE(CSR_DOUBLE), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrMixed2Float(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_FLOAT), INTENT(OUT) :: outputCsr
TYPE(CSR_MIXED), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrMixed2Double(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_DOUBLE), INTENT(OUT) :: outputCsr
TYPE(CSR_MIXED), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrFloat2FloatZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_FLOAT_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_FLOAT), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrFloat2DoubleZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_DOUBLE_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_FLOAT), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrFloat2MixedZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_MIXED_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_FLOAT), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrDouble2FloatZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_FLOAT_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_DOUBLE), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrDouble2DoubleZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_DOUBLE_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_DOUBLE), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrDouble2MixedZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_MIXED_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_DOUBLE), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrMixed2FloatZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_FLOAT_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_MIXED), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrMixed2DoubleZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_DOUBLE_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_MIXED), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrMixed2MixedZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_Mixed_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_MIXED), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrFloatZ2DoubleZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_DOUBLE_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_FLOAT_COMPLEX), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrFloatZ2MixedZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_MIXED_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_FLOAT_COMPLEX), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrDoubleZ2FloatZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_FLOAT_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_DOUBLE_COMPLEX), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrDoubleZ2MixedZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_MIXED_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_DOUBLE_COMPLEX), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrMixedZ2FloatZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_FLOAT_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_MIXED_COMPLEX), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

SUBROUTINE copyCsrMixedZ2DoubleZ(outputCsr, inputCsr)

IMPLICIT NONE

TYPE(CSR_DOUBLE_COMPLEX), INTENT(OUT) :: outputCsr
TYPE(CSR_MIXED_COMPLEX), INTENT(IN) :: inputCsr

CALL createCsr(outputCsr, inputCsr%nnz, inputCsr%nr, inputCsr%nc)

outputCsr%csrVal = inputCsr%csrVal
outputCsr%csrRowPtr = inputCsr%csrRowPtr
outputCsr%csrColIdx = inputCsr%csrColIdx
outputCsr%nr = inputCsr%nr
outputCsr%nc = inputCsr%nc
outputCsr%nnz = inputCsr%nnz
outputCsr%lFinalized = inputCsr%lFinalized

#ifdef __INTEL_MKL
outputCsr%pardisoPermute = inputCsr%pardisoPermute
#endif

END SUBROUTINE

FUNCTION subCsrCsrFloat(leftCsr, rightCsr) RESULT(resultCsr)

IMPLICIT NONE

TYPE(CSR_FLOAT), INTENT(IN) :: leftCsr, rightCsr
TYPE(CSR_FLOAT) :: resultCsr
REAL(4) :: diff
INTEGER :: i, left_ptr, right_ptr

CALL createCsr(resultCsr, leftCsr%nnz + rightCsr%nnz, leftCsr%nr, leftCsr%nc)

left_ptr = 1; right_ptr = 1
DO i = 1, leftCsr%nr

  DO WHILE (left_ptr .LT. leftCsr%csrRowPtr(i + 1) .AND. right_ptr .LT. rightCsr%csrRowPtr(i + 1))
    IF (leftCsr%csrColIdx(left_ptr) .LT. rightCsr%csrColIdx(right_ptr)) THEN
      CALL pushCsr(resultCsr, leftCsr%csrVal(left_ptr), i, leftCsr%csrColIdx(left_ptr))
      left_ptr = left_ptr + 1
    ELSEIF (leftCsr%csrColIdx(left_ptr) .EQ. rightCsr%csrColIdx(right_ptr)) THEN
      diff = leftCsr%csrVal(left_ptr) - rightCsr%csrVal(right_ptr)
      IF (abs(diff) .GT. 1.0D-10) CALL pushCsr(resultCsr, diff, i, leftCsr%csrColIdx(left_ptr))
      left_ptr = left_ptr + 1
      right_ptr = right_ptr + 1
    ELSE
      CALL pushCsr(resultCsr, - rightCsr%csrVal(right_ptr), i, rightCsr%csrColIdx(right_ptr))
      right_ptr = right_ptr + 1
    ENDIF
  ENDDO

  DO WHILE (left_ptr .LT. leftCsr%csrRowPtr(i + 1))
    CALL pushCsr(resultCsr, leftCsr%csrVal(left_ptr), i, leftCsr%csrColIdx(left_ptr))
    left_ptr = left_ptr + 1
  ENDDO

  DO WHILE (right_ptr .LT. rightCsr%csrRowPtr(i + 1))
    CALL pushCsr(resultCsr, - rightCsr%csrVal(right_ptr), i, rightCsr%csrColIdx(right_ptr))
    right_ptr = right_ptr + 1
  ENDDO

ENDDO

CALL finalizeCsr(resultCsr, .FALSE.)

END FUNCTION

FUNCTION subCsrCsrDouble(leftCsr, rightCsr) RESULT(resultCsr)

IMPLICIT NONE

TYPE(CSR_DOUBLE), INTENT(IN) :: leftCsr, rightCsr
TYPE(CSR_DOUBLE) :: resultCsr
REAL(8) :: diff
INTEGER :: i, left_ptr, right_ptr

CALL createCsr(resultCsr, leftCsr%nnz + rightCsr%nnz, leftCsr%nr, leftCsr%nc)

left_ptr = 1; right_ptr = 1
DO i = 1, leftCsr%nr

  DO WHILE (left_ptr .LT. leftCsr%csrRowPtr(i + 1) .AND. right_ptr .LT. rightCsr%csrRowPtr(i + 1))
    IF (leftCsr%csrColIdx(left_ptr) .LT. rightCsr%csrColIdx(right_ptr)) THEN
      CALL pushCsr(resultCsr, leftCsr%csrVal(left_ptr), i, leftCsr%csrColIdx(left_ptr))
      left_ptr = left_ptr + 1
    ELSEIF (leftCsr%csrColIdx(left_ptr) .EQ. rightCsr%csrColIdx(right_ptr)) THEN
      diff = leftCsr%csrVal(left_ptr) - rightCsr%csrVal(right_ptr)
      IF (abs(diff) .GT. 1.0D-10) CALL pushCsr(resultCsr, diff, i, leftCsr%csrColIdx(left_ptr))
      left_ptr = left_ptr + 1
      right_ptr = right_ptr + 1
    ELSE
      CALL pushCsr(resultCsr, - rightCsr%csrVal(right_ptr), i, rightCsr%csrColIdx(right_ptr))
      right_ptr = right_ptr + 1
    ENDIF
  ENDDO

  DO WHILE (left_ptr .LT. leftCsr%csrRowPtr(i + 1))
    CALL pushCsr(resultCsr, leftCsr%csrVal(left_ptr), i, leftCsr%csrColIdx(left_ptr))
    left_ptr = left_ptr + 1
  ENDDO

  DO WHILE (right_ptr .LT. rightCsr%csrRowPtr(i + 1))
    CALL pushCsr(resultCsr, - rightCsr%csrVal(right_ptr), i, rightCsr%csrColIdx(right_ptr))
    right_ptr = right_ptr + 1
  ENDDO

ENDDO

CALL finalizeCsr(resultCsr, .FALSE.)

END FUNCTION

END MODULE
