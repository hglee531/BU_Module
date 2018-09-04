MODULE MatExp_Mod
IMPLICIT NONE
TYPE Mat_Type
  SEQUENCE
  LOGICAL :: lSparse=.TRUE.                                      
  LOGICAL :: lHessenberg=.FALSE.                      !
  INTEGER :: n, nmaxoffdiag
  REAL, POINTER :: LMNT(:,:)                          !Element Data for Full Matrix (i,j), i
  REAL, POINTER :: DIAG(:), OffDiag(:,:)              !Diag and Off Diag Elements for Sparse Matrix
  INTEGER, POINTER :: nlmnt(:), lmntIdx(:,:)          !#of element for row, non-zero element index
END TYPE

TYPE Arnoldi_Type
  SEQUENCE
  INTEGER :: m, dum
  REAL :: beta
  REAL, POINTER :: Hm(:, :), Vm(:, :)
END TYPE

TYPE MatExp_Type
  SEQUENCE
  INTEGER :: nisodepl
  TYPE(Mat_TYPE), POINTER :: Mat
  LOGICAL :: lAllocVec = .FALSE.
  REAL, POINTER :: Viso0(:), Viso(:)
ENDTYPE
INTEGER, PARAMETER :: MaxRank = 300          !Maximum rank of Matrix
!INTEGER, PARAMETER :: MaxOrder = 100          !Maximum order of Sub Space
INTEGER, PARAMETER :: MaxOrder = 300          !Maximum order of Sub Space  !--- BYS edit 16/01/04
CONTAINS

SUBROUTINE MatExpKrylov(DelpInfo)
USE BasicOperation, ONLY : CP_CA
IMPLICIT NONE
TYPE(MatExp_Type) :: DelpInfo
TYPE(Mat_Type), POINTER :: A
TYPE(Mat_Type) :: HM
TYPE(Arnoldi_Type) :: Arnoldi

REAL :: lmnt
REAL, ALLOCATABLE :: yapprox(:)
REAL, ALLOCATABLE :: ExpHm(:,:)
REAL, POINTER :: Viso0(:),Viso(:)
#ifdef DBG
REAL, POINTER :: Vtaylor(:), EXPM(:, :)
#endif
INTEGER :: n, m
INTEGER :: i, j, k

INTEGER :: idum1, idum2

ALLOCATE(yapprox(MaxOrder), ExpHm(MaxOrder, MaxOrder))
n = DelpInfo%nisodepl; A => DelpInfo%Mat
Viso => DelpInfo%Viso; Viso0 => DelpInfo%Viso0

CALL CP_CA(Viso, 0._8, n)
CALL ArnoldiProcess(A, Viso0, n, Arnoldi, .TRUE., m)
!print*, m
HM%Lmnt => Arnoldi%Hm; Hm%n = Arnoldi%m
Hm%lSparse = .FALSE.; Hm%lHessenberg = .TRUE.

!Calculate Matrix Exponential of Hessenberg Matrix
CALL MatExpSnS(ExpHm(1:Hm%n,1:Hm%n), Hm, Hm%n)
Yapprox(1:m) = ExpHm(1,1:m) * Arnoldi%beta
DO j = 1, m
  Viso(1:n) = Viso(1:n) + Yapprox(j) * Arnoldi%Vm(1:n, j)
ENDDO
DEALLOCATE(Arnoldi%Hm)
DEALLOCATE(Arnoldi%Vm)
DEALLOCATE(yapprox, ExpHm)
CONTINUE

END SUBROUTINE

! Below one is sub-matrix cram toy... edit by LHG 061018

SUBROUTINE FullMat2Sparse(A, A_full, n)
IMPLICIT NONE
TYPE(Mat_type), POINTER :: A
INTEGER :: n
REAL :: A_full(:,:)

REAL, POINTER :: Diag(:), OffDiag(:,:)
INTEGER, POINTER :: nlmnt(:), lmntIdx(:,:)
INTEGER, ALLOCATABLE :: lmntIdx_temp(:,:)

INTEGER :: i, j, k, maxlmnt

ALLOCATE(A%Diag(n), A%nlmnt(n))
ALLOCATE(lmntIdx_temp(n,n))
Diag => A%DIAG; nlmnt => A%nlmnt;
Diag(:) = 0._8; nlmnt(:) = 0

DO i = 1, n
  maxlmnt = 0
  DO j = 1, n
    IF (j .EQ. i) CYCLE
    IF (A_full(i,j) .NE. 0._8) THEN
      maxlmnt=maxlmnt+1; lmntIdx_temp(maxlmnt,i) = j
    END IF
  END DO
  Diag(i) = A_full(i,i); nlmnt(i) = maxlmnt
END DO
maxlmnt = maxval(nlmnt)
A%nmaxoffdiag = maxlmnt; A%n = n;

ALLOCATE(A%OffDiag(maxlmnt,n), A%lmntIdx(maxlmnt,n))
OffDiag => A%OffDiag;  lmntIdx => A%lmntIdx
lmntIdx(1:maxlmnt,:) = lmntIdx_temp(1:maxlmnt,:)
DEALLOCATE(lmntIdx_temp)

DO i = 1, n
  maxlmnt = nlmnt(i)
  DO j = 1, maxlmnt
    OffDiag(j,i) = A_full(i,lmntIdx(j,i))
  END DO
END DO

NULLIFY(Diag); NULLIFY(Offdiag); NULLIFY(nlmnt); NULLIFY(lmntIdx)

END SUBROUTINE

SUBROUTINE Sparse2FullMat(A, A_full, n)
! Get not-transposed matrix , edit by LHG on 061018
IMPLICIT NONE
TYPE(Mat_type), POINTER :: A
INTEGER :: n
REAL :: A_full(n,n)

REAL, POINTER :: Diag(:), OffDiag(:, :), Mat(:, :)
INTEGER, POINTER :: nlmnt(:), lmntIdx(:, :)

INTEGER :: jmax, jmin
INTEGER :: i, j, k

A_full = 0.0_8;

IF(A%lsparse) THEN
  Diag => A%Diag; OffDiag => A%OffDiag
  nlmnt => A%nlmnt; lmntIdx => A%lmntIdx
  DO i = 1, n
    A_full(i,i) = Diag(i)
    jmax = nlmnt(i)
    DO j = 1, jmax
      k = lmntidx(j,i)
      A_full(i,k) = OffDiag(j,i)
    ENDDO
  ENDDO
  NULLIFY(DIAG); NULLIFY(OffDiag)
  NULLIFY(nlmnt); NULLIFY(lmntIdx)
  RETURN
ELSE IF(A%lHessenberg) THEN
  Mat => A%lmnt
  DO i= 1, n
    jmin = max(1, i-1)
    A_full(i,jmin:n) = Mat(jmin:n,i)
  ENDDO
  NULLIFY(Mat)
  RETURN
ELSE
    print*, 'Something Wrong'
    CALL exit()
ENDIF

END SUBROUTINE Sparse2FullMat

SUBROUTINE MatExpSnS(ExpM, A, n, mtaylor)
!     ================================================================================
!     SUBROUTINE  : MatExpSnS                                                        +
!     FUNCTION    : Matrix Exponential Calculation Based on Scailing & Squaring      +
!     AUTHOR      : Jung Yeon Sang(SNU)                                              +
!     DATE        : Apr, 07                                                          +
!     REMARKS     : Output : ExpM(:,:)                                               +
!     ================================================================================
USE BasicOperation, ONLY : CP_CA, CP_VA, MULTI_CA
IMPLICIT NONE
REAL :: ExpM(n, n)
TYPE(Mat_Type) :: A
TYPE(Mat_Type) :: AHat
INTEGER :: n
INTEGER, OPTIONAL :: mtaylor

INTEGER :: NSQ0, NSQ, nmult, nMult0
INTEGER :: i, j, k
INTEGER :: nlmnt
INTEGER :: mtaylor_Dynamic

REAL :: mult, maxlmnt
REAL, POINTER ::Msq(:, :), pMsq(:, :), M(:, :, :)
!Maximum Squaring Factor
i = 0;
j = 1;
DO
  i = i + 1;  j = j * 2;
  IF(J .GE. n) EXIT
ENDDO
nMult0 = i 

MaxLmnt = MaxMatLmnt(A)

NSQ = 2; 
DO I = 1, 25
  MaxLmnt = MaxLmnt/2.;   
  IF(MaxLmnt .LT. 1.) EXIT
  NSQ = NSQ * 2.
ENDDO

mult = 1._8 / NSQ

IF(I .GE. nMult0) THEN
  NSQ = I - nMult0
  NMULT = 2**nMult0
ELSE
  NSQ = 0
  NMULT = 2*I
ENDIF



CALL Copy_Mat(AHat, A)
IF(AHAT%lSparse) THEN
  CALL MULTI_CA(mult, AHAT%Diag(1:n), n)
  DO i = 1, n
    j = A%nlmnt(i)
    IF(j .EQ. 0) CYCLE
    CALL MULTI_CA(mult, AHAT%OffDiag(1:j,i), j)
  ENDDO
ELSE
  CALL MULTI_CA(mult, AHAT%lmnt(1:n, 1:n), n, n)
ENDIF

!Taylor Expansion

mtaylor_Dynamic = 4
DO While(.TRUE.)
CALL MatExpTaylor(ExpM, AHat, n, mtaylor_Dynamic)
!CALL Free_Mat(AHat)

!Squaring the Matrix
ALLOCATE(M(n,n, 0:1))
CALL CP_VA(M(:, :, 0), ExpM, n, n)
DO i = 1, NSQ
  IF(MOD(i, 2) .EQ. 1) THEN
    Msq => M(:, :, 1); pMsq => M(:, :, 0)
  ELSE
    Msq => M(:, :, 0); pMsq => M(:, :, 1)
  ENDIF
  CALL FullMatMatProduct(pMsq, pMsq, Msq, n, .FALSE.)
  CONTINUE
ENDDO
IF(NSQ .NE. 0) CALL CP_VA(ExpM, Msq, n, n)
IF(ANY(ISNAN(Msq(:,:)))) THEN
  DEALLOCATE(M)
  mtaylor_Dynamic = mtaylor_Dynamic*2
ELSE
  !print*, mtaylor_Dynamic, n
  EXIT
END IF
IF(mtaylor_Dynamic .GT. n) THEN
  !WRITE(*,*) 'Non Convergable Exponential Matrix Problem with SnS'
  STOP
END IF
END DO

CALL CP_VA(M(:, :, 1), ExpM, n, n)

DO i = 1, NMULT - 1
   CALL FullMatMatProduct(M(:, :, 1), ExpM, M(:, :, 0), n, .FALSE.)
   CALL CP_VA(ExpM, M(:, :, 0), n, n)
ENDDO

CALL Free_Mat(AHat);
NULLIFY(Msq); NULLIFY(pMsq)
DEALLOCATE(M)
END SUBROUTINE

SUBROUTINE MatExpTaylor(ExpM, A, n, MaxOrder)
!     ================================================================================
!     SUBROUTINE  : MatExpTaylor                                                     +
!     FUNCTION    : Matrix Exponential Calculation Based on Taylor Expansion         +
!     AUTHOR      : Jung Yeon Sang(SNU)                                              +
!     DATE        : Apr, 07                                                          +
!     REMARKS     : Output : ExpM(:,:)                                               +
!     ================================================================================
USE BasicOperation, ONLY : CP_CA, CP_VA
IMPLICIT NONE
REAL :: ExpM(n, n)
TYPE(Mat_Type) :: A
INTEGER :: n
INTEGER, OPTIONAL :: MaxOrder
LOGICAL :: lAutoOrder

LOGICAL :: lSparse, lHessenberg

REAL, POINTER :: An(:, :), pAn(:, :)
REAL, POINTER :: FullMat(:, :)
REAL, POINTER :: Diag(:), OffDiag(:, :)
INTEGER, POINTER :: nlmnt(:), lmntIdx(:, :)
INTEGER :: i, j, k, l
INTEGER :: jmax
INTEGER :: mx
REAL :: coeff  !Inverse of Factorial
REAL :: L2Norm, L2Norm0, RelL2Norm
REAL :: L1Norm, L1Norm0, RelL1Norm

lAutoOrder = .TRUE.
!lAutoOrder = .FALSE. !BYS test 15/12/31

mx = n
IF(PRESENT(MaxOrder)) THEN
  mx = MaxOrder;   lAutoOrder = .FALSE.
ENDIF
lSparse = A%lSparse; lHessenberg = A%lHessenberg
ALLOCATE(An(n,n))
ALLOCATE(pAn(n,n))

IF(lSparse) THEN
  Diag => A%Diag; OffDiag => A%OffDiag
  nlmnt => A%nlmnt; lmntIdx => A%lmntIdx
ELSE
  FullMat => A%lmnt
ENDIF

CALL CP_CA(ExpM, 0._8, n, n)
!ExpM = I + A  Upto 1st order
IF(lSparse) THEN
  DO i = 1, n
    ExpM(i, i) = Diag(i)
    jmax = nlmnt(i)
    DO j = 1, jmax
      k = lmntIdx(j, i)
      ExpM(k, i) = OffDiag(j, i)
    ENDDO
  ENDDO
ELSE
  CALL CP_VA(ExpM, FullMat, n, n)
ENDIF

CALL CP_VA(pAn, ExpM, n, n) 
L2Norm0 = MatL2Norm(ExpM, n); L1Norm0 = MatL1Norm(ExpM, n)
!ExpM = I + A   !First Order Taylor Expansion
DO i = 1, n
  ExpM(i, i) = ExpM(i, i) + 1._8
ENDDO

Coeff = 1._8
DO i=2, mx
  Coeff = Coeff/dble(i)
  IF(lSparse) THEN
    CALL GMatMatProduct(A, pAn, An, n)  !A : Sparse, pAn : Full Mat, An : Full Matrix
  ELSE
    CALL FullMatMatProduct(pAn, FullMat, An, n, lHessenberg)
  ENDIF
  CALL CP_VA(pAn, An, n, n)

  DO j = 1, n
    DO k = 1, n
      An(k, j) = An(k, j) * Coeff
      ExpM(k, j) = ExpM(k, j) + An(k, j)
    ENDDO
  ENDDO
  
  IF(lAutoOrder) THEN
    L2Norm = MatL2Norm(An, n)
    L1Norm = MatL1Norm(ExpM, n)
    RelL2Norm = L2Norm/L2Norm0
    RelL1Norm = L1Norm/L1Norm0
    IF(RelL2Norm .lt. 1.0E-6) EXIT
  ENDIF
ENDDO

IF(lSparse) THEN
  NULLIFY(Diag); NULLIFY(OffDiag)
  NULLIFY(nlmnt); NULLIFY(lmntIdx)
ELSE
  NULLIFY(FullMat)
ENDIF
Deallocate(An, pAn)

END SUBROUTINE

SUBROUTINE ArnoldiProcess(A, v, n, Arnoldi, lAuto, m)
!     ================================================================================
!     SUBROUTINE  : ArnoldiProcess                                                   +
!     FUNCTION    : ArnoldiProcess for Orthogonal Basis Generation                   +
!     AUTHOR      : Jung Yeon Sang(SNU)                                              +
!     DATE        : Apr, 07                                                          +
!     REMARKS     : Output : ExpM(:,:)                                               +
!     ================================================================================
USE BasicOperation,   ONLY : DotProduct, MULTI_CA,   CP_VA, &
                             CP_CA,      CP_CAVB,    SUB_VA
IMPLICIT NONE

REAL :: V(n)
REAL, ALLOCATABLE :: Hm(:,:), Vm(:,:)
!REAL :: Hm(MaxOrder+1, MaxOrder+1), Vm(n, MaxOrder)
TYPE(Mat_Type) :: A
TYPE(Arnoldi_Type) :: Arnoldi
INTEGER :: n, m
LOGICAL :: lAuto
LOGICAL :: lSparse
INTEGER :: i, j, k, l, icount

REAL :: beta, h
REAL :: p(n), p0(n)
REAL :: hdata(0:9), hdatasq(0:9), hstd, havg, hmout

ALLOCATE(Hm(MaxOrder+1, MaxOrder+1), Vm(n, MaxOrder))

lSparse = A%lSparse
IF(lAuto) m = MaxOrder
CALL CP_CA(Hm, 0._8, MaxOrder+1, MaxOrder+1)

beta = DotProduct(V, V, n); beta = SQRT(beta)
CALL CP_CAVB(1._8/Beta, V(1:n), Vm(1:n, 1), n)
!CALL CP_VA(Vm(:,1), V(:), n)
!CALL Multi_CA(1._8/beta, Vm(1:n, 1), n)
l = 0; k = 0; icount = 0
DO j = 1, m
  l = l + 1
  CALL GMatVecProduct(A, Vm(1:n, j), P, n)
  !Update Hessenberg Matrix
  DO i = 1, j
    h = DotProduct(Vm(:,i), P, n); Hm(j, i) = h 
    CALL CP_CAVB(h, Vm(1:n, i), p0, n)
    CALL SUB_VA(P, P, P0, n)               !p = p - h*Vm(:,i)
  ENDDO
  h = DotProduct(P, P, n); h = SQRT(h)
  CALL CP_CAVB(1._8/h, p, Vm(1:n, j+1), n)
  Hm(j, j+1) = h
  IF(J .EQ. 25) HMOUT = ABS(h)
  IF(lAuto .AND. J .GT. 25) THEN     
     Hmout = (abs(h) + hmout) * 0.5_8
     IF(HMOUT .LT. 4._8) icount = icount + 1
     IF(icount .GE. 5) EXIT
!    i = mod(j, 10)
!    hdata(i) = h; hdatasq(i) = h * h
!    IF(J .GT. 10) THEN
!      havg = sum(hdata(0:9))*0.1_8
!      hstd = sum(hdatasq(0:9))*0.1_8 - havg * havg
!      hstd = SQRT(hstd)
!    ENDIF
!    IF(J .GT. 10 .AND. hstd .lt. 5.E-4) EXIT
  ENDIF
ENDDO
IF(lAuto) m = l

ALLOCATE(Arnoldi%Hm(m, m))
ALLOCATE(Arnoldi%Vm(n, m))
CALL CP_VA(Arnoldi%Hm(1:m,1:m), Hm(1:m, 1:m), m, m)
CALL CP_VA(Arnoldi%Vm(1:n,1:m), Vm(1:n, 1:m), n, m)
Arnoldi%beta = beta; Arnoldi%m = m
DEALLOCATE(Hm, Vm)
END SUBROUTINE

FUNCTION MaxMatLmnt(A)
IMPLICIT NONE
TYPE(Mat_Type) :: A
REAL :: MaxMatLmnt

REAL, POINTER :: Diag(:), OffDiag(:, :), MAT(:, :)
INTEGER, POINTER :: nLmnt(:), LmntIdx(:, :)

INTEGER :: n
INTEGER :: i, j, jmax, jmin

n = A%n;MaxMatLmnt = 0

IF(A%lSparse) THEN
  Diag => A%Diag; OffDiag => A%OffDiag
  nlmnt => A%nlmnt; LmntIdx => A%LmntIdx
  DO i = 1, n
    MaxMatLmnt = MAX(MaxMatLmnt, Diag(i))
    jmax= nlmnt(i)
    !MaxMatLmnt = MAX(MaxMatLmnt, MAXVAL(OffDiag(1:jmax,i)))
    DO j = 1, jmax
      MaxMatLmnt = MAX(MaxMatLmnt, ABS(OffDiag(J,i)))
    ENDDO
  ENDDO
  NULLIFY(Diag); NULLIFY(OffDiag)
  NULLIFY(nlmnt); NULLIFY(LmntIdx)
  RETURN
ENDIF

MAT => A%lmnt
IF(A%lHessenberg) THEN
  DO i = 1, n
    jmin = MAX(1, i-1)
    DO j = jmin, n
      MaxMatLmnt = MAX(MaxMatLmnt, ABS(MAT(j,i)))
    ENDDO
  ENDDO
  NULLIFY(MAT)
  RETURN
ENDIF

DO i = 1, n
  MaxMatLmnt = MAX(MaxMatLmnt, MAXVAL(MAT(1:n,i)))
  DO j = 1, n
    MaxMatLmnt = MAX(MaxMatLmnt, ABS(MAT(j,i)))
  ENDDO  
ENDDO
NULLIFY(MAT)
RETURN
END FUNCTION

SUBROUTINE Copy_Mat(A,B)
!     ================================================================================
!     SUBROUTINE  : Copy_Mat                                                         +
!     FUNCTION    : Gopy Generalized Matrix                                          +
!     AUTHOR      : Jung Yeon Sang(SNU)                                              +
!     DATE        : Apr, 07                                                          +
!     REMARKS     :                                                                  +
!     ================================================================================
USE BasicOperation, ONLY : CP_VA, CP_CA
IMPLICIT NONE
TYPE(Mat_Type) :: A, B
INTEGER :: nlmnt,n
LOGICAL :: lSparse, lHessenberg

n = B%n
nlmnt = 0
A%n = n;
lSparse = B%lSparse
IF(lSparse) THEN
  nlmnt = B%nMaxOffDiag;  A%nMaxOffDiag = nlmnt
  ALLOCATE(A%Diag(n)); ALLOCATE(A%nlmnt(n))
  ALLOCATE(A%OffDiag(nlmnt,n)); ALLOCATE(A%lmntIdx(nlmnt, n))
  CALL CP_VA(A%Diag, B%Diag, n); CALL CP_VA(A%nlmnt, B%nlmnt, n)
  CALL CP_VA(A%OffDiag, B%OffDiag, nlmnt, n); CALL CP_VA(A%lmntIdx, B%lmntIdx, nlmnt, n)
  A%lSparse = .TRUE.
  RETURN
ENDIF

A%lSparse =.FALSE.; A%lHessenberg = B%lHessenberg
ALLOCATE(A%lmnt(n, n))
CALL CP_VA(A%lmnt, B%lmnt, n, n)
END SUBROUTINE

SUBROUTINE Free_Mat(A)
!     ================================================================================
!     SUBROUTINE  : Copy_Mat                                                         +
!     FUNCTION    : Free Generalized Matrix                                          +
!     AUTHOR      : Jung Yeon Sang(SNU)                                              +
!     DATE        : Apr, 07                                                          +
!     REMARKS     :                                                                  +
!     ================================================================================
IMPLICIT NONE
TYPE(Mat_Type) :: A

IF(A%lSparse) THEN
  DEALLOCATE(A%Diag, A%nlmnt)
  DEALLOCATE(A%OffDiag, A%lmntIdx)  
  RETURN
ENDIF

DEALLOCATE(A%Lmnt)
END SUBROUTINE

SUBROUTINE GMatVecProduct(A,x,y,n)
!     ================================================================================
!     SUBROUTINE  : GMatVecProduct                                                   +    
!     FUNCTION    : Generalized Matrix Vector Product                                +
!     AUTHOR      : Jung Yeon Sang(SNU)                                              +
!     DATE        : Apr, 07                                                          +
!     REMARKS     : Matrix Vector Product for Deplition                              +
!                 :                                                                  +
!     ================================================================================
USE BasicOperation, ONLY : CP_CA
IMPLICIT NONE
TYPE(Mat_Type) A
REAL :: x(n), y(n)


REAL :: lmnt
REAL, POINTER :: Diag(:), OffDiag(:, :), Mat(:, :)
INTEGER, POINTER :: nlmnt(:), lmntIdx(:, :)

INTEGER :: n, jmin, jmax
INTEGER :: i, j, k


CALL CP_CA(y, 0._8, n)

IF(A%lsparse) THEN
  Diag => A%Diag; OffDiag => A%OffDiag
  nlmnt => A%nlmnt; lmntIdx => A%lmntIdx
  DO i = 1, n
    lmnt = Diag(i) * x(i)
    jmax = nlmnt(i)
    DO j = 1, jmax
      k = lmntidx(j,i)
      lmnt = lmnt + OffDiag(j,i) * x(k)
    ENDDO
    y(i) = lmnt
  ENDDO
  NULLIFY(DIAG); NULLIFY(OffDiag)
  NULLIFY(nlmnt); NULLIFY(lmntIdx)
  RETURN
ENDIF

Mat => A%lmnt
IF(A%lHessenberg) THEN
  DO i= 1, n
    lmnt = 0
    jmin = max(1, i-1)
    DO j = jmin, n
      lmnt = lmnt + Mat(j, i) * x(j)
    ENDDO
    y(i) = lmnt
  ENDDO
  NULLIFY(Mat)
  RETURN
ENDIF

!Full Matrix Operation
DO i= 1, n
  lmnt = 0
  DO j = 1, n
    lmnt = lmnt + Mat(j, i) * x(j)
  ENDDO
  y(i) = lmnt
ENDDO
NULLIFY(Mat)
END SUBROUTINE

SUBROUTINE FullMatrixVector(A,x, y, n, lHessenberg)
IMPLICIT NONE
REAL :: A(n, n)
REAL :: x(n), y(n)
INTEGER :: n
LOGICAL :: lHessenberg

INTEGER :: i, j, jmin
REAL :: lmnt

IF(lHessenberg) THEN
  DO i= 1, n
    lmnt = 0
    jmin = max(1, i-1)
    DO j = jmin, n
      lmnt = lmnt + A(j, i) * x(j)
    ENDDO
    y(i) = lmnt
  ENDDO
  RETURN
ENDIF

!Full Matrix Operation
DO i= 1, n
  lmnt = 0
  DO j = 1, n
    lmnt = lmnt + A(j, i) * x(j)
  ENDDO
  y(i) = lmnt
ENDDO
END SUBROUTINE

SUBROUTINE GMatMatProduct(A, B, C, n)
USE BasicOperation, ONLY : CP_CA, CP_VA
TYPE(Mat_Type) :: A
REAL :: B(n,n), C(n,n)
INTEGER :: n


REAL :: lmnt
REAL, ALLOCATABLE :: V(:), W(:)
!REAL :: V(MaxRank), W(MaxRank)
INTEGER :: i

ALLOCATE(V(MaxRank), W(MaxRank))

CALL CP_CA(C, 0._8, n, n)
DO i = 1, n
  CALL CP_VA(V(1:n), B(i, 1:n), n)
  CALL GMatVecProduct(A, V(1:n), W(1:n), n)
  CALL CP_VA(C(i,1:n), W(1:n), n)
ENDDO

DEALLOCATE(V,W)
END SUBROUTINE

SUBROUTINE FullMatMatProduct(A, B, C, n, lHessenberg)
!     ================================================================================
!     SUBROUTINE  : MatSquare                                                        +    
!     FUNCTION    : Generalized Matrix Vec Product                                   +
!     AUTHOR      : Jung Yeon Sang(SNU)                                              +
!     DATE        : Apr, 07                                                          +
!     REMARKS     : Matrix Vector Product for Deplition                              +
!                 : OUTPUT - C(:,:) % C = A*B                                        +
!     ================================================================================
USE BasicOperation, ONLY : CP_CA
IMPLICIT NONE
REAL :: A(n,n), B(n,n),C(n,n)
INTEGER :: n
LOGICAL :: lHessenberg

REAL :: lmnt
INTEGER :: kmax
INTEGER :: i, j, k

CALL CP_CA(C, 0._8, n, n)
IF(lHessenberg) THEN
  DO i = 1, n
    DO j = 1, n
      lmnt = 0
      kmax = min(n, j + 1)
      DO k=1,kmax
        !lmnt = lmnt + A(i,k) * B(k, j)
        lmnt = lmnt + A(k, i) * B(j, k)
      ENDDO
      C(j, i) = lmnt
    ENDDO
  ENDDO
  RETURN
ENDIF

DO i = 1, n
  DO j = 1, n
    lmnt = 0
    DO k=1, n
      lmnt = lmnt + A(k, i) * B(j, k)
    ENDDO
    C(j, i) = lmnt
  ENDDO
ENDDO

END SUBROUTINE

FUNCTION VecL2Norm(X,n)
USE BasicOperation,   ONLY : DotProduct
IMPLICIT NONE
REAL :: X(n)
INTEGER :: n
REAL :: VecL2Norm
VecL2Norm = DotProduct(X, X, n)
VecL2Norm = SQRT(VecL2Norm)
END FUNCTION

FUNCTION MatL2Norm(V, n)
USE BasicOperation,   ONLY : DotProduct
IMPLICIT NONE
REAL :: V(n, n)
INTEGER :: n

REAL :: MatL2Norm
INTEGER :: i 
MatL2Norm = 0
DO i =1 ,n
  MatL2Norm = MatL2Norm + DotProduct(V(:, i), V(:, i), n)
ENDDO
MatL2Norm = SQRT(MatL2Norm)
END FUNCTION

FUNCTION MatL1Norm(V, n)
USE BasicOperation,   ONLY : DotProduct
IMPLICIT NONE
REAL :: V(n, n)
INTEGER :: n

REAL :: MatL1Norm, VecSum
INTEGER :: i, j 
MatL1Norm = 0
DO j =1 ,n
  VecSum = 0
  DO i =1, n
    VecSum = VecSum + ABS(V(j, i))
  ENDDO
  MatL1Norm = MAX(MatL1NORM, VecSum)
ENDDO
END FUNCTION

SUBROUTINE MakeDeplMat(DMat, Amat)
IMPLICIT NONE
TYPE(Mat_Type) :: DMat
REAL :: Amat(:,:)

REAL, POINTER :: DIAG(:), OffDiag(:,:)
INTEGER, POINTER :: nlmnt(:), lmntIdx(:,:)
INTEGER :: nmaxoffdiag

INTEGER :: i, j, k
INTEGER :: rank

rank = SIZE(Amat, 1)



END SUBROUTINE


END MODULE