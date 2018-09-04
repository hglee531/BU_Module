!#include <KrylovDefines.h>
#include <Depletion.h>
#ifdef __INTEL_MKL
  INCLUDE 'mkl_spblas.f90'
  INCLUDE 'mkl_service.f90'
#endif

  MODULE MatExponential
  USE CSRMATRIX
#ifdef __INTEL_MKL
  USE mkl_service
#endif
#ifdef __PGI
  USE CUDAFOR
  USE CUBLAS
  USE CUSPARSE
#endif
  IMPLICIT NONE
#ifdef CRAM_14
  INTEGER, PARAMETER::CRAM_Order = 14;
  COMPLEX(8), PARAMETER::Pole(7) = (/ (-8.897773186468888, 16.630982619902085), (-3.703275049423448, 13.656371871483268), &
    (-0.208758638250130, 10.991260561901260), (3.993369710578568, 6.004831642235037), &
    (5.089345060580624, 3.588824029027006), (5.623142572745977, 1.194069046343966), &
    (2.269783829231112, 8.461797973040221) /);
  COMPLEX(8), PARAMETER::Res(7) = (/ (-7.154288063589067e-5, 1.436104334854130e-4), (9.439025310736168e-3, -1.718479195848301e-2)  ,&
    (-3.763600387822696e-1, 3.351834702945010e-1), (-2.349823209108270e+1, -5.808359129714207)    ,&
    (4.693327448883129e+1, 4.564364976882776e+1), (-2.787516194014564e+1, -1.0214733999015645e+2) ,&
    (4.807112098832508, -1.320979383742872) /);
  REAL, PARAMETER::Res0 = 1.832174378254041e-14
#endif    
#ifdef CRAM_16
  INTEGER, PARAMETER::CRAM_Order = 16;
  COMPLEX(8), PARAMETER::Pole(8) = (/(-1.0843917078696988026e1, 1.9277446167181652284e1), (-5.2649713434426468895, 1.6220221473167927305e1)  ,&
    (5.9481522689511774808, 3.5874573620183222829), (3.5091036084149180974, 8.4361989858843750826)          ,&
    (6.4161776990994341923, 1.1941223933701386874), (1.4193758971856659786, 1.0925363484496722585e1)        ,&
    (4.9931747377179963991, 5.9968817136039422260), (-1.4139284624888862114, 1.3497725698892745389e1)/);
  COMPLEX(8), PARAMETER::Res(8) = (/(-5.0901521865224915650e-7, -2.4220017652852287970e-5), (2.1151742182466030907e-4, 4.3892969647380673918e-3) ,&
    (1.1339775178483930527e2, 1.0194721704215856450e2), (1.5059585270023467528e1, -5.7514052776421819979)        ,&
    (-6.4500878025539646595e1, -2.2459440762652096056e2), (-1.4793007113557999718, 1.7686588323782937906)        ,&
    (-6.2518392463207918892e1, -1.1190391094283228480e1), (4.1023136835410021273e-2, -1.5743466173455468191e-1)/);
  REAL, PARAMETER::Res0 = 2.1248537104952237488e-16
#endif
  TYPE MatExpVars_Type
  LOGICAL :: lOutMat
  INTEGER :: rank
  REAL(8), POINTER :: A(:,:), x0(:), x1(:), ExpA(:,:)
  TYPE(CSR_DOUBLE) :: A_Csr
#ifdef __PGI
  REAL(8), ALLOCATABLE, DEVICE :: dA(:), dx0(:), dx1(:), dExpA(:,:)
#endif
  END TYPE

  CONTAINS
#ifdef __PGI
! CSR Format for Krylov, CRAM // Full Matrix Format for SnS, Taylor
  SUBROUTINE MatExpKrylov_cuda(KrylovVars)
  IMPLICIT NONE  
  TYPE(MatExpVars_Type) :: KrylovVars, SnSVars
  INTEGER :: n, rank
  REAL(8), DEVICE :: beta
  REAL(8) :: beta_host
  REAL(8), ALLOCATABLE, DEVICE :: e1(:)
  INTEGER :: m          ! reduced order
  INTEGER :: i, ierr
  REAL :: time
  
  TYPE(DIM3) :: blocks,threads

  n = KrylovVars%rank

  CALL ArnoldiProcess_cuda(KrylovVars, SnSVars, beta_host)
  print*, 'Arnoldi Done', beta_host

  !ALLOCATE(h_small(m*m), yapprox(m), e1(m))
  !ALLOCATE(h_debug(m*m))
  !DO i = 1, m
  !  ierr = cudaMemCpy(h_small(1+(i-1)*m:i*m), h(1+(i-1)*n:m+(i-1)*n), m, cudaMemcpyDeviceToDevice)
  !END DO
  !h_debug = h_small
  !!DO i = 1, m
  !  !print*, h_debug(1+(i-1)*m:i*m), '//'
  !!END DO
  !e1 = 0._8; e1(1) = 1._8;
  !
  !CALL MatExpSns_cuda(h_small, e1, yapprox, m)
  !!print*, 'SnS Done'
  !
  !threads = dim3(256,1,1); blocks = dim3(ceiling(m/dble(256)),1,1);
  !CALL cusMulv<<<blocks,threads>>>(yapprox, beta, e1, m)
  !threads = dim3(16,16,1); blocks = dim3(n,1,1)
  !CALL cumv<<<blocks,threads,256*8>>>(v(1:n*m), e1, n, m, x1)
  !DEALLOCATE(v,h, h_small, yapprox, e1)

  CONTAINS

SUBROUTINE ArnoldiProcess_cuda(KrylovVars, SnSVars, beta)
  IMPLICIT NONE
  TYPE(MatExpVars_Type) :: KrylovVars, SnSVars
  REAL(8) :: beta
  REAL(8), ALLOCATABLE, DEVICE :: p(:), p_cpy(:), h(:), v(:)
  INTEGER :: i, j, icount, ierr
  REAL :: HmOut, h_val_host, dum_h
  INTEGER :: n, nnz, nr, nc
  INTEGER, DEVICE :: dn, dnnz, dnr, dnc
  INTEGER :: m
  
  TYPE(DIM3) :: threads16, blocksMv
  TYPE(DIM3) :: threads256, blocks1, blocksAdd
  REAL(8), DEVICE :: dum, one
  REAL, ALLOCATABLE :: v_debug(:)

  nnz = KrylovVars%A_Csr%nnz; nr = KrylovVars%A_Csr%nr; nc = KrylovVars%A_Csr%nc
  n = KrylovVars%rank
  dn = n; dnnz = nnz; dnr = nr; dnc = nc
  ALLOCATE(p(n), p_cpy(n), h(n*n), v(n*n)); ALLOCATE(v_debug(n))
  print*, 'Set'
  
  threads256 = dim3(256,1,1); threads16 = dim3(16,1,1)
  blocksMv = dim3(nr,1,1);  blocks1 = dim3(1,1,1)
  blocksAdd = dim3(ceiling(n/dble(256)),1,1)
  print*, 'start beta get'
  CALL cuNorm2<<<blocks1, threads256, 256*8>>>(KrylovVars%dx0, dn, dum)
  beta = dum; dum = 1./beta
  print*, 'beta get'
  CALL cusMulv<<<blocksAdd, threads256>>>(KrylovVars%dx0, dum, v, dn)
  print*, 'normalize v1'
  icount = 0;
  m = 0;
  h = 0.; one = 1.0;
  DO i = 1, n-1 ! n-1 for last h(i+1, i)
    print*, i,'th process'
    CALL cumv_Csr<<<blocksMv,threads16,16*8>>>(KrylovVars%A_Csr%d_csrval,KrylovVars%A_Csr%d_csrrowptr,KrylovVars%A_Csr%d_csrcolIdx, &
                                              dnnz,dnr,dnc,v((i-1)*n+1:i*n),p)
    !IF (i .EQ. 1) THEN
    !  v_debug = p
    !  print*, v_debug
    !END IF
   ! print*, 'p = Av get'
    DO j = 1, i
      !print*, (j-1)*n+1, j*n
      CALL cuDotP<<<blocks1,threads256,256*8>>>(v((j-1)*n+1:j*n), p, dn, h(j+(i-1)*n))
      !print*, 'h(i,j) get'
      !ierr = cudamemcpy(h(j+(i-1)*n), h_val, 1, cudaMemcpyDeviceToDevice)
      h_val_host = h(j+(i-1)*n); h_val_host = -h_val_host; dum = h_val_host
      !print*, i,j,h_val_host
      CALL cuAddvv<<<blocksAdd,threads256>>>(p, v(1+(j-1)*n:j*n), one, dum, p_cpy, dn)
      ierr = cudamemcpy(p, p_cpy, n, cudaMemcpyDeviceToDevice)
    END DO
    CALL cuNorm2<<<blocks1,threads256,256*8>>>(p,dn,h((i+1)+(i-1)*n))
    h_val_host = h((i+1)+(i-1)*n);
    !print*, i,'h_val :', h_val_host
    IF (i .EQ. 25) HmOut = h_val_host
    IF (i .GT. 25) THEN
      HmOut = 0.5_8*(HmOut+h_val_host)
      IF (HmOut .LT. 4._8) icount = icount+1
      IF (icount .GE. 5) THEN
        m = i; exit
      END IF
    END IF
    dum = 1./h_val_host
    CALL cusMulv<<<blocksAdd,threads256>>>(p, dum, v(1+i*n:(i+1)*n), dn)
  END DO
  IF(m .EQ. 0) m = n-1
  
  ALLOCATE(KrylovVars%dA(n*m), SnSVars%dA(m*m))
  DO i = 1, m
    j = min(i+1, m)
    ierr = cudaMemcpy(SnSVars%dA(1+(i-1)*m), h(1+(i-1)*n), j, cudaMemcpyDeviceToDevice)
  END DO
  ierr = cudaMemcpy(KrylovVars%dA, v, n*m, cudaMemcpyDeviceToDevice)

  DEALLOCATE(p, p_cpy)
  END SUBROUTINE
  END SUBROUTINE
  
!  SUBROUTINE MatExpSnS_cuda(A, x0, x1, rank)  
!  USE ieee_arithmetic
!  IMPLICIT NONE
!  REAL(8), INTENT(IN), DEVICE :: A(:)
!  REAL(8), INTENT(IN), DEVICE :: x0(:)
!  REAL(8), INTENT(INOUT), DEVICE :: x1(:)
!  INTEGER, INTENT(IN) :: rank
!
!  REAL(8), ALLOCATABLE, DEVICE :: ExpMat(:), Ahat(:)           ! Used for first Taylor Exponential and Scailing
!  INTEGER :: m, i, j
!  INTEGER :: SqrK, RecMulK, K                        ! K = SqrK + log_2(RecMulK) -> {exp(2**(-K)*A)}**(2**K)
!  INTEGER :: O_Taylor
!  REAL(8), ALLOCATABLE, DEVICE :: SaveSqrdMat(:), SqrdMat(:), savex1(:)
!  REAL(8), ALLOCATABLE :: x_temp(:)
!  REAL :: MaxInA, Div
!  REAL(8), ALLOCATABLE :: A_mxfnd(:)
!  
!  TYPE(dim3) :: threads, blocksSMulV, blocksSqr, blocksRed
!  INTEGER :: ierr
!  REAL(8), DEVICE :: Div_d
!
!  m = rank
!  ALLOCATE(Ahat(m*m), SaveSqrdMat(m*m), SqrdMat(m*m), x_temp(m), savex1(m))
!  ALLOCATE(A_mxfnd(m*m))
!  ALLOCATE(ExpMAT(m*m))
!  
!  A_mxfnd = A
!  MaxInA = maxval(ABS(A_mxFnd))
!  !DEALLOCATE(A_mxfnd)
!  !print*, MaxInA
!
!  K = INT(log(DBLE(MaxInA))/log(2.))
!  RecMulK = INT(log(DBLE(m))/log(2.))
!
!  IF (K .LT. RecMulK) THEN
!    IF (K .LE. 0) THEN
!      K = 1; SqrK = 0
!    END IF
!    RecMulK = 2**K
!  ELSE
!    SqrK = K - RecMulK
!    RecMulK = 2**RecMulK
!  END IF
!  threads = dim3(16,16,1); blocksSMulV = dim3(ceiling(m/dble(16)),ceiling(m/dble(16)),1); Div = 1._8/dble(2**K); Div_d = Div
!  CALL cusMulv<<<blocksSMulV,threads>>>(A,Div_d,Ahat,m*m)
!  
!  O_Taylor = 4; blocksSqr = dim3(m, m, 1)
!  blocksRed = dim3(m,1,1)
!  DO WHILE(.TRUE.)
!    CALL MatExpTaylor_cuda(Ahat, ExpMat, O_Taylor, m)
!    SqrdMat = ExpMat;
!    !! ----------------- Squaring the matrix exponentials, {{exp(A)}**2}**2 ...-----------    
!    DO i = 1, SqrK
!      CALL cumm<<<blocksSqr,threads,256*8>>>(SqrdMat, ExpMat, m, m, m, SaveSqrdmat)
!      IF(i.NE.SqrK) THEN
!        ierr = cudaMemCpy(ExpMat, SaveSqrdMat, m*m, cudaMemCpyDeviceToDevice)
!        ierr = cudaMemCpy(SqrdMat, SaveSqrdMat, m*m, cudaMemCpyDevicetoDevice)
!      END IF
!    END DO
!    A_mxfnd = SaveSqrdMat;
!    !DO i = 0, m-1
!    !  print*, A_mxfnd(1+i*m:(i+1)*m), '//'
!    !END DO
!    !! -----------------------------------------------------------------------------------
!    x1 = x0;      
!    !! -------------------- Recursively Multiplying on x0 Vector ------------------------- 
!    blocksRed = dim3(m,1,1)
!    DO i = 1, RecMulK
!      CALL cumv<<<blocksRed,threads,256*8>>>(SaveSqrdMat, x1, m, m, savex1)
!      ierr = cudaMemCpy(x1, savex1, m, cudaMemCpyDevicetoDevice)
!    END DO
!    !! -----------------------------------------------------------------------------------
!    x_temp = x1
!    !! ----------------------------- Convergence Check -----------------------------------
!    IF (ANY(Ieee_is_nan(x_temp))) THEN
!      O_Taylor = O_Taylor + 4
!    ELSE
!      EXIT
!    END IF
!    IF (O_Taylor .GT. m) THEN
!      print*, 'Non Convergable Exponential Matrix with SnS'
!      STOP
!    END IF
!    !! -----------------------------------------------------------------------------------
!  END DO
!DEALLOCATE(A_mxfnd)
!  DEALLOCATE(ExpMat, savex1)
!  DEALLOCATE(Ahat, SaveSqrdMat, x_temp)
!  
!  END SUBROUTINE
!  
!  SUBROUTINE MatExpTaylor_cuda(A, ExpA, order, rank)
!  IMPLICIT NONE
!  REAL(8), INTENT(IN), DEVICE :: A(:)
!  REAL(8), INTENT(INOUT), DEVICE :: ExpA(:)
!  INTEGER, INTENT(IN) :: order, rank
!  
!  INTEGER :: i, j
!  REAL(8) :: Coef
!  REAL(8), ALLOCATABLE, DEVICE :: SqrMat(:), SaveSqrMat(:), Exp_temp(:)
!  REAL(8), ALLOCATABLE :: eye(:)
!  
!  TYPE(dim3) :: blocksSqr, blocksAdd, threads
!  REAL(8), DEVICE :: one, Coef_d
!  INTEGER :: ierr
!  
!  ALLOCATE(eye(rank*rank), Exp_temp(rank*rank))
!  ALLOCATE(SqrMat(rank*rank), SaveSqrMat(rank*rank))
!  DO i = 1, rank
!    eye(1+(i-1)*rank:i*rank) = 0._8
!    eye(i+(i-1)*rank) = 1._8
!  END DO
!  threads = dim3(16,16,1); blocksSqr = dim3(rank, rank, 1); blocksAdd = dim3(ceiling(rank/dble(256)),rank,1)
!  
!  
!  ExpA = eye; SqrMat = eye; Coef = 1.; one = 1.0;
!  DO i = 1, order
!    Coef = Coef/dble(i); Coef_d = Coef
!    CALL cumm<<<blocksSqr,threads,256*8>>>(A, SqrMat, rank, rank, rank, SaveSqrMat)
!    ierr = cudaMemCpy(SqrMat, SaveSqrMat, rank*rank, cudaMemcpyDeviceToDevice)
!    CALL cuAddvv<<<blocksAdd,threads>>>(ExpA, SaveSqrMat, one, Coef_d, Exp_temp, rank*rank)
!    ierr = cudaMemCpy(ExpA, Exp_temp, rank*rank, cudaMemCpyDeviceToDevice)
!    !eye = SaveSqrMat
!    !print*, i, eye(1:rank)*Coef, '//'
!  END DO
!  DEALLOCATE(SqrMat, SaveSqrMat, Exp_temp, eye)
!  END SUBROUTINE
!    
  ATTRIBUTES(GLOBAL) SUBROUTINE cumv_CSR(val, rowptr, colIdx, nnz, nr, nc, v0, v1)
  IMPLICIT NONE
  REAL(8), DEVICE :: val(:)
  INTEGER, DEVICE :: rowptr(:), colIdx(:)
  INTEGER :: nnz, nr, nc   
  REAL(8), DEVICE :: v0(:) ! (nc)
  
  INTEGER :: ThId, BkId, NofBlock, BkSize, halfwidth
  INTEGER :: IterOut, IterIn, Ir, Ic, onset, offset
  INTEGER :: i,j
  REAL(8),INTENT(INOUT), DEVICE :: v1(nr)
  REAL(8), SHARED :: ReduceBlock(*)
  REAL(8) :: val1, val2
  REAL(8) :: redval(0:2)
  
  NofBlock = GridDim%x*GridDim%y*GridDim%z; BkSize = BlockDim%x*BlockDim%y*BlockDim%z
  ThId = ThreadIdx%z; ThId = BlockDim%y*(ThId-1)+ThreadIdx%y; ThId = BlockDim%x*(ThId-1)+ThreadIdx%x
  BkId = BlockIdx%z; BkId = GridDim%y*(BkId-1)+BlockIdx%y; BkId = GridDim%x*(BkId-1)+BlockIdx%x
  
  ! Reduce
  redval(1) = 0.0
  IterOut = CEILING(nr/real(NofBlock,4))
  DO  i =1, IterOut
    Ir = (i-1)*NofBlock+BkId
    redval(2) = 0.0
    IF (Ir .LE. nr) THEN
      onset = rowptr(Ir)
      offset = rowptr(Ir+1)
      IterIn = CEILING((offset-onset)/real(Bksize,4))
      DO j = 1, IterIn
        Ic = (j-1)*Bksize+ThId+onset-1
        IF(Ic .LT. offset) THEN
          val1 = val(Ic); val2 = v0(colIdx(Ic))
          redval(0) = val1*val2
        ELSE
          redval(0) = 0.0
        END IF
        redval(2) = redval(2)+redval(0)
      END DO
      ReduceBlock(ThId) = redval(2)
      CALL syncthreads()
      
      Ic = BkSize; halfwidth = Ic/2
      DO WHILE(halfwidth .GT. 0)
        Ic = Ic-halfwidth
        IF (ThId .LE. Ic) THEN
          redval(2) = ReduceBlock(ThId); redval(0) = ReduceBlock(ThId+halfwidth)
          ReduceBlock(ThId) = redval(2)+redval((Ic-ThId)/halfwidth)
        END IF
        halfwidth = Ic/2
        CALL syncthreads()
      END DO
      IF (ThId .EQ. 1) v1(Ir) = ReduceBlock(ThId)
    END IF
  END DO
  END SUBROUTINE
  
  ATTRIBUTES(GLOBAL) SUBROUTINE cumv(A, v0, m, n, v1)
  IMPLICIT NONE
  REAL(8), DEVICE :: A(:), v0(:)
  INTEGER :: m, n
  REAL(8), DEVICE :: v1(m)
  
  INTEGER :: ThId, BkId, NofBlock, BkSize, halfwidth
  INTEGER :: IterOut, IterIn, Ir, Ic, IrA
  INTEGER :: i, j
  REAL(8), SHARED :: ReduceBlock(*)
  REAL(8) :: val1, val2
  REAL(8) :: redval(0:2)
  
  NofBlock = GridDim%x*GridDim%y*GridDim%z; BkSize = BlockDim%x*BlockDim%y*BlockDim%z
  ThId = ThreadIdx%z; ThId = BlockDim%y*(ThId-1)+ThreadIdx%y; ThId = BlockDim%x*(ThId-1)+ThreadIdx%x
  BkId = BlockIdx%z; BkId = GridDim%y*(BkId-1)+BlockIdx%y; BkId = GridDim%x*(BkId-1)+BlockIdx%x
  
  ! Reduce
  redval(1) = 0.0
  IterOut = CEILING(m/real(NofBlock,4))
  DO  i =1, IterOut
    Ir = (i-1)*NofBlock+BkId
    redval(2) = 0.0
    IF (Ir .LE. m) THEN
      IterIn = CEILING(n/real(Bksize,4))
      DO j = 1, IterIn
        Ic = (j-1)*BkSize+Thid
        IrA = (Ic-1)*m+Ir
        IF(Ic .LE. n) THEN
          val1 = A(IrA); val2 = v0(Ic)
          redval(0) = val1*val2
        ELSE
          redval(0) = 0.0
        END IF
        redval(2) = redval(2)+redval(0)
      END DO
      ReduceBlock(ThId) = redval(2)
      CALL syncthreads()
      
      Ic = BkSize; halfwidth = Ic/2
      DO WHILE(halfwidth .GT. 0)
        Ic = Ic-halfwidth
        IF (ThId .LE. ic) THEN
          redval(2) = ReduceBlock(ThId); redval(0) = ReduceBlock(ThId+halfwidth)
          ReduceBlock(ThId) = redval(2)+redval((Ic-ThId)/halfwidth)
        END IF
        halfwidth = Ic/2
        CALL syncthreads()
      END DO
      IF (ThId .EQ. 1) v1(Ir) = ReduceBlock(ThId)
    END IF
  END DO
  END SUBROUTINE
  
  ATTRIBUTES(GLOBAL) SUBROUTINE cumm(A, B, m, n, k, C)
  IMPLICIT NONE
  REAL(8), DEVICE :: A(:), B(:)
  INTEGER :: m, n, k
  REAL(8), DEVICE :: C(:)
  
  INTEGER :: ThId, BkId, NofBlock, BkSize
  INTEGER :: IterOut, IterIn
  INTEGER :: Icr, Ic, Ir, IA, IB        ! Icr = (Ic-1)*m+Ir
  INTEGER :: i, j, Onset
  REAL(8), SHARED :: ReduceBlock(*)
  REAL(8) :: val1, val2
  REAL(8) :: redval(0:2)
  
  NofBlock = GridDim%x*GridDim%y*GridDim%z; BkSize = BlockDim%x*BlockDim%y*BlockDim%z
  ThId = ThreadIdx%z; ThId = BlockDim%y*(ThId-1)+ThreadIdx%y; ThId = BlockDim%x*(ThId-1)+ThreadIdx%x
  BkId = BlockIdx%z; BkId = GridDim%y*(BkId-1)+BlockIdx%y; BkId = GridDim%x*(BkId-1)+BlockIdx%x
  
  ! Reduce
  redval(1) = 0.0
  IterOut = CEILING(m*k/real(NofBlock,4))
  DO  i =1, IterOut
    Icr = (i-1)*NofBlock+BkId
    redval(2) = 0.0
    IF (Icr .LE. m*k) THEN
      Ir = mod(Icr-1,m)+1; Ic = (Icr-Ir)/m + 1      ! on C
      IterIn = CEILING(n/real(Bksize,4))
      DO j = 1, IterIn
        Onset = (j-1)*BkSize+ThId
        IA = (Onset-1)*m+Ir
        IB = Onset+(Ic-1)*k
        IF (Onset .LE. n) THEN
          val1 = A(IA); val2 = B(IB)
          redval(0) = val1*val2
        ELSE
          redval(0) = 0.0
        END IF
        redval(2) = redval(2)+redval(0)
      END DO
      ReduceBlock(ThId) = redval(2)
      CALL syncthreads()
      
      Ic = BkSize; Onset = Ic/2
      DO WHILE(Onset .GT. 0)
        Ic = Ic-Onset
        IF (ThId .LE. Ic) THEN
          redval(2) = ReduceBlock(ThId); redval(0) = ReduceBlock(ThId+Onset)
          ReduceBlock(ThId) = redval(2)+redval((Ic-ThId)/Onset)
        END IF
        Onset = Ic/2
        CALL syncthreads()
      END DO
      IF (ThId .EQ. 1) C(Icr) = ReduceBlock(ThId)
    END IF
  END DO
  END SUBROUTINE
  
  ATTRIBUTES(GLOBAL) SUBROUTINE cuDotP(v1, v2, n, dotP)
  IMPLICIT NONE
  REAL(8), DEVICE :: v1(:), v2(:)
  INTEGER :: n
  REAL(8), DEVICE :: dotP
  
  INTEGER :: ThId, BkId, NofBlock, BkSize, halfwidth
  INTEGER :: Ic, IterIn
  INTEGER :: i, j
  REAL(8), SHARED :: ReduceBlock(*)
  REAL(8) :: val1, val2
  REAL(8) :: redval(0:2)
  
  NofBlock = GridDim%x*GridDim%y*GridDim%z; BkSize = BlockDim%x*BlockDim%y*BlockDim%z
  ThId = ThreadIdx%z; ThId = BlockDim%y*(ThId-1)+ThreadIdx%y; ThId = BlockDim%x*(ThId-1)+ThreadIdx%x
  BkId = BlockIdx%z; BkId = GridDim%y*(BkId-1)+BlockIdx%y; BkId = GridDim%x*(BkId-1)+BlockIdx%x
  
  ! Reduce
  IF ( NofBlock .GT. 1) RETURN
  redval(1) = 0.0; redval(2) = 0.0
  IterIn = CEILING(n/real(Bksize,4))
  DO j = 1, IterIn
    Ic = (j-1)*Bksize+ThId
    IF(Ic .LE. n) THEN
      val1 = v1(Ic); val2 = v2(Ic)
      redval(0) = val1*val2
    ELSE
      redval(0) = 0.0
    END IF
    redval(2) = redval(2)+redval(0)
  END DO
  ReduceBlock(ThId) = redval(2)
  CALL syncthreads()
  
  Ic = BkSize; halfwidth = Ic/2
  DO WHILE(halfwidth .GT. 0)
    Ic = Ic-halfwidth
    IF (ThId .LE. ic) THEN
      redval(2) = ReduceBlock(ThId); redval(0) = ReduceBlock(ThId+halfwidth)
      ReduceBlock(ThId) = redval(2)+redval((Ic-ThId)/halfwidth)
    END IF
    halfwidth = Ic/2
    CALL syncthreads()
  END DO
  IF (ThId .EQ. 1) dotP = ReduceBlock(ThId)
  
  END SUBROUTINE
  
  ATTRIBUTES(GLOBAL) SUBROUTINE cuNorm2(v1, n, norm)
  IMPLICIT NONE
  REAL(8), DEVICE :: v1(:)
  INTEGER :: n
  REAL(8), DEVICE :: norm
  
  INTEGER :: ThId, BkId, NofBlock, BkSize, halfwidth
  INTEGER :: Ic, IterIn
  INTEGER :: i, j
  REAL(8), SHARED :: ReduceBlock(*)
  REAL(8) :: val1, val2
  REAL(8) :: redval(0:2)
  
  NofBlock = GridDim%x*GridDim%y*GridDim%z; BkSize = BlockDim%x*BlockDim%y*BlockDim%z
  ThId = ThreadIdx%z; ThId = BlockDim%y*(ThId-1)+ThreadIdx%y; ThId = BlockDim%x*(ThId-1)+ThreadIdx%x
  BkId = BlockIdx%z; BkId = GridDim%y*(BkId-1)+BlockIdx%y; BkId = GridDim%x*(BkId-1)+BlockIdx%x
  
  ! Reduce
  IF ( NofBlock .GT. 1) RETURN
  redval(1) = 0.0; redval(2) = 0.0
  IterIn = CEILING(n/real(Bksize,4))
  DO j = 1, IterIn
    Ic = (j-1)*Bksize+ThId
    IF(Ic .LE. n) THEN
      val1 = v1(Ic)
      redval(0) = val1*val1
    ELSE
      redval(0) = 0.0
    END IF
    redval(2) = redval(2)+redval(0)
  END DO
  ReduceBlock(ThId) = redval(2)
  CALL syncthreads()
  
  Ic = BkSize; halfwidth = Ic/2
  DO WHILE(halfwidth .GT. 0)
    Ic = Ic-halfwidth
    IF (ThId .LE. ic) THEN
      redval(2) = ReduceBlock(ThId); redval(0) = ReduceBlock(ThId+halfwidth)
      ReduceBlock(ThId) = redval(2)+redval((Ic-ThId)/halfwidth)
    END IF
    halfwidth = Ic/2
    CALL syncthreads()
  END DO
  IF (ThId .EQ. 1) norm = sqrt(ReduceBlock(ThId))
  
  END SUBROUTINE
  
  ATTRIBUTES(GLOBAL) SUBROUTINE cuAddvv(v1, v2, a, b, v3, n)
  ! ----------------------------------------------------------------
  !                       v3 = a*v1 + b*v2                         !
  ! ----------------------------------------------------------------
  IMPLICIT NONE
  REAL(8), DEVICE :: v1(:), v2(:)
  REAL(8) :: a, b
  INTEGER :: n
  REAL(8), DEVICE :: v3(:)
  
  INTEGER :: thidx, thidy, thidz ! thidz would be 1 -- recommned not to use
  INTEGER :: id
  REAL(8) :: val1, val2, val3, val4
  REAL(8) :: dval1, dval2
  
  thidx = (blockIdx%x-1)*blockDim%x+threadIdx%x; thidy = (blockIdx%y-1)*blockDim%y+threadIdx%y;
  thidz = (blockIdx%z-1)*blockDim%z+threadIdx%z
  id = thidx+(thidy-1)*(gridDim%x*blockDim%x+(thidz-1)*(gridDim%y*blockDim%y))
  IF (id .GT. n) RETURN
  
  val1 = a; val2 = v1(id); dval1 = val1*val2
  val3 = b; val4 = v2(id); dval2 = val3*val4
  v3(id) = dval1+dval2
  END SUBROUTINE
  
  ATTRIBUTES(GLOBAL) SUBROUTINE cusMulv(v0, a, v1, n)
  ! ----------------------------------------------------------------
  !                       v3 = a*v1 + b*v2                         !
  ! ----------------------------------------------------------------
  IMPLICIT NONE
  REAL(8), DEVICE :: v0(:)
  REAL(8) :: a
  INTEGER :: n
  REAL(8), DEVICE :: v1(:)
  
  INTEGER :: thidx, thidy, thidz ! thidz would be 1 -- recommned not to use
  INTEGER :: id
  REAL(8) :: val1, val2
  thidx = (blockIdx%x-1)*blockDim%x+threadIdx%x; thidy = (blockIdx%y-1)*blockDim%y+threadIdx%y;
  thidz = (blockIdx%z-1)*blockDim%z+threadIdx%z
  id = thidx+(thidy-1)*(gridDim%x*blockDim%x+(thidz-1)*(gridDim%y*blockDim%y))
  IF (id .GT. n) RETURN
  
  val1 = a; val2 = v0(id);
  v1(id) = val1*val2 
  END SUBROUTINE
#endif
  END MODUlE