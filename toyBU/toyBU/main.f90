  PROGRAM Main
  USE MatExp_Mod
  USE MatExponential
  USE CSRMATRIX
  USE AuxilCSR
  IMPLICIT NONE
  INTEGER,PARAMETER :: InputID = 11, OutputID = 22
  CHARACTER(64)     :: Filename = 'DeplInfo_toy_EOC.txt', FMT
  INTEGER :: NofIso, i, nthds
  REAL, ALLOCATABLE :: V0(:), V1(:), Amat(:,:), AmatSub1(:,:), V1_BLAS(:,:)
  REAL, ALLOCATABLE :: Vref(:), dummy(:,:)
  INTEGER :: time_beg(8), time_end(8)

  TYPE(CSR_DOUBLE) :: Amat_CSR, ExpA_CSR
  TYPE(CSR_DOUBLE_COMPLEX) :: Amat_CsrZ
  TYPE(MatExp_Type) :: DeplInfo

  ! Initialize and Read Inputs
  print*, '-----------Toy Depletion Codes Start-----------'
  print*, Filename
  CALL ReadDeplInfo(InputID, Filename, NofIso, Amat, V0, Vref)
  WRITE(FMT, '(A1, I3, A8)'), '(',NofIso,'E20.8E3)'
  !print*, FMT
  !WRITE(*,FMT) V0
  !print*, 'Reference Vector :'      ! From Krylov
  !WRITE(*,FMT) Vref
  ALLOCATE(V1(NofIso), dummy(NofIso,NofIso))

  CALL Full2CSR(Amat, Amat_CSR)  
  !ALLOCATE(DeplInfo%Mat, DeplInfo%Viso0(NofIso), DeplInfo%Viso(NofIso))
  !DeplInfo%lAllocVec = .TRUE.; DeplInfo%nisodepl = NofIso
  !CALL Fullmat2Sparse(DeplInfo%Mat, Amat, NofIso)
  !DeplInfo%Viso0(:) = V0(:)
  
  !nthds = OMP_GET_MAX_THREADS()
  CALL OMP_SET_NUM_THREADS(4)
  ! Calculation Part --- Full Matrix
! --------------------------------------------------------------------------------!
  
  !CALL DATE_AND_TIME(VALUES= time_beg)                                           !
  !!$OMP PARALLEL DO PRIVATE(DeplInfo)
  !DO i = 1, 1000                                                                  !
  !ALLOCATE(DeplInfo%Mat, DeplInfo%Viso0(NofIso), DeplInfo%Viso(NofIso))
  !DeplInfo%lAllocVec = .TRUE.; DeplInfo%nisodepl = NofIso
  !CALL Fullmat2Sparse(DeplInfo%Mat, Amat, NofIso)
  !DeplInfo%Viso0(:) = V0(:)
  ! ---------------------------------------------------------------------------   !
  !CALL MatExpTaylor_full(.FALSE., Amat, dummy, V0, V1, 100, NofIso)              !
  !CALL MatExpSns_full(.FALSE., Amat, dummy, V0, V1, NofIso)                      !
  !CALL MatExpKrylov_Full(Amat, V0, V1, NofIso)                                   !
  !CALL MatExpKrylov_Csr(Amat_Csr, V0, V1)
  !CALL MatExpCRAM_full(.FALSE., Amat, dummy, V0, V1, NofIso)                     !
  !CALL MatExpKrylov(DeplInfo)
  ! ---------------------------------------------------------------------------   !
  !print*, 'From Taylor Expansion w/o CSR:'                                       !
  !print*, 'From Sns/Taylor w/o CSR:'                                             !
  !print*, 'From Krylov/Sns/Taylor w/o CSR :'                                     !
  !print*, 'From CRAM w/o CSR :'                                                  !
  !WRITE(*,FMT) V1                                                                !
  ! ---------------------------------------------------------------------------   !
  !END DO                                                                         !
  !!$OMP END PARALLEL DO
  !CALL DATE_AND_TIME(VALUES= time_end)                                           !
  !                                                                               !
  !print*, 'From Taylor Expansion :', ElapsedTime(time_beg, time_end)             !
  !print*, 'From Sns/Taylor w/o CSR :', ElapsedTime(time_beg, time_end)           !
  !print*, 'From Krylov/Sns/Taylor w/o CSR :', ElapsedTime(time_beg, time_end)    !
  !print*, 'From CRAM w/o CSR :', ElapsedTime(time_beg, time_end)                 !
  !print*, 'From nTRACER Krylov routine :', ElapsedTime(time_beg, time_end)
!---------------------------------------------------------------------------------!

  !V1(:) = DeplInfo%Viso(:)
  !print*, 'The error is ', sum(abs(Vref-V1))/NofIso

  ! Calculation part ---- CSR
! -----------------------------------------------------------------------------------!
  !CALL DATE_AND_TIME(VALUES= time_beg)                                              !
  !!$OMP PARALLEL DO
  !DO i = 1, 1000                                                                     !
  ! ---------------------------------------------------------------------------------!
  !CALL MatExpTaylor_CSR(.FALSE., Amat_CSR, ExpA_CSR, V0, V1, 100)                   !
  !CALL MatExpSns_CSR(.FALSE., Amat_csr, ExpA_CSR, V0, V1)                           !
  !CALL Full2CSR(Amat, Amat_CSR)  
  CALL MatExpKrylov_CSR(Amat_Csr, V0, V1, 1)                                         !
  !CALL MatExpCRAM_Full(.FALSE., Amat, dummy, V0, V1, NofIso)
  !CALL MatExpCRAM_CSR(.FALSE., Amat_Csr, ExpA_Csr, V0, V1, 4)
  ! ---------------------------------------------------------------------------------!
  !print*, 'From Taylor Expansion w CSR :'                                           !
  !print*, 'From Sns/Taylor w CSR :'                                                 !
  !print*, 'From Krylov/Sns/Taylor w CSR :'                                          !
  !print*, 'From CRAM w CSR :'
  !WRITE(*,FMT) V1                                                                   !
  ! ---------------------------------------------------------------------------------!
  !END DO                                                                            !
  !1$OMP END PARALLEL DO
  !CALL DATE_AND_TIME(VALUES= time_end)                                              !
  !                                                                                  !
  !print*, 'From Taylor Expansion w CSR, max thds :', ElapsedTime(time_beg, time_end)!
  !print*, 'From Sns/Taylor w CSR :', ElapsedTime(time_beg, time_end)                !
  !print*, 'From Krylov/Sns/Taylor w cSR :', ElapsedTime(time_beg, time_end)         !
  !print*, 'From CRAM w CSR :', ElapsedTime(time_beg, time_end)
!------------------------------------------------------------------------------------!
  
  ! Validation
  !print*, 'The error is ', sum(abs(Vref-V1))/NofIso
  ! Write Outputs

  CONTAINS

  REAL FUNCTION ElapsedTime(T_beg, T_end)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: T_beg(8), T_end(8)

  INTEGER :: T_sec, T_milli

  T_sec = (T_end(5)-T_beg(5))*3600;
  T_sec = T_sec + (T_end(6)-T_beg(6))*60
  T_sec = T_sec + (T_end(7)-T_beg(7))
  T_milli = T_end(8)-T_beg(8)
  ElapsedTime = dble(T_sec)+0.001*dble(T_milli)
  END FUNCTION

  SUBROUTINE ReadDeplInfo(IO, Filename, rank, A, x0, xref)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: IO
  CHARACTER(64), INTENT(IN) :: Filename
  INTEGER, INTENT(INOUT) :: rank
  REAL, ALLOCATABLE, INTENT(INOUT) :: A(:,:), x0(:), xref(:)

  INTEGER :: i

  OPEN(IO, FILE = Filename)
  READ(IO,*), rank
  ALLOCATE(A(rank,rank), x0(rank), xref(rank))
  READ(IO,*), x0(:)
  DO i = 1, rank
    READ(IO,*), A(i,:)
  END DO
  READ(IO,*), xref(:)

  END SUBROUTINE

  END PROGRAM
