  PROGRAM Main
  USE MatExp_Mod
  USE MatExponential
  USE CSRMATRIX
  USE AuxilCSR
#ifdef __PGI
  USE CUDAFOR
  USE CUDAAUXIL
#endif  
  IMPLICIT NONE
  INTEGER,PARAMETER :: InputID = 11, OutputID = 22
  CHARACTER(64)     :: Filename = 'DeplInfo_toy_EOC.txt', FMT
  INTEGER :: NofIso, i, j, nthds, ierr
  REAL(8), ALLOCATABLE :: V0(:), V1(:), Amat(:,:), AmatSub1(:,:)
  REAL(8), ALLOCATABLE :: Vref(:), dummy(:,:)
  INTEGER :: time_beg(8), time_end(8)

  TYPE(CSR_DOUBLE) :: Amat_CSR, ExpA_CSR
  TYPE(CSR_DOUBLE_COMPLEX) :: Amat_CsrZ
  TYPE(MatExpVars_Type) :: MatExpVar
  !TYPE(MatExp_Type) :: DeplInfo(1000)
#ifdef __PGI
  !TYPE(cuCntl_Type) :: cuCntl
#endif  
  ! Initialize and Read Inputs
  print*, '-----------Toy Depletion Codes Start-----------'
  print*, Filename
  CALL ReadDeplInfo(InputID, Filename, NofIso, Amat, V0, Vref)
  WRITE(FMT, '(A1, I3, A8)'), '(',NofIso,'E20.8E3)'
  print*, 'Reference Vector :'      ! From Krylov
  !WRITE(*,FMT) Vref
  ALLOCATE(V1(NofIso), dummy(NofIso,NofIso))

  CALL Full2CSR(Amat, Amat_CSR);
  
  MatExpVar%A_Csr = Amat_CSR; CALL finalizecsr(MatExpVar%A_Csr, .true.)
  ALLOCATE(MatExpVar%dx0(NofIso), MatExpVar%dx1(NofIso))
  MatExpVar%dx0 = V0; MatExpVar%dx1 = V1; MatExpVar%rank = NofIso
  
  !print*, Amat_Csr%csrval
  !DO i = 1, 1000
  !  ALLOCATE(DeplInfo(i)%Mat, DeplInfo(i)%Viso0(NofIso), DeplInfo(i)%Viso(NofIso))
  !  DeplInfo(i)%lAllocVec = .TRUE.; DeplInfo(i)%nisodepl = NofIso
  !  CALL Fullmat2Sparse(DeplInfo(i)%Mat, Amat, NofIso)
  !  DeplInfo(i)%Viso0(:) = V0(:)
  !END DO
#ifdef __PGI
  !CALL CUDAINIT(cuCntl)
#endif  
  CALL DATE_AND_TIME(VALUES= time_beg) 
  DO i = 1, 1
  CALL MatExpKrylov_cuda(MatExpVar)
  !WRITE(*,FMT) v1(NofIso-10:NofIso), vref(NofIso-10:NofIso)
  !print*, sum(ABS((v1(:)-Vref(:))/max(1.D-50,abs(Vref(:)))))/dble(NofIso)
  !DO i = 1, NofIso
    !IF(Vref(i).NE.0.0) print*, i, ABS((v1(i)-Vref(i))/max(1.D-50,abs(Vref(i)))), Vref(i)
    !IF(Vref(i).NE.0.0) print*, i, v1(i), Vref(i)
  !END DO
  print*, i, 'Done'
  CALL DATE_AND_TIME(VALUES= time_end)
  print*, 'For',i,'Cycle :', ElapsedTime(time_beg, time_end)
  END DO
  CALL DATE_AND_TIME(VALUES= time_end)
  print*, 'From CUDA Krylov routine :', ElapsedTime(time_beg, time_end)
  CALL destroycsr(Amat_Csr)
  !nthds = OMP_GET_MAX_THREADS()
  !CALL OMP_SET_NUM_THREADS(4)
  ! Calculation Part --- Full Matrix
! --------------------------------------------------------------------------------!
  
  !CALL DATE_AND_TIME(VALUES= time_beg)                                           !
  !!$OMP PARALLEL DO
  !DO i = 1, 1000                                                                   !
  ! ---------------------------------------------------------------------------   !
  !CALL MatExpTaylor_full(.FALSE., Amat, dummy, V0, V1, 100, NofIso)              !
  !CALL MatExpSns_full(.FALSE., Amat, dummy, V0, V1, NofIso)                      !
  !CALL MatExpKrylov_Full(Amat, V0, V1, NofIso)                                   !
  !CALL MatExpKrylov_Csr(Amat_Csr, V0, V1)
  !CALL MatExpCRAM_full(.FALSE., Amat, dummy, V0, V1, NofIso)                     !
  !CALL MatExpKrylov(DeplInfo(i))
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

  !V1(:) = DeplInfo(1)%Viso(:)
  !print*, 'The error is ', sum(abs(Vref-V1))/NofIso

  ! Calculation part ---- CSR
! -----------------------------------------------------------------------------------!
  !CALL DATE_AND_TIME(VALUES= time_beg)                                              !
  !!$OMP PARALLEL DO
  !DO i = 1, 1000                                                                     !
  ! ---------------------------------------------------------------------------------!
  !CALL MatExpTaylor_CSR(.FALSE., Amat_CSR, ExpA_CSR, V0, V1, 100)                   !
  !CALL MatExpSns_CSR(.FALSE., Amat_csr, ExpA_CSR, V0, V1)                           !
  !CALL MatExpKrylov_CSR(Amat_Csr, V0, V1,4)                                           !
  !CALL MatExpCRAM_Full(.FALSE., Amat, dummy, V0, V1, NofIso)
  !CALL MatExpCRAM_CSR(.FALSE., Amat_Csr, ExpA_Csr, V0, V1)
  ! ---------------------------------------------------------------------------------!
  !print*, 'From Taylor Expansion w CSR :'                                           !
  !print*, 'From Sns/Taylor w CSR :'                                                 !
  !print*, 'From Krylov/Sns/Taylor w CSR :'                                          !
  !print*, 'From CRAM w CSR :'
  !WRITE(*,FMT) V1                                                                   !
  ! ---------------------------------------------------------------------------------!
  !END DO                                                                            !
  !!$OMP END PARALLEL DO
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
