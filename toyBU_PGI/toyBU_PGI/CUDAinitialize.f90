#ifdef __PGI
  MODULE CUDAAUXIL
  USE CUDAFOR
  USE CUBLAS !, ONLY : cubalsCreate, cublaseSetStream
  USE CUSPARSE !, ONLY : cusparseCreate, cusparseSetStream
  USE OPENACC
  USE OMP_LIB
  IMPLICIT NONE 
  TYPE cuCntl_Type
    INTEGER :: nDevice
    TYPE(cuDevice_Type), POINTER :: cuDevice(:)
    
  END TYPE
  TYPE cuDevice_Type
    TYPE(cublasHandle) :: myblasHandle
    TYPE(cusparseHandle) :: mySparseHandle
    INTEGER(KIND = cuda_stream_kind) :: myStream, myStreamComm
    !--- Device Property Variables
    INTEGER :: cuSMXCount
    INTEGER :: cuArchitecture
    INTEGER :: cuWarpSize
    INTEGER :: cuMaxThreadPerSMX
    INTEGER :: cuMaxThreadPerBlock
    INTEGER :: cuMaxBlockPerSMX
    INTEGER :: cuMaxWarpPerSMX
    INTEGER :: cuWarpPerBlock
    INTEGER :: cuThreadPerBlock
    INTEGER :: sharedMemoryDim
  END TYPE
  
  CONTAINS
  SUBROUTINE CUDAINIT(cuCntl)
  TYPE(cuCntl_Type), INTENT(INOUT) :: cuCntl
  
  TYPE(cudaDeviceProp) :: cudaProperty
  INTEGER(KIND=cuda_stream_kind), POINTER :: cuStream(:), cuStreamComm(:)
  
  INTEGER :: i, j, ierr, tid
  INTEGER :: nDevice
  INTEGER(8) :: totalByte(0:10), allocByte(0:10), freeByte(0:10, 2)
  CHARACTER(10) :: totalMBChar, allocMBChar, freeMBChar
  REAL :: totalMB, allocMB, freeMB
  
  nDevice = acc_get_num_devices(acc_device_nvidia)
  
  cuCntl%nDevice = nDevice
  ALLOCATE(cuCntl%cuDevice(0:cuCntl%nDevice-1))
  ALLOCATE(cuStream(0:cuCntl%nDevice-1))
  ALLOCATE(cuStreamComm(0:cuCntl%nDevice-1))
  
  CALL omp_set_num_threads(cuCntl%nDevice)
  
  !$OMP PARALLEL PRIVATE(tid, ierr, cudaProperty)
  tid = omp_get_thread_num()
  CALL acc_set_device(tid, acc_device_nvidia)
  
  ierr = cudaMemGetInfo(freeByte(tid,1), totalByte(tid))
  
  ierr = cudaStreamCreate(cuStream(tid))
  ierr = cudaStreamCreate(cuStreamComm(tid))
  cuCntl%cuDevice(tid)%myStream = cuStream(tid)
  cuCntl%cuDevice(tid)%myStreamComm = cuStreamComm(tid)
  ierr = cublasCreate(cucntl%cuDevice(tid)%myblasHandle)
  ierr = cusparseCreate(cucntl%cuDevice(tid)%mySparseHandle)
  ierr = cublasSetStream(cuCntl%cuDevice(tid)%myblasHandle, cuStream(tid))
  ierr = cusparseSetStream(cuCntl%cuDevice(tid)%mySparseHandle, cuStream(tid))
  
  ierr = cudaMemGetInfo(freeByte(tid,2), totalByte(tid))
  allocByte(tid) = freeByte(tid,1) - freeByte(tid,2)
  
  ierr = cudaGetDeviceProperties(cudaproperty, tid)
  cuCntl%cuDevice(tid)%cuSMXCount = cudaproperty%multiProcessorCount
  cuCntl%cuDevice(tid)%cuArchitecture = cudaproperty%major
  cuCntl%cuDevice(tid)%cuWarpSize = cudaproperty%warpSize
  cuCntl%cuDevice(tid)%cuMaxThreadPerSMX = cudaproperty%maxThreadsPerMultiprocessor
  cuCntl%cuDevice(tid)%cuMaxThreadPerBlock = cudaproperty%maxThreadsPerBlock
  cuCntl%cuDevice(tid)%cuMaxWarpPerSMX = cudaproperty%maxThreadsPerMultiprocessor / cudaproperty%warpSize
  
  SELECT CASE (cuCntl%cuDevice(tid)%cuArchitecture)
  CASE (2)   !--- Fermi
    cuCntl%cuDevice(tid)%cuMaxBlockPerSMX = 8
  CASE (3)   !--- Kepler
    cuCntl%cuDevice(tid)%cuMaxBlockPerSMX = 16
  CASE (5)   !--- Maxwell
    cuCntl%cuDevice(tid)%cuMaxBlockPerSMX = 32
  CASE (6)   !--- Pascal
    cuCntl%cuDevice(tid)%cuMaxBlockPerSMX = 32
  END SELECT
  
  cuCntl%cuDevice(tid)%cuWarpPerBlock = cuCntl%cuDevice(tid)%cuMaxWarpPerSMX / cuCntl%cuDevice(tid)%cuMaxBlockPerSMX
  cuCntl%cuDevice(tid)%cuThreadPerBlock = cuCntl%cuDevice(tid)%cuWarpPerBlock * cuCntl%cuDevice(tid)%cuWarpSize
  !$OMP END PARALLEL
  
  DO tid = 0, cuCntl%nDevice-1
    totalMB = DBLE(totalByte(tid))/1024.0**2
    allocMB = DBLE(allocByte(tid))/1024.0**2
    freeMB = DBLE(freeByte(tid,2))/1024.0**2
    ierr = cudaGetDeviceProperties(cudaProperty, tid)
    WRITE(totalMBChar, '(f8.2)') totalMB; totalMBChar = ADJUSTL(totalMBChar)
    WRITE(allocMBChar, '(f8.2)') allocMB; allocMBChar = ADJUSTL(allocMBChar)
    WRITE(freeMBChar, '(f8.2)') freeMB; freeMBChar = ADJUSTL(freeMBChar)
    WRITE(*, '(5x, a)'), TRIM(cudaProperty%name)//' : '//TRIM(totalMBChar)//'MB total, '//  &
                          TRIM(allocMBChar)//'MB Allocated, '//TRIM(freeMBChar)//'MB Free'
    print*, cuCntl%cuDevice(tid)%cuMaxThreadPerBlock
  END DO
   
  END SUBROUTINE
  END MODULE
#endif  