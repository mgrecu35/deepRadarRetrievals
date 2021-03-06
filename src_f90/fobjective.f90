subroutine fobj_f90(att,z,z_obs, invCovZZ,invCovAtt,z_mean,&
     attka_mean,coeffsAtt,dr,n,&
     wobs,wz,watt,wzatt,f)
  implicit none
  integer :: n
  real :: att(n), z(n), z_obs(n), invCovZZ(n,n), invCovAtt(n,n), z_mean(n),&
       attka_mean(n), coeffsAtt(n,2), dr, wobs, wz, watt, wzatt
  real, intent(out) :: f
  real :: piaka, zsim
  integer k, i, j
  f=0
  piaKa=0

  do  k=n,1,-1
     piaKa=piaKa+ exp(att(k))*dr
     zsim=z(k)-piaKa
     f=f+wobs*(zsim-z_obs(k))**2
     piaKa=piaKa+exp(att(k))*dr
  enddo
  do i=1,n
     do j=1,n
        f=f+wz*(z(i)-z_mean(i))*invCovZZ(i,j)*(z(j)-z_mean(j))
     enddo
  enddo
  do i=1,n
     do j=1,n
        f=f+watt*(exp(att(i))-attka_mean(i))*invCovAtt(i,j)*&
             (exp(att(j))-attka_mean(j))
     enddo
  enddo
  do k=1,n
     f=f+wzatt*(coeffsAtt(k,1)*att(k)+coeffsAtt(k,2)-z(k))**2
  enddo
end subroutine fobj_f90

subroutine fobj_f90t(att,z,z_obs, invCovZZ,invCovAtt,z_mean,&
     attka_mean,coeffsAtt,dr,n,&
     wobs,wz,watt,wzatt,f)
  implicit none
  integer :: n
  real :: att(n), z(n), z_obs(n), invCovZZ(n,n), invCovAtt(n,n), z_mean(n),&
       attka_mean(n), coeffsAtt(n,2), dr, wobs, wz, watt, wzatt
  real, intent(out) :: f
  real :: piaka, zsim(n)
  integer k, i, j
  f=0
  piaKa=0

  do  k=n,1,-1
     piaKa=piaKa+ exp(att(k))*dr
     zsim(n)=z(k)-piaKa
     if (z_obs(k)>12) then
        f=f+wobs*(zsim(n)-z_obs(k))**2
     else
        if (zsim(n)>12) then
           f=f+wobs*(zsim(n)-12)**2
        endif
     endif
     piaKa=piaKa+exp(att(k))*dr
  enddo
  do i=1,n
     do j=1,n
        f=f+wz*(z(i)-z_mean(i))*invCovZZ(i,j)*(z(j)-z_mean(j))
     enddo
  enddo
  do i=1,n
     do j=1,n
        f=f+watt*(exp(att(i))-attka_mean(i))*invCovAtt(i,j)*&
             (exp(att(j))-attka_mean(j))
     enddo
  enddo
  do k=1,n
     f=f+wzatt*(coeffsAtt(k,1)*att(k)+coeffsAtt(k,2)-z(k))**2
  enddo
end subroutine fobj_f90t

!        Generated by TAPENADE     (INRIA, Ecuador team)
!  Tapenade 3.16 (develop) - 31 May 2021 11:17
!
!  Differentiation of fobj_f90 in reverse (adjoint) mode:
!   gradient     of useful results: f
!   with respect to varying inputs: f z att
!   RW status of diff variables: f:in-zero z:out att:out
SUBROUTINE FOBJ_F90_B(att, attb, z, zb, z_obs, invcovzz, invcovatt, &
& z_mean, attka_mean, coeffsatt, dr, n, wobs, wz, watt, wzatt, f, fb)
  IMPLICIT NONE
  INTEGER :: n
  REAL :: att(n), z(n), z_obs(n), invcovzz(n, n), invcovatt(n, n), &
& z_mean(n), attka_mean(n), coeffsatt(n, 2), dr, wobs, wz, watt, wzatt
  REAL,intent(out) :: attb(n), zb(n)
  REAL :: f
  REAL :: fb
  REAL :: piaka, zsim(n)
  REAL :: piakab, zsimb
  INTEGER :: k, i, j
  INTRINSIC EXP
  REAL :: tempb
  piaka = 0
  DO k=n,1,-1
    piaka = piaka + EXP(att(k))*dr
    !CALL PUSHREAL4(zsim)
    zsim(k) = z(k) - piaka
    piaka = piaka + EXP(att(k))*dr
  END DO
  zb = 0.0
  attb = 0.0
  DO k=n,1,-1
    tempb = 2*(coeffsatt(k, 2)+coeffsatt(k, 1)*att(k)-z(k))*wzatt*fb
    attb(k) = attb(k) + coeffsatt(k, 1)*tempb
    zb(k) = zb(k) - tempb
  END DO
  DO i=n,1,-1
    DO j=n,1,-1
      tempb = watt*invcovatt(i, j)*fb
      attb(i) = attb(i) + EXP(att(i))*(EXP(att(j))-attka_mean(j))*tempb
      attb(j) = attb(j) + EXP(att(j))*(EXP(att(i))-attka_mean(i))*tempb
    END DO
  END DO
  DO i=n,1,-1
    DO j=n,1,-1
      tempb = wz*invcovzz(i, j)*fb
      zb(i) = zb(i) + (z(j)-z_mean(j))*tempb
      zb(j) = zb(j) + (z(i)-z_mean(i))*tempb
    END DO
  END DO
  piakab = 0.0
  DO k=1,n,1
    attb(k) = attb(k) + EXP(att(k))*dr*piakab
    zsimb = 2*(zsim(k)-z_obs(k))*wobs*fb
    !CALL POPREAL4(zsim)
    zb(k) = zb(k) + zsimb
    piakab = piakab - zsimb
    attb(k) = attb(k) + EXP(att(k))*dr*piakab
  END DO
  fb = 0.0
END SUBROUTINE FOBJ_F90_B

!        Generated by TAPENADE     (INRIA, Ecuador team)
!  Tapenade 3.16 (develop) - 31 May 2021 11:17
!
!  Differentiation of fobj_f90t in reverse (adjoint) mode:
!   gradient     of useful results: f
!   with respect to varying inputs: f z att
!   RW status of diff variables: f:in-zero z:out att:out
SUBROUTINE FOBJ_F90T_B(att, attb, z, zb, z_obs, invcovzz, invcovatt, &
& z_mean, attka_mean, coeffsatt, dr, n, wobs, wz, watt, wzatt, f, fb)
  IMPLICIT NONE
  INTEGER :: n
  REAL :: att(n), z(n), z_obs(n), invcovzz(n, n), invcovatt(n, n), &
& z_mean(n), attka_mean(n), coeffsatt(n, 2), dr, wobs, wz, watt, wzatt
  REAL, intent(out) :: attb(n), zb(n)
  REAL :: f
  REAL :: fb
  REAL :: piaka, zsim(n)
  REAL :: piakab, zsimb(n)
  INTEGER :: k, i, j
  INTRINSIC EXP
  REAL :: tempb
  INTEGER :: branch
  piaka = 0
  DO k=n,1,-1
    piaka = piaka + EXP(att(k))*dr
    zsim(n) = z(k) - piaka
    piaka = piaka + EXP(att(k))*dr
  END DO
  zb = 0.0
  attb = 0.0
  DO k=n,1,-1
    tempb = 2*(coeffsatt(k, 2)+coeffsatt(k, 1)*att(k)-z(k))*wzatt*fb
    attb(k) = attb(k) + coeffsatt(k, 1)*tempb
    zb(k) = zb(k) - tempb
  END DO
  DO i=n,1,-1
    DO j=n,1,-1
      tempb = watt*invcovatt(i, j)*fb
      attb(i) = attb(i) + EXP(att(i))*(EXP(att(j))-attka_mean(j))*tempb
      attb(j) = attb(j) + EXP(att(j))*(EXP(att(i))-attka_mean(i))*tempb
    END DO
  END DO
  DO i=n,1,-1
    DO j=n,1,-1
      tempb = wz*invcovzz(i, j)*fb
      zb(i) = zb(i) + (z(j)-z_mean(j))*tempb
      zb(j) = zb(j) + (z(i)-z_mean(i))*tempb
    END DO
  END DO
  piakab = 0.0
  zsimb = 0.0
  DO k=1,n,1
    attb(k) = attb(k) + EXP(att(k))*dr*piakab
    if (z_obs(k)>12) then
      zsimb(n) = zsimb(n) + 2*(zsim(n)-z_obs(k))*wobs*fb
    ELSE IF (zsim(n)>12) THEN
      zsimb(n) = zsimb(n) + 2*(zsim(n)-12)*wobs*fb
    END IF
    zb(k) = zb(k) + zsimb(n)
    piakab = piakab - zsimb(n)
    zsimb(n) = 0.0
    attb(k) = attb(k) + EXP(att(k))*dr*piakab
  END DO
  fb = 0.0
END SUBROUTINE FOBJ_F90T_B
