library(KernSmooth);library(ks); 
library(mvtnorm); library(misc3d); library(MASS)
library(mvoutlier); library(outliers)

  # xxi and yyi are data; "out" is the number of ourliers to be given for selecting the bandwidth  
gmcpvalue=function(xxi, yyi, out) {
  n=length(xxi)
  m=20000; Mid=5; dd=seq(-Mid,Mid,length=50000)
  Kf=function(x) {dnorm(x,0,1)*((-sign(x-Mid)*sign(x+Mid)+1)/2)/(pnorm(Mid,0,1)-pnorm(-Mid,0,1))}
  c1=mean(dd^2*Kf(dd))*(max(dd)-min(dd))
  c2=mean((Kf(dd))^2)*(max(dd)-min(dd))
  c3=mean(((dd^2-1)*Kf(dd))^2)*(max(dd)-min(dd))
  EK1=0;  EK2=c1
  h=c1^(-2/5)*c2^(1/5)*c3^(-1/5)/n^(1/5)
  stor1=matrix(0, ncol=n, nrow=m); stor2=stor1 
  A=matrix(0,ncol=2,nrow=6)
  
  xi=xxi; yi=yyi
  rrr=cbind(xi,yi)
  if (out>0) { for (i in 1:out) {rrr=rm.outlier(rrr)} }
  #BCV2=Hpi.diag(rrr)
  #BCV2=Hbcv.diag(rrr,whichbcv=2,amise=FALSE)
   BCV2=Hlscv(rrr)
 
  
  
  hh10=sqrt(BCV2[1,1]);  hh20=sqrt(BCV2[2,2])
  
  
  
  
  ###################
  h10=hh10; h20=hh20; h=h10;  
  xi=xxi; yi=yyi
  ybar=mean(yi); yvar=var(yi)*(length(yi)-1)/length(yi)
  x=seq(sort(xi)[1]-6*h,sort(xi)[n]+6*h,length=m)
  for (j in 1:n)
  {
    stor1[,j]=yi[j]/h10*Kf((x-xi[j])/h10); stor2[,j]=1/h10*Kf((x-xi[j])/h10)
  }
  store1=apply(stor1, 1, mean); store2=apply(stor2, 1, mean)
  tt=(store1)^2/store2; temp0=c(na.omit(tt));  temp0=temp0[temp0>0]
  temp=mean(temp0)*length(temp0)*(x[m]-x[1])/m
  denote1=min((temp-(ybar+h20*EK1)^2)/(yvar+h20^2*EK2),1)
  
 
  rr=seq(1,n,1) 
  for (j in 1:n)
  { 
    tt=(2*yi[j]-store1/store2)*(store1/store2)*(1/h10)*Kf((x-xi[j])/h10)
    temp0=c(na.omit(tt));  temp0=temp0[abs(temp0)>0]
    if (length(temp0)==0) {rr[j=0]} else {
      rr[j]=mean(temp0)*length(temp0)*(x[m]-x[1])/m 
    }
  }
  MM=cbind(rr/(yvar+h20^2*EK2), yi/sqrt(yvar+h20^2*EK2), yi^2)
  A[1,1]=1
  A[2,1]=0-2*ybar/sqrt(yvar+h20^2*EK2)+2*(temp-(ybar)^2)*sqrt(yvar+h20^2*EK2)*ybar/(yvar+h20^2*EK2)^2 
  A[3,1]=0-(temp-(ybar)^2)/(yvar+h20^2*EK2)^2
  A[4:6,1]=c(0,0,0)
  tt10=(temp-(ybar)^2)*(1/(yvar+h20^2*EK2)-1/yvar)
  
  
  
  MM=cbind(MM,matrix(0,ncol=3,nrow=n))
  meanMM=apply(MM, 2,mean)
  Sigma=t(MM)%*%MM/n-meanMM%*%t(meanMM)
  
  SA=t(A)%*%Sigma%*%A
  CI3=sqrt(n)*(denote1-0)/sqrt(SA[1,1])
  pvalue=2*(1-pnorm((CI3),0,1))

  return(c(denote1, tt10, pvalue))
 }

