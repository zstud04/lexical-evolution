source('GMC.R')
source('Equality.R')


rho <- 0.5
mu <- c(0.0001,-0.0001)
phi <- c(0.2,0.8)
theta <- c(0.3,-0.7)
ns <- 200
d <- ts(matrix(0,ncol=2,nrow=ns+1))
e <- ts(rmvnorm(ns+1,sigma=cbind(c(1,rho),c(rho,1))))
for(i in 2:ns+1)
  d[i,] <- mu + phi*d[i-1,] - theta*(e[i-1,]+e[i,])

xxi <- d[2:ns+1,1]
yyi <- d[2:ns+1,2]

out <- 0

gmcxgy <- gmcpvalue(xxi,yyi,out)
gmcygx <- gmcpvalue(yyi, xxi, out)

gmceq <- gmcPvalue(yyi,xxi,out)

