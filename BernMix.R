# Jerry Chee
# Pattern Recognition Project, LDA
setwd("/home/jerry/GoogleDrive/Documents/College/3rd Year/Spring Quarter/Pattern Recognition/")
load("digits.RData")

# setup data
num.class <- dim(training.data)[1] # Number of classes
num.training <- dim(training.data)[2] # Number of training data per class
d <- prod(dim(training.data)[3:4]) # Dimension of each training image (rowsxcolumns)
num.test <- dim(test.data)[2] # Number of test data
dim(training.data) <- c(num.class * num.training, d) # Reshape training data to 2-dim matrix
dim(test.data) <- c(num.class * num.test, d) # Same for test.
training.label <- rep(0:9, num.training) # Labels of training data.
test.label <- rep(0:9, num.test) # Labels of test data
D = 400
test_0 = test.data[test.label == 0,]
test_1 = test.data[test.label == 1,]
test_2 = test.data[test.label == 2,]
test_3 = test.data[test.label == 3,]
test_4 = test.data[test.label == 4,]
test_5 = test.data[test.label == 5,]
test_6 = test.data[test.label == 6,]
test_7 = test.data[test.label == 7,]
test_8 = test.data[test.label == 8,]
test_9 = test.data[test.label == 9,]


log_mixture_component <- function(x_i, pi_m, mu_m) {
  tmp = 0.0
  for (j in 1:D) {
    #print(x_i[j])
    #print(log(mu_m[i]))
    tmp = tmp + x_i[j] * log(mu_m) + (1 - x_i[j]) * log(1 - mu_m)
  }
  return(log(pi_m) + tmp)
}
eval_mixture <- function(x_i, pi, mu) {
  M = length(pi)
  out = 0
  for (m in 1:M) {
    likelihood = 1
    for (j in 1:D) {
      likelihood = likelihood * mu[m,j]^x_i[j] * (1-mu[m,j])^(1-x_i[j])
    }
    out = out + pi[m] * likelihood
  }
  return(out)  
}

# E-Step
gamma_mi <- function(x_i, pi, mu, m) {
  M = length(pi)
  ls_gamma = rep(0, M)
  for (alpha in 1:M) {
    ls_gamma[alpha] = log_mixture_component(x_i, pi[alpha], mu[alpha])
  }
  component_max = which.max(ls_gamma)
  
  denom = 0.0
  for (alpha in 1:M) {
    denom = denom + exp(ls_gamma[alpha] - component_max)
  }
  return(exp(ls_gamma[m] - component_max) / denom)
}
gamma_all <- function(X, pi, mu) {
  #n x M matrix
  n = dim(X)[1]
  M = length(pi)
  gamma = matrix(rep(0, n*M), nrow=n, ncol=M)
  
  for (i in 1:n) {
    for (m in 1:M) {
      gamma[i,m] = gamma_mi(X[i,], pi, mu, m)
    }
  }
  return(gamma)
}
# M-Step
pi_m_hat <- function(X, gamma_m) {
  n = dim(X)[1]
  M = length(pi)
  numerator = sum(gamma_m) + 1
  return(numerator / (n+M))
}
mu_mj_hat <- function(X, gamma_m, j) {
  n = dim(X)[1]
  numerator = 1.0
  denom = 0
  # sum to get numerator
  for (i in 1:n) {
    numerator = numerator + gamma_m[i] * X[i,j]
  }
  # sum to get denominator
  denom = sum(gamma_m) + 2
  return(numerator / denom)
}
pi_all <- function(X, gamma) {
  M = dim(gamma)[2]
  pi = rep(0, M)
  for (m in 1:M) {
    pi[m] = pi_m_hat(X, gamma[,m])
  }
  return(pi)
}
mu_all <- function(X, gamma) {
  M = dim(gamma)[2]
  mu = matrix(rep(0, M*D), nrow=M, ncol=D)
  for (m in 1:M) {
    for (j in 1:D) {
      mu[m,j] = mu_mj_hat(X, gamma[,m], j)
    }
  }
  return(mu)
}

# Q() + lnp(theta) eval
Q_theta_eval <- function(X, pi_new, pi_old, mu_new, mu_old) {
  n = dim(X)[1]
  M = length(pi_new)
  gamma_old = gamma_all(X, pi_old, mu_old)
  val = 0.0
  # Q(theta, theta_old)
  for (i in 1:n) {
    for (m in 1:M) {
      tmp = 0
      for (j in 1:D) {
        tmp = tmp + X[i,j]*log(mu_new[m,j])+ (1-X[i,j])*log(1-mu_new[m,j])
      }
      val = val + gamma_old[i,m] * (log(pi_new[m]) + tmp)
    }
  }
  # ln(p(theta))
  for (m in 1:M) {
    tmp = 0
    for (j in 1:D) {
      tmp = tmp + log(mu_new[m,j]) + log(1-mu_new[m,j])
    }
    val = val + log(pi_new[m]) + tmp
  }
  return(val)
}

# EM algorithm
init_EM <- function(X, M) {
  n = dim(X)[1]
  # Z
  Z = rep(0, n*M)
  dim(Z) = c(n, M)
  for (i in 1:n) {
    j = sample(1:M, 1)
    Z[i,j] = 1
  }
  # init pi
  pi = pi_all(X, Z)
  # init mu
  mu = mu_all(X, Z)
  out = list("pi"=pi, "mu"=mu)
  return(out)
}

EM <- function(X, M, epsilon) {
  init = init_EM(X,M)
  pi_old = init$pi
  mu_old = init$mu
  Q_old = 0
  error = epsilon + 1
  
  while (error > epsilon) {
    # E-step
    print("E-step")
    gamma_new = gamma_all(X, pi_old, mu_old)
    # M-step
    print("M-step")
    pi_new = pi_all(X, gamma_new)
    mu_new = mu_all(X, gamma_new)
    print("Eval")
    # Eval Q+lnp(theta)
    Q_new = Q_theta_eval(X, pi_new, pi_old, mu_new, mu_old)
    error = abs((Q_new - Q_old) / Q_new)
    print(error)
    # switch variables
    pi_old = pi_new
    mu_old = mu_new
    Q_old = Q_new
  }
  out = list("pi"=pi_new, "mu"=mu_new)
  return(out)
}

# Classifier
classify <- function(test_i, pi_a, pi_b, mu_a, mu_b, class_a, class_b) {
  p_a = eval_mixture(test_i, pi_a, mu_a)
  p_b = eval_mixture(test_i, pi_b, mu_b)
  if (p_a > p_b) {
    return(class_a)
  } else {
    return(class_b)
  }
}
eval_classifier <- function(test, label, pi_a, pi_b, mu_a, mu_b, class_a, class_b) {
  n = dim(test)[1]
  miss = 0
  for (i in 1:n) {
    if (classify(test[i,], pi_a, pi_b, mu_a, mu_b, class_a, class_b) != label[i]) {
      miss = miss + 1
    }
  }
  return(miss)
}
