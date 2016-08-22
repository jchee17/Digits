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

# main functions

gen_training_param <- function(training.data_subset, training.label_subset, lambda) {
  # mu_est
  training_0 = training.data_subset[training.label_subset == 0,]
  training_1 = training.data_subset[training.label_subset == 1,]
  training_2 = training.data_subset[training.label_subset == 2,]
  training_3 = training.data_subset[training.label_subset == 3,]
  training_4 = training.data_subset[training.label_subset == 4,]
  training_5 = training.data_subset[training.label_subset == 5,]
  training_6 = training.data_subset[training.label_subset == 6,]
  training_7 = training.data_subset[training.label_subset == 7,]
  training_8 = training.data_subset[training.label_subset == 8,]
  training_9 = training.data_subset[training.label_subset == 9,]
  
  m = dim(training_0)[1]
  n = dim(training_0)[2]
  mu_est_0    = .colMeans(training_0, m, n)
  mu_est_1    = .colMeans(training_1, m, n)
  mu_est_2    = .colMeans(training_2, m, n)
  mu_est_3    = .colMeans(training_3, m, n)
  mu_est_4    = .colMeans(training_4, m, n)
  mu_est_5    = .colMeans(training_5, m, n)
  mu_est_6    = .colMeans(training_6, m, n)
  mu_est_7    = .colMeans(training_7, m, n)
  mu_est_8    = .colMeans(training_8, m, n)
  mu_est_9    = .colMeans(training_9, m, n)
  
  # each row a mu_k
  mu_est = matrix( c(mu_est_0, mu_est_1, mu_est_2, mu_est_3, 
                     mu_est_4, mu_est_5, mu_est_6, mu_est_7, 
                     mu_est_8, mu_est_9), 
                   nrow=10, ncol=n, byrow=TRUE)
  
  # cov_est
  cov_est = matrix(rep.int(0, 400*400), nrow=400, ncol=400)
  N = dim(training.data_subset)[1]
  for (i in 1:N) {
    x_i = training.data_subset[i,]
    dim(x_i) = c(400,1)
    mu_i = mu_est[training.label_subset[i]+1,]
    dim(mu_i) = c(400,1)
    cov_est = cov_est + (x_i - mu_i) %*% t(x_i - mu_i) 
  }
  cov_est = cov_est / N
  cov_est = (1 - lambda)*cov_est + (lambda/4)*diag(400)
  
  # pi_est
  n_0 = dim(training_0)[1]
  n_1 = dim(training_1)[1]
  n_2 = dim(training_2)[1]
  n_3 = dim(training_3)[1]
  n_4 = dim(training_4)[1]
  n_5 = dim(training_5)[1]
  n_6 = dim(training_6)[1]
  n_7 = dim(training_7)[1]
  n_8 = dim(training_8)[1]
  n_9 = dim(training_9)[1]
  
  pi_est = 1/N * array( c(n_0, n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, n_9))
  
  # return
  output = list("mu_est"=mu_est, "cov_est"=cov_est, "pi_est"=pi_est)
  return(output)
}

bayes_classifier <- function(parameters, x){
  mu_est = parameters$mu_est
  cov_est = parameters$cov_est
  out = which.max(mu_est %*% cov_est %*% x - 0.5 * diag(mu_est %*% cov_est %*% t(mu_est))) - 1
  return(out)
}

eval_classifier <- function(test, labels, parameters){
  miss_classify = 0
  num_test = dim(test)[1]
  print(num_test)
  for (i in 1:num_test) {
    if (bayes_classifier(parameters, test[i,]) != labels[i]) {
      miss_classify = miss_classify + 1
    }
  }
  return(miss_classify)
}

train_lambda <- function(training.data, training.label, ls_lambda) {
  # get training subsets by digit
  training_0 = training.data[training.label == 0,]
  training_1 = training.data[training.label == 1,]
  training_2 = training.data[training.label == 2,]
  training_3 = training.data[training.label == 3,]
  training_4 = training.data[training.label == 4,]
  training_5 = training.data[training.label == 5,]
  training_6 = training.data[training.label == 6,]
  training_7 = training.data[training.label == 7,]
  training_8 = training.data[training.label == 8,]
  training_9 = training.data[training.label == 9,]
  
  # array to hold lambda missclassification
  n_lambda = length(ls_lambda)
  miss_lambda = rep(0, n_lambda)
  
  # 5-fold cross validation
  for (i in 1:5){
    # for each class 
    idx = sample(1:500, 400)
    train_0 = training_0[idx,]
    train_1 = training_1[idx,]
    train_2 = training_2[idx,]
    train_3 = training_3[idx,]
    train_4 = training_4[idx,]
    train_5 = training_5[idx,]
    train_6 = training_6[idx,]
    train_7 = training_7[idx,]
    train_8 = training_8[idx,]
    train_9 = training_9[idx,]
    
    test_0  = training_0[-idx,]
    test_1  = training_1[-idx,]
    test_2  = training_2[-idx,]
    test_3  = training_3[-idx,]
    test_4  = training_4[-idx,]
    test_5  = training_5[-idx,]
    test_6  = training_6[-idx,]
    test_7  = training_7[-idx,]
    test_8  = training_8[-idx,]
    test_9  = training_9[-idx,]
    
    # form matrix subsets and labels
    train_subset = rbind(train_0, train_1, train_2, train_3,
                         train_4, train_5, train_6, train_7,
                         train_8, train_9)
    train_label  = c(rep(0,400), rep(1,400), rep(2,400), 
                     rep(3,400), rep(4,400), rep(5,400),
                     rep(6,400), rep(7,400), rep(8,400),
                     rep(9, 400))
    test_subset  = rbind(test_0, test_1, test_2, test_3,
                        test_4, test_5, test_6, test_7,
                        test_8, test_9)
    test_label   = c(rep(0,100), rep(1,100), rep(2,100), 
                     rep(3,100), rep(4,100), rep(5,100),
                     rep(6,100), rep(7,100), rep(8,100),
                     rep(9, 100))
    
    # go over lambda values
    for (l in 1:n_lambda){
      # get parameters for this training subset
      parameters = gen_training_param(train_subset, train_label, ls_lambda[l])

      # missclassification
      miss_lambda[l] = miss_lambda[l] + eval_classifier(test_subset, test_label, parameters)
    }
  }
  
  return(miss_lambda)
}

# ==========================
ls_lambda = seq(0.1, 0.9, 0.1)
#misscount_lambda = train_lambda(training.data, training.label, ls_lambda)
#lambda_hat = ls_lambda[which.min(misscount_lambda)]
#parameters_hat = gen_training_param(training.data, training.label, lambda_hat)

#error = eval_classifier(test.data, test.label, parameters_hat) / dim(test.data)[1]



  