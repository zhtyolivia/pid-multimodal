if(!require(e1071)) install.packages('e1071', dependencies = TRUE)
if(!require(caret)) install.packages('caret', repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages('dplyr', repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages('tidyverse', repos = "http://cran.us.r-project.org")
if(!require(readxl)) install.packages('readxl', repos = "http://cran.us.r-project.org")
if(!require(xlsx)) install.packages('xlsx', repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages('glmnet', repos = "http://cran.us.r-project.org")
if(!require(dummies)) install.packages('dummies', repos = "http://cran.us.r-project.org")
if(!require(mRMRe)) install.packages('mRMRe', repos = "http://cran.us.r-project.org")
if(!require(geometry)) install.packages('geometry', repos = "http://cran.us.r-project.org")
if(!require(survival)) install.packages('survival', repos = "http://cran.us.r-project.org")
if(!require(survminer)) install.packages('survminer', repos = "http://cran.us.r-project.org")
if(!require(ggfortify)) install.packages('ggfortify', repos = "http://cran.us.r-project.org")
if(!require(ggpubr)) install.packages('ggpubr', repos = "http://cran.us.r-project.org")
if(!require(PMA)) install.packages('PMA', repos = "http://cran.us.r-project.org")
if(!require(magrittr)) install.packages('magrittr', repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot)) install.packages('ggcorrplot', repos = "http://cran.us.r-project.org")
if(!require(GGally)) install.packages('GGally', repos = "http://cran.us.r-project.org")
if(!require(glm)) install.packages('glm', repos = "http://cran.us.r-project.org")
if(!require(Metrics)) install.packages('Metrics', repos = "http://cran.us.r-project.org")
if(!require(OptimalCutpoints)) install.packages('(OptimalCutpoints', repos = "http://cran.us.r-project.org")
if(!require(writexl)) install.packages('writexl', repos = "http://cran.us.r-project.org")
if (!require(FSinR)) install.packages('FSinR', repos = "http://cran.us.r-project.org")

## set random seeds 
RNGkind(sample.kind = "Rounding")

## pre-selection 
n1_mrmr_features = 7
n2_mrmr_features = 7

## CV 
num_folds = 5 
num_rep_cv = 10

## cox lasso hyperparams 
lasso_alpha = 0   # 0 - ridge regularization (L2)
                  # 1 - lasso regularization (L1)

#================================================
# Load function and data #
#================================================

# lung radiomic-pathomic 
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
root_dir <- dirname(script_dir)
data_path <- file.path(root_dir, "datasets", "lung_radiopathomic")
print(data_path)

X1 = read_csv(file.path(data_path, "all_radiomics.csv"), col_names = TRUE)
X1 = X1[, -which(names(X1) == "PatientID")]
X1 = X1[, -which(names(X1) == "...1")]
X2 = read_csv(file.path(data_path, "all_pathomics.csv"), col_names = TRUE)
X2 = X2[, -which(names(X2) == "PatientID")]
X2 <- X2[, colSums(is.na(X2)) != nrow(X2)] # remove features with only na (in which case there is an error)
clinical = read_csv(file.path(data_path, "outcome.csv"), col_names = TRUE)
clinical = clinical[, -which(names(clinical) == "PatientID")]
clinical = clinical[, -which(names(clinical) == "...1")]
outcome = read_csv(file.path(data_path, "outcome.csv"), col_names = TRUE)
outcome = outcome[, -which(names(outcome) == "PatientID")]
outcome = outcome[, -which(names(outcome) == "...1")]


#================================================
#                       utils                   #
#================================================

# function to perform inference on validation or test set 
run_inference = function(model, X, y) {
  # Testing the cox regression model #
  y_val = data.frame(status = as.matrix(y$status),
                     time = as.matrix(y$time))
  colnames(y_val) = c('status', 'time') # rename columns 
  
  X_val = as.matrix(X)
  risk_scores_val = predict(model_fit, newx = X_val, s = lasso_lambda)
  c_index_val = apply(risk_scores_val, 2, Cindex, y=y_val)
  
  risk_groups_val = as.factor(case_when(
    risk_scores_val >= risk_threshold ~ "High-risk",
    risk_scores_val < risk_threshold ~ "Low-risk",
  ))
  
  risk_groups_val_numeric = as.factor(case_when(
    risk_groups_val == c('High-risk') ~ 1,
    risk_groups_val == c('Low-risk') ~ 0,
  ))
  
  time_val = y$time
  status_val = y$status
  coxph_data_val = as.data.frame(cbind(time_val, status_val, risk_scores_val))
  
  # Sort features indicated by LASSO feature weight absolute value
  sorted_features_indices = sort.int(abs(lasso_selected_feature_beta), decreasing = TRUE, index.return = TRUE)$ix
  sorted_features_weights = round(sort.int(abs(lasso_selected_feature_beta), decreasing = TRUE, index.return = TRUE)$x, digits = 2)
  selected_features = colnames(X_val[, lasso_selected_feature_indices[sorted_features_indices]])
  selected_features_and_weights = cbind(selected_features, sorted_features_weights) 
  # print(X_val)
  return (list(c_index_val, selected_features_and_weights))
}

normalize_and_fill_na = function(train_features, val_features, test_features) {
  # Normalize training data
  train_mins = apply(train_features, 2, min, na.rm = TRUE)
  train_maxs = apply(train_features, 2, max, na.rm = TRUE)
  train_range = train_maxs - train_mins
  
  train_features = sweep(train_features, 2, train_mins, "-")
  train_features = sweep(train_features, 2, train_range, "/")
  
  # Fill NA with mean in training data
  train_features = apply(train_features, 2, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
  
  # Normalize validation data
  val_features = sweep(val_features, 2, train_mins, "-")
  val_features = sweep(val_features, 2, train_range, "/")
  
  # Fill NA with mean in validation data
  val_features = apply(val_features, 2, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
  
  # Normalize test data
  test_features = sweep(test_features, 2, train_mins, "-")
  test_features = sweep(test_features, 2, train_range, "/")
  
  # Fill NA with mean in test data
  test_features = apply(test_features, 2, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
  
  return (list(train_features = as.data.frame(train_features), 
               val_features = as.data.frame(val_features), 
               test_features = as.data.frame(test_features)))
}

#================================================
#              Feture Selection                 #
#================================================

# perform mrmr selection in the training and apply to val and test. 
run_mrmr_selection = function(features, outcome, features_val, features_test, num_features) { 
  features_with_events = cbind(features, outcome$status)
  features_mrmr_type = mRMR.data(data = as.data.frame(features_with_events))
  features_selected = mRMR.classic(data = features_mrmr_type, target_indices = ncol(features_with_events), feature_count = num_features)
  features_selected_indices = unlist(features_selected@filters)
  selected_features = features[, features_selected_indices]
  
  # apply to validation and test sets 
  features_val = features_val[, features_selected_indices]
  features_test = features_test[, features_selected_indices]
  
  return (list(selected_features, features_val, features_test))
  
}

#================================================
#         Repeated Cross Validation             #
#================================================

# Use the best lambda to train the final model
lasso_lambda = 1 

# initialize a list to store the results from each fold
fold_train_results <- vector("list", num_rep_cv*num_folds)
fold_val_results <- vector("list", num_rep_cv*num_folds)
fold_test_results <- vector("list", num_rep_cv*num_folds)

# Iterate through the random seeds (repetitions)
for  (rep in 1:num_rep_cv) {
  set.seed(rep)
  cat("rep =", rep, "\n")
  
  # 5-fold cross-validation with stratified sampling
  folds = createFolds(outcome$status, k = num_folds, list = TRUE)
  
  for (fold in 1:num_folds) {
    cat("    fold = ", fold, "\n")
    # get the indices of training and test set for this fold
    train_val_indices = unlist(folds[-fold])
    test_indices = folds[[fold]]
    # Split the data into training and test sets
    train_val_X1 = X1[train_val_indices, ]
    test_X1 = X1[test_indices, ]
    train_val_X2 = X2[train_val_indices, ]
    test_X2 = X2[test_indices, ]
    train_val_outcome = outcome[train_val_indices, ]
    test_outcome = outcome[test_indices, ]
    
    val_indices <- createDataPartition(train_val_outcome$status, p = 0.25, list = FALSE)
    train_indices <- setdiff(1:nrow(train_val_outcome), val_indices)
    
    # Create training and validation sets
    train_X1 = train_val_X1[train_indices, ]
    train_X2 = train_val_X2[train_indices, ]
    train_outcome = train_val_outcome[train_indices, ]
    
    val_X1 = train_val_X1[val_indices, ]
    val_X2 = train_val_X2[val_indices, ]
    val_outcome = train_val_outcome[val_indices, ]
    
    # Normalize the training data using min-max normalization and apply stats to val and test 
    X1_norm_results = normalize_and_fill_na(train_X1, val_X1, test_X1)
    train_X1 = X1_norm_results[[1]]
    val_X1 = X1_norm_results[[2]]
    test_X1 = X1_norm_results[[3]]
    X2_norm_results = normalize_and_fill_na(train_X2, val_X2, test_X2)
    train_X2 = X2_norm_results[[1]]
    val_X2 = X2_norm_results[[2]]
    test_X2 = X2_norm_results[[3]]
    
    # mrmr pre-selection
    X1_mrmr_results = run_mrmr_selection(train_X1, train_outcome, val_X1, test_X1, n1_mrmr_features)
    X2_mrmr_results = run_mrmr_selection(train_X2, train_outcome, val_X2, test_X2, n2_mrmr_features)
    
    train_X1 = X1_mrmr_results[[1]]
    val_X1 = X1_mrmr_results[[2]]
    test_X1 = X1_mrmr_results[[3]]
    
    train_X2 = X2_mrmr_results[[1]]
    val_X2 = X2_mrmr_results[[2]]
    test_X2 = X2_mrmr_results[[3]]
    
    # concatenate 2 modalities
    X_train_df = cbind(train_X1, train_X2)
    X_val_df = cbind(val_X1, val_X2)
    X_test_df = cbind(test_X1, test_X2)
    
    # get X and y ready for training the cox regression model
    y_train = data.frame(status = as.matrix(train_outcome$status),
                         time = as.matrix(train_outcome$time))
    colnames(y_train) = c('status', 'time') # rename columns
    y_train = Surv(time = y_train$time, event = y_train$status)
    X_train = as.matrix(X_train_df)
    
    # fit the cox regression model
    model_fit = glmnet(X_train, y_train, family = 'cox', alpha = lasso_alpha, lambda = lasso_lambda)
    
    # Get the risk scores for each patient
    risk_scores_train = predict(model_fit, newx = X_train, s = lasso_lambda)
    c_index_train = apply(risk_scores_train, 2, Cindex, y = y_train) # Calculate concordance index using the risk scores
    num_progression_train = sum(train_outcome[, c('status')] == 1)
    
    coxph_data_train = as.data.frame(cbind(time_train, status_train, risk_scores_train))
    fold_train_results[[(rep-1)*num_folds + fold]] = c_index_train
    
    # ===== Validation =====
    results_val = run_inference(model_fit, X_val_df, val_outcome)
    c_index_val = results_val[[1]]
    fold_val_results[[(rep-1)*num_folds + fold]] <- c_index_val
    
    # ======== Test ========
    results_test = run_inference(model_fit, X_test_df, test_outcome)
    c_index_val = results_test[[1]]
    fold_test_results[[(rep-1)*num_folds + fold]] <- c_index_val
    
  }
}
print(lasso_lambda)

fold_train_results <- unlist(fold_train_results)
print("Training:")
result_mean = round(mean(fold_train_results), 4)
result_sd = round(sd(fold_train_results), 4)
print(paste(result_mean, "±", result_sd))

print("Validation")
fold_val_results <- unlist(fold_val_results)
result_mean = round(mean(fold_val_results), 4)
result_sd = round(sd(fold_val_results), 4)
print(paste(result_mean, "±", result_sd))

print("Test")
fold_test_results <- unlist(fold_test_results)
result_mean = round(mean(fold_test_results), 4)
result_sd = round(sd(fold_test_results), 4)
print(paste(result_mean, "±", result_sd))


