# Install packages (run once)
install.packages(c("mice","VIM","smotefamily","rpart","randomForest","pROC","corrplot","readxl","glmnet","xgboost","vip","dplyr","ggplot2"))

# Load libraries
library(tidyverse)
library(caret)
library(mice)       
library(VIM)        
library(smotefamily) 
library(rpart)      
library(randomForest) 
library(pROC)       
library(corrplot)   
library(readxl)
library(glmnet)     
library(xgboost)    
library(vip)        
library(dplyr)
library(ggplot2)

# Load data
df <- readxl::read_excel('/Users/armaanhaque/Downloads/E Commerce Dataset 3.xlsx', sheet = "E Comm")

# 1. Data Preprocessing ------------------------------------------------------
# Check missing values
sapply(df, function(x) sum(is.na(x)))

# Visualize missing values
aggr(df, numbers = TRUE, prop = FALSE)

# Imputation using MICE
imputed_data <- mice(df, m = 3, method = "pmm", seed = 123)
df <- complete(imputed_data)

# Drop CustomerID
df <- df %>% select(-CustomerID)

# Corrected categorical variable encoding
cat_cols <- c('PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 
              'PreferedOrderCat', 'MaritalStatus')

# Create dummy variables while preserving the original Churn column
dummy_model <- caret::dummyVars(~ ., data = df %>% select(-Churn))
df_encoded <- predict(dummy_model, newdata = df) %>% 
  as.data.frame() %>%
  mutate(Churn = as.factor(df$Churn))  # Add back the original Churn column

# Verify the structure
str(df_encoded)

# Handle outliers using Winsorization
cap_outliers <- function(x) {
  qnt <- quantile(x, probs = c(0.05, 0.95), na.rm = TRUE)
  x[x < qnt[1]] <- qnt[1]
  x[x > qnt[2]] <- qnt[2]
  x
}

num_cols <- c('Tenure', 'WarehouseToHome', 'NumberOfAddress', 
              'DaySinceLastOrder', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
              'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
              'CashbackAmount', 'CityTier', 'SatisfactionScore')

df <- df %>%
  mutate(across(all_of(num_cols), cap_outliers))

# 2. Handle Class Imbalance --------------------------------------------------
# 1. Convert categorical variables to numeric factors first
df <- df %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.factor), ~as.numeric(.)-1))  # Convert to 0/1 encoding

# 2. Ensure Churn is numeric 0/1 for SMOTE
df$Churn <- as.numeric(as.character(df$Churn))

# 3. Verify all columns are now numeric
str(df)

# Enhanced SMOTE with better control of balancing
set.seed(123)
smote_data <- SMOTE(
  X = df %>% select(-Churn),
  target = df$Churn,
  K = 5,
  dup_size = (nrow(df[df$Churn == 0,]) - nrow(df[df$Churn == 1,])) / nrow(df[df$Churn == 1,])
)

df_balanced <- smote_data$data %>%
  rename(Churn = class) %>%
  mutate(Churn = factor(Churn, levels = c(0, 1), labels = c("No", "Yes")))

# Verify perfect balance
table(df_balanced$Churn)

# 3. Train/Test Split & Scaling ----------------------------------------------
set.seed(42)
trainIndex <- createDataPartition(df_balanced$Churn, p = 0.7, list = FALSE)
train_data <- df_balanced[trainIndex, ]
test_data <- df_balanced[-trainIndex, ]

# Scale numeric features only
preProc <- preProcess(train_data[, num_cols], method = c("range"))
train_scaled <- predict(preProc, train_data)
test_scaled <- predict(preProc, test_data)

# 4. Model Training ---------------------------------------------------------
# Logistic Regression
set.seed(42)
logit_model <- caret::train(
  Churn ~ .,
  data = train_scaled,
  method = "glmnet",
  family = "binomial",
  trControl = trainControl(
    method = "cv", 
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  tuneLength = 3,
  metric = "ROC"
)

# Decision Tree
dt_model <- rpart::rpart(
  Churn ~ .,
  data = train_scaled,
  method = "class",
  control = rpart.control(cp = 0.01, minsplit = 20)
)

# Random Forest
set.seed(42)
rf_model <- randomForest::randomForest(
  Churn ~ .,
  data = train_scaled,
  ntree = 500,
  mtry = floor(sqrt(ncol(train_scaled) - 1)),
  importance = TRUE
)

# XGBoost
set.seed(42)
xgb_model <- caret::train(
  Churn ~ .,
  data = train_scaled,
  method = "xgbTree",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE
  ),
  tuneLength = 3,
  verbosity = 0
)

# 5. Model Evaluation -------------------------------------------------------
evaluate_model <- function(model, test_data, model_name) {
  # Get predictions
  pred_prob <- predict(model, test_data, type = "prob")[,2]
  pred_class <- factor(ifelse(pred_prob > 0.5, "Yes", "No"), 
                       levels = c("No", "Yes"))
  
  # Ensure test_data Churn has same factor levels
  test_data$Churn <- factor(test_data$Churn, levels = c("No", "Yes"))
  
  # Confusion matrix
  cm <- confusionMatrix(pred_class, test_data$Churn, positive = "Yes")
  
  # ROC curve
  roc_obj <- pROC::roc(as.numeric(test_data$Churn)-1, pred_prob)
  
  list(
    Model = model_name,
    Confusion_Matrix = cm,
    ROC_AUC = pROC::auc(roc_obj),
    Accuracy = cm$overall["Accuracy"],
    Precision = cm$byClass["Precision"],
    Recall = cm$byClass["Recall"],
    F1 = cm$byClass["F1"],
    ROC_Object = roc_obj
  )
}

# Check test data structure
str(test_scaled)
table(test_scaled$Churn)

# Check factor levels
levels(test_scaled$Churn)

# Evaluate all models
results <- list(
  evaluate_model(logit_model, test_scaled, "Logistic Regression"),
  evaluate_model(dt_model, test_scaled, "Decision Tree"),
  evaluate_model(rf_model, test_scaled, "Random Forest"),
  evaluate_model(xgb_model, test_scaled, "XGBoost")
)

# Print confusion matrices
lapply(results, function(x) x$Confusion_Matrix)

# Compare AUC values
sapply(results, function(x) x$ROC_AUC)
# 6. Results & Visualization ------------------------------------------------
# Performance comparison
metrics_df <- do.call(rbind, lapply(results, function(x) {
  data.frame(
    Model = x$Model,
    Accuracy = x$Accuracy,
    AUC = x$ROC_AUC,
    Precision = x$Precision,
    Recall = x$Recall,
    F1 = x$F1
  )
}))

print(knitr::kable(metrics_df, digits = 3))

# ROC curves
ggroc(lapply(results, `[[`, "ROC_Object")) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "ROC Curves Comparison") +
  scale_color_brewer(palette = "Set1") +
  theme_minimal()

# Feature importance
vip(rf_model, num_features = 15) + 
  labs(title = "Random Forest Feature Importance")

vip(xgb_model$finalModel, num_features = 15) +
  labs(title = "XGBoost Feature Importance")


# Enhanced feature importance visualization
plot_importance <- function(importance_df, title) {
  ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_col(fill = "#3A506B") +
    coord_flip() +
    labs(title = title, x = "", y = "Importance Score") +
    theme_minimal()
}

# For Random Forest
rf_importance <- data.frame(
  Feature = rownames(rf_model$importance),
  Importance = rf_model$importance[, "MeanDecreaseAccuracy"]
)
plot_importance(rf_importance, "Random Forest - Top Predictive Features")

# For XGBoost
xgb_importance <- xgb.importance(model = xgb_model$finalModel)

# Convert to data frame with standardized column names
xgb_importance_df <- data.frame(
  Feature = xgb_importance$Feature,
  Importance = xgb_importance$Gain  # Using Gain as importance measure
)

# Plot with corrected column names
ggplot(xgb_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "#BA1F33") +
  coord_flip() +
  labs(title = "XGBoost - Top Predictive Features", 
       x = "", 
       y = "Importance (Gain)") +
  theme_minimal()

#PR-AUC Curve
# 1. Install and load the PRROC package (if not already done)
if (!require(PRROC)) install.packages("PRROC")
library(PRROC)

# 2. Update your evaluate_model() function
evaluate_model <- function(model, test_data, model_name) {
  # Get predicted probabilities for the positive class ("Yes")
  pred_prob <- predict(model, test_data, type = "prob")[, "Yes"]
  
  # Convert actual classes to binary (0="No", 1="Yes")
  actual_binary <- ifelse(test_data$Churn == "Yes", 1, 0)
  
  # Confusion Matrix (unchanged)
  pred_class <- factor(ifelse(pred_prob > 0.5, "Yes", "No"), 
                       levels = c("No", "Yes"))
  cm <- confusionMatrix(pred_class, test_data$Churn, positive = "Yes")
  
  # ROC curve (unchanged)
  roc_obj <- pROC::roc(actual_binary, pred_prob)
  
  # PR Curve - fixed version
  pr_curve <- pr.curve(
    scores.class0 = pred_prob[actual_binary == 1],  # Scores for positive class
    scores.class1 = pred_prob[actual_binary == 0],  # Scores for negative class
    curve = TRUE
  )
  
  list(
    Model = model_name,
    Confusion_Matrix = cm,
    ROC_AUC = pROC::auc(roc_obj),
    PR_AUC = pr_curve$auc.integral,
    Precision = cm$byClass["Precision"],
    Recall = cm$byClass["Recall"],
    F1 = cm$byClass["F1"],
    ROC_Object = roc_obj,
    PR_Object = pr_curve
  )
}

# 3. Re-evaluate models
results <- list(
  evaluate_model(logit_model, test_scaled, "Logistic Regression"),
  evaluate_model(dt_model, test_scaled, "Decision Tree"),
  evaluate_model(rf_model, test_scaled, "Random Forest"),
  evaluate_model(xgb_model, test_scaled, "XGBoost")
)

# 4. Plot PR Curves
plot_pr_curves <- function(results_list) {
  # Set up plot
  plot(NULL, xlim = c(0, 1), ylim = c(0, 1), 
       xlab = "Recall", ylab = "Precision",
       main = "Precision-Recall Curves")
  abline(h = mean(as.numeric(test_scaled$Churn) == "Yes"), 
         lty = 2, col = "gray")  # Baseline
  
  # Add curves for each model
  colors <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3")
  for (i in 1:length(results_list)) {
    lines(results_list[[i]]$PR_Object$curve[,1:2], 
          col = colors[i], lwd = 2)
  }
  
  # Add legend
  legend("topright",
         legend = sapply(results_list, `[[`, "Model"),
         col = colors, lwd = 2, cex = 0.8)
}

# Generate the plot
plot_pr_curves(results)

# Print PR-AUC values
cat("\nPR-AUC Values:\n")
for (res in results) {
  cat(sprintf("%-20s: %.3f\n", res$Model, res$PR_AUC))
}