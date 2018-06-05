#Set the working directory:
setwd("/Users/Kianamon/R/project")
rm(list=ls())
#####################################################################################
#libraries in use:
library(knitr)
library(httr)
library(readr)
library(dplyr)
library(tidyr)
library(XML)
library(ggplot2)
library(stringr)
library(lubridate)
library(grid)
library(caret)
library(glmnet)
library(ranger)
library(e1071)
library(Metrics)
library(rpart)  
library(rpart.plot)
library(ModelMetrics)   
library(ipred)  
library(randomForest)
library(gbm)  
library(ROCR)
library(mlr)
library(xgboost)
library(tidyverse)
library(magrittr)
library(data.table)
library(mosaic)
library(Ckmeans.1d.dp)
library(archdata)
#####################################################################################
#check for missing packages and install them:
list.of.packages <- c("knitr", "httr", "readr", "dplyr", "tidyr", "XML",
                      "ggplot2", "stringr", "lubridate", "grid", "caret", 
                      "rpart", "Metrics", "e1071", "ranger", "glmnet", 
                      "randomForest", "ROCR", "gbm", "ipred", "ModelMetrics", 
                      "rpart.plot", "xgboost", "tidyverse", "magrittr", "mosaic",
                      "Ckmeans.1d.dp", "archdata")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
#####################################################################################
#downloading the two main data sets:
#All the movies from IMDB website:
GET("https://raw.githubusercontent.com/kianamon/MovieRecom/master/movies.csv", 
    write_disk("movies.csv", overwrite = TRUE))
df_movies <- read_csv("movies.csv")
#All the movies that I have watched:
GET("https://raw.githubusercontent.com/kianamon/MovieRecom/master/kiana_watchlist.csv", 
    write_disk("kianamovies.csv", overwrite = TRUE))
#kimia's movies:
#df_kianamovies <- read_csv("WATCHLIST.csv")
#kiana's movies:
df_kianamovies <- read_csv("kianamovies.csv")
#####################################################################################
#renaming the columns for convenience:
names(df_kianamovies)[names(df_kianamovies) == 'IMDb Rating'] <- 'Score'
names(df_kianamovies)[names(df_kianamovies) == 'Runtime (mins)'] <- 'Duration'
names(df_kianamovies)[names(df_kianamovies) == 'Your Rating'] <- 'KianaRating'
names(df_kianamovies)[names(df_kianamovies) == 'Title Type'] <- 'Type'
names(df_kianamovies)[names(df_kianamovies) == 'Num Votes'] <- 'Num_Votes'

names(df_movies)[names(df_movies) == 'movie_title'] <- 'Title'
names(df_movies)[names(df_movies) == 'movie_imdb_link'] <- 'URL'
names(df_movies)[names(df_movies) == 'num_voted_users'] <- 'Num_Votes'
names(df_movies)[names(df_movies) == 'movie_title'] <- 'Title'
names(df_movies)[names(df_movies) == 'director_name'] <- 'Directors'
names(df_movies)[names(df_movies) == 'imdb_score'] <- 'Score'
names(df_movies)[names(df_movies) == 'duration'] <- 'Duration'
names(df_movies)[names(df_movies) == 'genres'] <- 'Genres'
#####################################################################################
#we have a big data set, we train our model for the kiana data set and apply the results 
#to the big data set and calculate the likability of the movie for the big data set.
#The output of the system is the Likability of each movie.
movies5 <- df_movies %>%
  select(Title, URL, Score, Duration, Genres, Directors, Num_Votes) 

movies1 <- movies5 %>% 
  separate(Genres, into = paste("Genres", 1:4, sep = ""), sep = "\\|") 
movies1$Genres2[is.na(movies1$Genres2)] <- movies1$Genres1
movies1$Genres3[is.na(movies1$Genres3)] <- movies1$Genres1
movies1$Genres4[is.na(movies1$Genres4)] <- movies1$Genres1
movies1$Directors[is.na(movies1$Directors)] <- "Kiana"
movies1$Genres1 <- as.factor(movies1$Genres1)
movies1$Genres2 <- as.factor(movies1$Genres2)
movies1$Genres3 <- as.factor(movies1$Genres3)
movies1$Genres4 <- as.factor(movies1$Genres4)
movies1$Likability[is.na(movies1$Likability)] <- 50
movies1$Duration[is.na(movies1$Duration)] <- 100
colSums(is.na(movies1))

kiana <- df_kianamovies %>%
  select(Title, URL, Score, Duration, Genres, Directors, Num_Votes, KianaRating)
kiana$KianaRating[is.na(kiana$KianaRating)] <- 5
#examining user's watched movies to see a pattern:
#I am definig a factor as Likability to calculate a score for each movie,
#Likability goes from 0 to 100 which is the probabilty of me liking a movie
df_kianatemp <- kiana %>%
  mutate(Likability = KianaRating *10)
df_kiana <- df_kianatemp %>%
  select(-KianaRating)
names(df_kiana)
head(df_kiana)
#####################################################################################
#now we should manipulate the data to do the analysis:
df_kiana1 <- as.data.frame(df_kiana) %>% 
  separate(Genres, into = paste("Genres", 1:4, sep = ""), sep = ",")
head(df_kiana1)
df_kianatrain <- df_kiana1 
#  select(-KianaRating, -URL, -Title) 
df_kianatrain$Genres2[is.na(df_kianatrain$Genres2)] <- df_kianatrain$Genres1
df_kianatrain$Genres3[is.na(df_kianatrain$Genres3)] <- df_kianatrain$Genres1
df_kianatrain$Genres4[is.na(df_kianatrain$Genres4)] <- df_kianatrain$Genres1
df_kianatrain$Directors[is.na(df_kianatrain$Directors)] <- "Kiana"
df_kianatrain$Genres1 <- as.factor(df_kianatrain$Genres1)
df_kianatrain$Genres2 <- as.factor(df_kianatrain$Genres2)
df_kianatrain$Genres3 <- as.factor(df_kianatrain$Genres3)
df_kianatrain$Genres4 <- as.factor(df_kianatrain$Genres4)
df_kianatrain$Likability[is.na(df_kianatrain$Likability)] <- 50
df_kianatrain$Duration[is.na(df_kianatrain$Duration)] <- 100
colSums(is.na(df_kianatrain))
head(df_kianatrain)
#####################################################################################
#
#Now we are done with cleaning the data, we start modeling:##########################
#
#####################################################################################
#test and train seperation:
smp_size <- floor(0.80 * nrow(df_kianatrain))
# set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(df_kianatrain)), size = smp_size)
train <- df_kianatrain[train_ind, ]
test <- df_kianatrain[-train_ind, ]
head(train)
#####################################################################################
########################################XGBoost######################################
#####################################################################################
#convert data frame to data table
setDT(train) 
setDT(test)
df_train_table <- df_kianatrain
setDT(df_train_table)
#using one hot encoding 
labels <- train$Likability 
ts_label <- test$Likability
new_tr <- model.matrix(~.+0,data = train[,-c("Likability"),with=F]) 
new_ts <- model.matrix(~.+0,data = test[,-c("Likability"),with=F])
newrealtest <- model.matrix(~.+0,data = movies1)
#convert factor to numeric 
labels <- as.numeric(labels)
ts_label <- as.numeric(ts_label)
#Convert the training and testing sets into DMatrixes: 
#DMatrix is the recommended class in xgboost
#preparing matrix 
X_train <- xgb.DMatrix(data = new_tr,label = labels) 
X_test <- xgb.DMatrix(data = new_ts,label=ts_label)
drealtest <- xgb.DMatrix(data = newrealtest)
y_train <- train$Likability
y_test <- test$Likability

#Specify cross-validation method and number of folds. Also enable parallel computation
xgb_trcontrol = trainControl(
  method = "cv",
  number = 5,  
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE
)
#This is the grid space to search for the best hyperparameters
xgbGrid <- expand.grid(nrounds = c(100,200),  
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5), 
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1
)
#train the model
set.seed(7383) 
xgb_model <- caret::train(
  X_train, y_train,  
  trControl = xgb_trcontrol,
  tuneGrid = xgbGrid,
  method = "xgbTree"
)
#choosing the best hyperparameter:

predicted1 <- predict(xgb_model, X_test)
predicted <- 10*floor(predicted1/10)+10
residuals <- y_test - predicted
RMSE <- sqrt(mean(residuals^2))
cat('The root mean square error of the test data is ', round(RMSE,3),'\n')

y_test_mean <- mean(y_test)
# Calculate total sum of squares
tss <-  sum((y_test - y_test_mean)^2 )
# Calculate residual sum of squares
rss <-  sum(residuals^2)
# Calculate R-squared
rsq  <-  1 - (rss/tss)
cat('The R-square of the test data is ', round(rsq,3), '\n')
#Plotting:
options(repr.plot.width=8, repr.plot.height=4)
my_data <- as.data.frame(cbind(predicted = predicted,
                              observed = y_test))
# Plot predictions vs test data
ggplot(my_data,aes(predicted, observed)) + 
  geom_point(color = "darkred", alpha = 0.5) + 
  geom_smooth(method=lm)+ ggtitle('Linear Regression') + 
  ggtitle("Extreme Gradient Boosting: Prediction vs Test Data") +
  xlab("Predecited Probability to Like a Movie") + 
  ylab("Observed Probability") + 
  theme(plot.title = element_text(color="darkgreen",size=16,hjust = 0.5),
        axis.text.y = element_text(size=12), axis.text.x = element_text(size=12,hjust=.5),
        axis.title.x = element_text(size=14), axis.title.y = element_text(size=14))

finalpred1 <- predict(xgb_model, drealtest)
finalpredicted <- 10*floor(finalpred1/10)+10
#####################################################################################
#submission
recom2 <- data.frame(Title=movies1$Title, URL=movies1$URL, Score=movies1$Score,
                     Duration=movies1$Duration, Genres1=movies1$Genres1, 
                     Genres2=movies1$Genres2, Genres3=movies1$Genres3, 
                     Genres4=movies1$Genres4, Directors=movies1$Directors,
                     Num_Votes=movies1$Num_Votes, Likability=finalpredicted,
                     stringsAsFactors = TRUE)
recomr <- arrange(recom2, desc(Likability))
write.csv(recomr, file = "SortedRecomXGB.csv", row.names=F)