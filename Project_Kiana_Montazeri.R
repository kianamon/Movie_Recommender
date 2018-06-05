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
library(mlr)
#####################################################################################
#check for missing packages and install them:
list.of.packages <- c("knitr", "httr", "readr", "dplyr", "tidyr", "XML",
                      "ggplot2", "stringr", "lubridate", "grid", "caret", 
                      "rpart", "Metrics", "e1071", "ranger", "glmnet", "mlr")
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
df_kiana <- kiana %>%
  mutate(Likability = factor(floor(KianaRating) *10)) %>%
  select(-KianaRating)
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
#test and train seperation:
set.seed(1234)
idx <- sample(1:nrow(df_kianatrain), round(0.8 * nrow(df_kianatrain)))
df_train <- df_kianatrain[idx, ]
df_test <- df_kianatrain[-idx, ]
write.csv(df_train, file = "kinoo.csv", row.names=F)
#####################################################################################
#ranger
grid <-  expand.grid(mtry = c(3,4), splitrule = "gini", min.node.size = 10)

fitControl <- trainControl(method = "CV",
                           number = 5,
                           verboseIter = TRUE)
modranger <- ranger(Likability ~ .-URL-Title-Directors-Genres4, df_train)
modranger
modranger$confusion.matrix
print(1-modranger$prediction.error)
#0.31 is the accuracy which is he best so far!
pred <- predict(modranger, df_test)
#####################################################################################

#####################################################################################
#Testing the model on the big set of all the movies:
recom1 <- predict(modranger, movies1)
recom2 <- data.frame(Title=movies1$Title, URL=movies1$URL, Score=movies1$Score,
                     Duration=movies1$Duration, Genres1=movies1$Genres1, 
                     Genres2=movies1$Genres2, Genres3=movies1$Genres3, 
                     Genres4=movies1$Genres4, Directors=movies1$Directors,
                     Num_Votes=movies1$Num_Votes, Likability=recom1$predictions, 
                     stringsAsFactors = TRUE)
recomr <- arrange(recom2, desc(Likability))
write.csv(recomr, file = "SortedRecomranger.csv", row.names=F)

