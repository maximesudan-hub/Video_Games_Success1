I would like to propose the following project for my final assignment. The goal is to analyze and
predict the success of a video game based on its main characteristics, using methods covered in
the course.

The central question is: Which factors best explain the commercial and critical success of a game?
To answer this, I will use the public dataset "Video Game Sales with Ratings" from Kaggle, which
contains more than 16,000 titles released between 1980 and 2016. The variables include genre,
platform, publisher, release year, ESRB rating, critic and user scores, and global sales.

The project will combine supervised and unsupervised learning methods:

- Linear Regression (and possibly Ridge regularization) to predict global sales and critic scores.
- Classification models such as k-Nearest Neighbors, Decision Tree, and Naive Bayes to identify
  whether a game is "successful" (above a given sales threshold).
- As a complementary analysis, Principal Component Analysis (PCA) and k-means clustering will be
  used to group similar games and reveal common patterns across genres and platforms.
- Model performance will be evaluated through cross-validation and metrics such as R², RMSE, and
  F1-score.
