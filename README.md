# NLP - Bot Detection in Reddit Posts

### Motivation : 
In this project we use NLP to classify Reddit posts based solely on their content to detect those written by reddit's bot `AutoModerator`. 
<br>

### Summary : 
The data contains over 200,000 posts from reddit, with other features such as : 
the `autor`, the number of `upvotes`, etc... Because I specifically want to know if I can distinguish bots from humans we binarize the author flagging the `AutoModerator` (reddit bot).  Ultimately the data is solely : 
<br> 
1) Text in Post
2) Boolean : `IsAutoModerator`
<br>

During the Exploratory Data Analysis portion of this project, I realize that the `AutoModerator` uses many more URLs in its post.  URLs are then flagged using regex command, and counted for the data.  A few different methods are used for text vectorization, but ultimately I use the *Term Frequency - Inverse Document Frequency* method. 
<br>

With prepped data we train a Random Forest Classifier on our data.  


### Results : 

Results on our test set are promissing.  The test size is about 9100 posts, and only one `AutoModerator` post is missclassified.  A closer look at the post in question shows no indication that it is unrepresentative of ohters, but futher inspection could include sentiment analysis, and more fine-tuning of hyperparameters. 
