# UFCMachine_Copy
Copy of UFC Machine

# UFCMachine

UFCMachine is a site which offers UFC fight predictions based on a deep learning model. The model is designed to predict the outcome of future fights based on the historical data from all UFC fights between 1993-2019, and allows users to input two fighters of their choosing using a backend dataset of UFC rostered fighters and their statistics.

The model behind UFCMachine utilizes a Keras driven neural network with eight layers, and was trained using the difference in fight statistics between the winner and loser of each fight.

# Approach

After finding the dataset https://www.kaggle.com/rajeevw/ufcdata containing all UFC fights since 1993, we were inspired to create a predictive model which could provide us with the most probable winner based on which statistics were most highly correlated to winning past matches.

Scraping
-------------
In order to give website users the ability to predict future fights based on searching fighter names instead of statistics, we scraped the career stats for every fighter on the UFC roster from http://ufcstats.com/statistics/fighters. These stats were put into a dataset which related career statistics such as fight record, height, weight, reach, significant strikes landed per minute, and more for each fighters name. This scraping method can be seen in UFC_scraping.ipynb.

![df_fighters](https://github.com/griffinpeifer/UFCMachine/blob/master/df_fighters.PNG)

Cleaning
------------
Once all fighters and their career stats were input into a dataframe, the data was cleaned and all fighters who had missing data fields were eliminated from the dataset, yielding 1672 fighters. This was an acceptable refinement because the fighters who were missing data tended to be older, retired fighters, and as a result they won't have much applicability to a future fight predictor. 

![df_matches with names](https://github.com/griffinpeifer/UFCMachine/blob/master/df_matches_names.PNG)

A value of 1 in the 'result' column indicates a win by the 'Red' fighter.

![df_matches with stat differential](https://github.com/griffinpeifer/UFCMachine/blob/master/df_matches_differential.PNG)

Model Development
-------------
Once the dataset was scraped and cleaned, we used it to enhance the df_matches dataframe, which contained the winning and losing fighters from all matches since 1993. By replacing the fighters names in this dataframe with their career stats, we were able to calculate a stat differential between the two fighters in each match, and train our model to learn which statistical categories were most highly correlated with winning. This can be seen in the UFC_Machine_Learning_Model.pynb file.

Using the newly updated df_matches, an eight layer neural network was developed which achieved 71.35% test accuracy and is being used as a viable prediction model for future fights. The model was finally used in the backend for the UFCMachine website, which was created using Flask and Python, and can be viewed in the Webpage_files.
