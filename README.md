# PitcherGamePreds

A neural network utilized to predict starting pitcher strikeouts for indivdual games. Primarily used for strikeout props in betting. The initial dataset begins with 456 different features, including pitcher season statistics, pitcher game-by-game statistics and opposing team statistics obtained using the pybaseball scrapers. After applying PCA to reduce feature dimensionality, the dataset is reduced down to **50** features. The model created is a sequential neural network, with optimal hyperparameters determined through the hyperband algorithm.

The data is than recollected after inputting the player name, team, and opposing team to produce a final prediction. The model will be updated frequently in order to maintain accuracy. 

A current issue with the network is that the opposing team data comes in format of city name (ie. Boston), and therefore the teams that share a city (New York, Chicago & Los Angeles) will produce two predictions when inputted as the opposing team. Please consult (https://www.teamrankings.com/mlb/stat/strikeouts-per-game) before predicting with these teams to understand which prediction will be which. The predictions will align with the comparison between team strikeout rates. (ie. if Mets K% is higher than Yankees K% rate than the larger prediction will be for the Mets)

As of August 8th 2023:
New York - Mets strikeout less than Yankees (by 0.32 per game)
Chicago - Sox strikeout less than Cubs (by 0.15 per game)
Los Angeles - Dodgers strikeout less than Angels (by 0.64 per game)

This model should not be used directly to place bets - please use your own discretion along with the model predictions to determine the 'best bets'. As a reccomendation focus your predictions on pitchers with a larger season-long database. The model will be biased in some way for pitchers who have started a minimal amount of games this season. Also be wary of the expected length a pitcher is supposed to pitch.. the model is not aware if a player is an opener or on a certain pitch count. 

Since pitchers do not throw every single pitch, individual pitch statistics have been removed from the model. 

Good Luck!





Credit to pybaseball for data scraping from baseball savant and baseball reference. 
