# PitcherGamePreds

A neural network utilized to predict starting pitcher strikeouts for indivdual games. Primarily used for strikeout props in betting. The initial dataset begins with 456 different features, including pitcher season statistics, pitcher game-by-game statistics and opposing team statistics obtained using the pybaseball scrapers. After applying PCA to reduce feature dimensionality, the dataset is reduced down to **50** features. The model created is a sequential neural network, with optimal hyperparameters determined through the hyperband algorithm.

A current issue with the network is that the opposing team data comes in format of city name (ie. Boston), and therefore the teams that share a city (New York, Chicago & Los Angeles) are all grouped together in the model, making their performace unreliable. Please consult the current team strikeout rates (https://www.teamrankings.com/mlb/stat/strikeouts-per-game) before predicting with these teams to understand the possible impact this grouping has on the prediction

As of August 8th 2023:
New York - Mets strikeout less than Yankees (by 0.32 per game)
Chicago - Sox strikeout less than Cubs (by 0.15 per game)
Los Angeles - Dodgers strikeout less than Angels (by 0.64 per game)

With these current strikeout rates it seems that Los Angeles would have the only **significant** effect, therefore if predicting against the dodgers reduce the prediction slightly (~ 0.3) and vice versa for the Angels.

Since pitchers do not throw every single pitch, "missing" individual pitch statistics have been imputed with the overall mean for that statistic. 





Credit to pybaseball for data scraping from baseball savant and baseball reference. 
