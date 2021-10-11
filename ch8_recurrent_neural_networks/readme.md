# Ch8 Recurrent Neural Networks

- With CNN we assume that the distribution of data is uniform. However that need not be the case for RNNS
- While CNNs can parse spatial information, RNN are there to parse sequential information

Sequence models are suspect to lot of biases :

1. Anchoring: during a particular Oscar ceremony the rating of the movie goes up even though its the same movie.
2. Hedonic adaptation : The new normal based on good movies, you then only tend to respect good movies.
3. Seasonality :  Santa Claus movie should be shown in December.
4. Infamous: Movies become popular for the wrong reasons.
5. Cult movie syndrome: some movies become cult movies because they are so so bad.

There are other things to consider in sequential model

1. Netflix is famous after school, while trading apps are mostly used when trade is open.
2. Estimating in between observation is interpolation, predicting future is extrapolation. We all know hindsight is easier than foresight.
3. Permutation does not work in sequential model, dog bites man isnt the same as man bites dog.
4. Things like earthquakes are strongly correlated
5. Human interact in sequential nature, they trade blows on twitter sequentially.

- we need statistical tool to deal with all these data.