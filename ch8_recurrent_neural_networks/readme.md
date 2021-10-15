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

Autoregressive models - models that perform regression on themselves. That is we consider them for a small time window Tau. If Tau is one we have a Markov model.
Latent Autoregressive Models - It keeps a summary ht of past observations, and at the same time update ht in addition to prediction xt

We predict this on the assumption that dynamics of the sequence wont change. This is called stationary.

Causality means that reverse would also be true.Going back and predicting in reverse through model would also work.

![](simple_seq.png)

### skip ahead prediction:

Generally, for an observed sequence up to xt
, its predicted output xˆt+k at time step t + k is called
the k-step-ahead prediction. Since we have observed up to x604, its k-step-ahead prediction is xˆ604+k.
However multistep prediction ait so simple:

![](bad_multistep_prediction.png)

s the above example shows, this is a spectacular failure. The predictions decay to a constant
pretty quickly after a few prediction steps. Why did the algorithm work so poorly? This is ultimately due to the fact that the errors build up. Let us say that after step 1 we have some error
ϵ1 = ¯ϵ. Now the input for step 2 is perturbed by ϵ1, hence we suffer some error in the order of
ϵ2 = ¯ϵ + cϵ1 for some constant c, and so on. The error can diverge rather rapidly from the true
observations. This is a common phenomenon. For instance, weather forecasts for the next 24
hours tend to be pretty accurate but beyond that the accuracy declines rapidly. We will discuss
methods for improving this throughout this chapter and beyond
