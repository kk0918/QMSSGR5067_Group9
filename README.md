# QMSSGR5067_Group9

# NLP : Effects of Critic/User Review Language on Movie Perception

## Background
We are interested in the effect that both audience and critic reactions/reviews 
have on the success of a movie release.
 - With more movies being released directly to streaming and fewer being 
    released in theaters, word of mouth reactions have become more critical in evaluating the success of a movie.
    
## Potential Methods:
Sentiment analysis of public reviews and reactions is the most obvious choice 
for approaching our question of interest. 
-  We would likely focus on Rotten Tomatoes reviews for critic reviews 
    since it is the most widely-known and used site for critical reviews 
    
- For user reviews there are many options, including Twitter commentary 
    and reviews left on sites like RT/IMDb/Metacritic.

The sentiment analysis could be cut and compared on different variables 
(e.g. “Top Critics” vs. Not), used to compare to sentiment from other sites, or used in predictive recommendations.

## Potential Questions:
- Critic vs user/public review sentiments in terms of types of language used
- Sentiment of review vs provided score (Predictive models)
  - Language used in review on movies from different platforms, such as Netflix vs Disney, and how they correspond to the critics’ score
- Comparison of scores on different sites using different methods of score calculation
  - I.e. RT uses a binary system, while Metacritic uses a weighted average, but both take in many of the same reviews to produce scores
Movie recommendation model based on the extracted sentiment similarity

## TODOS for team: 

Part 1: Analyzing RT aggregate sentiment scores vs binary provided RT score 
- Preprocess the review text 
- Calculate sentiments for preprocessed reviews 
  - Vader? another method? multiple? 
- Get actual RT critic score from another dataset + merge with current
- Create visualizations
    - Word clouds for different types of movies/genres
    - Examples of sentences/words with large difference between our sentiment score vs RT score 
    - Examples of sentences/words with small difference between our sentiment score vs RT score

Part 2: Adding in box office mojo data + additional trends
- Gather box office mojo data (manually, scrape, or another kaggle dataset)
- Merge datasets 
- Create visualizations
  - Compare score vs box office amount trends
  - Word clouds
- Predictions 
  - What kind of words in reviews would lead to a high box office? 

Part 3: Presentation work/setup