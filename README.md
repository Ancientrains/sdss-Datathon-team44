# sdss-Datathon-team44
This is the submittable repo for the SDSS Datathon (DUE 03-01 12:30PM)

start after 1:00pm after lunch

Canva Link: https://www.canva.com/design/DAHCpTlUDsY/P2k6uiDXZITQ9-58Nf7pcg/edit?utm_content=DAHCpTlUDsY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton


# This is the information file of the submittable for Team 44 SDSS 2026. 
## Abstract
The data file chosen for team 44 is Public_Service, from it we got famaliar with all of the data predictors such as OCCUPANCY_DATE, LOCATION_POSTAL_CODE, OCCUPANCY_RATE, and more. There are also addition made to the dataframe to help us understand and extrapilate the data further, such as longitude and longitudinal and latitudinal data calculated roughly with the given postal codes. Here we will explain some of the intuitions and reason behind some of the choices made and not made. 

### Data visualiazation and cleaning

We mainly followed the description given with each data file and basic pandas executables such as .isna for cleaning up the data. Then we looked as the spread of the desired responce variable such as occupancy rates with kernel density estimation probability and check over some assumptions using R. We observed the data is very skewed towards the left and right leaning. 
We realized that the data given is a time serise thus we randomly seperated 10 days out of the a total of 2 year to fit out first model.
Since it's a valid probability, we first used a logit transformation to fit a linear model to see how the estimated parameters beheaves and get an basic understanding of how the data over all fitted. Due to the heaviely skewed tail to the left, the cluster of data near 1 and the large amount of catagorical predictors compared to continous ones, we failed to extract any usefull result other than we noticed  all the catagorical data are significant in some sense. some other visualizations we did was the modling comparison between shocks between 2024 jan to 2025 dec, accounting for geopolitical shock such as the introduction of tariffs, the election day, major holidays, and the hottest day/coldest day. < put image here >

### [--put your additional information here--]
