# sdss-Datathon-team44
This is the submittable repo for the SDSS Datathon (DUE 03-01 12:30PM)

start after 1:00pm after lunch

Canva Link: https://www.canva.com/design/DAHCpTlUDsY/P2k6uiDXZITQ9-58Nf7pcg/edit?utm_content=DAHCpTlUDsY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton


# This is the information file of the submittable for Team 44 SDSS 2026. 
## Abstract
The data file chosen for team 44 is Public_Service, from it we got famaliar with all of the data predictors such as OCCUPANCY_DATE, LOCATION_POSTAL_CODE, OCCUPANCY_RATE, and more. There are also addition made to the dataframe to help us understand and extrapilate the data further, such as longitude and longitudinal and latitudinal data calculated roughly with the given postal codes. Here we will explain some of the intuitions and reason behind some of the choices made and not made. 

## Team Contributions
| Member | Role |
|---|---|
| A | Put it here |
| B | Put it here |
| C | Put it here |
| Qijun Xie | Key Insights on System-Wide Shelter Capacity Pressure, Recommendations |


### Data visualiazation and cleaning

We mainly followed the description given with each data file and basic pandas executables such as .isna for cleaning up the data. Then we looked as the spread of the desired responce variable such as occupancy rates with kernel density estimation probability and check over some assumptions using R. We observed the data is very skewed towards the left and right leaning. 
We realized that the data given is a time serise thus we randomly seperated 10 days out of the a total of 2 year to fit out first model.
Since it's a valid probability, we first used a logit transformation to fit a linear model to see how the estimated parameters beheaves and get an basic understanding of how the data over all fitted. Due to the heaviely skewed tail to the left, the cluster of data near 1 and the large amount of catagorical predictors compared to continous ones, we failed to extract any usefull result other than we noticed  all the catagorical data are significant in some sense. some other visualizations we did was the modling comparison between shocks between 2024 jan to 2025 dec, accounting for geopolitical shock such as the introduction of tariffs, the election day, major holidays, and the hottest day/coldest day. < put image here >

### Key Insights on System-Wide Shelter Capacity Pressure
This analysis addresses the following five research questions:
| # | Question |
|---|---|
| Q1 | Which sectors are most under pressure, and how consistently? |
| Q2 | Does the season or time of year matter? |
| Q3 | How much does unavailable capacity make things worse? |
| Q4 | Which sectors are most operationally fragile? (Fragility Index) |
| Q5 | What happens if demand increases or capacity is added? (Simulation) |

## Key Findings

1. **The system has almost no buffer.** The average occupancy rate across all records is 97.2%,
   with 78% of all daily records at exactly 100% occupancy.

2. **Women's shelters are the most consistently strained sector**, at full capacity 87% of all days
   and achieving the highest fragility score (Critical).

3. **The official numbers understate the real pressure.** When unavailable beds are factored out,
   every sector except Youth operates above 100% effective occupancy — meaning the system is
   already in deficit before new arrivals walk in.

4. **The pressure is year-round, not seasonal.** No month in 2024 or 2025 dropped below 94%
   occupancy. There is no natural recovery window.

5. **The system is one small shock away from breaking down.** A demand increase of just +2%
   pushes 83–89% of Women, Families, and Men records over capacity. However, adding just
   +10 beds per program could drop average occupancy from ~98% to ~75%.

## Recommendations

| # | Recommendation | Data Support |
|---|---|---|
| 1 | Prioritize buffer capacity in Women and Families sectors | Q1: 87% days at full capacity; Q4: Critical fragility score |
| 2 | Implement real-time bed availability tracking system | Q3: Unavailable beds push effective occupancy above 100% |
| 3 | Create transitional housing pathways for long-stay residents | Q5: High occupancy and low turnover leaves no room for new arrivals |
| 4 | Expand support services for Youth mental health and 2SLGBTQI+ | Kerman et al. (2025): Youth sector has 2.63x higher self-injury rate; Canada PiT Count 2024: 25% of homeless youth identify as 2SLGBTQI+ |




### [--put your additional information here--]
