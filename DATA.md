# BVS input data

`om/models/lifespan-merged.csv` contains only the seven original columns needed
by the BVS experiment: six nutrition covariates and age at death in weeks.
Values and row order are preserved; unused columns have been omitted.
The loader drops rows with missing required values, builds the interaction
and proportion predictors, standardizes predictors, and adds a fixed intercept.
There are 337 complete observations and 13 selectable predictors.

Source publication:

S. M. Solon-Biet et al. (2014). *The Ratio of Macronutrients, Not Caloric Intake,
Dictates Cardiometabolic Health, Aging, and Longevity in Ad Libitum-Fed Mice.*
Cell Metabolism 19(3), 418–430.

This is third-party research data; attribution is retained separately from the
anonymous submission's authorship. No LSAC data are distributed here.
