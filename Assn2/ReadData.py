import data as Data

cities = Data.cities # set of cities
scenarios = Data.scenarios # set of scenarios
theta = Data.theta # unit cost of delivery to city n in the first stage
theta_s = Data.theta_prime # unit cost of transportion between city n and center in the second stage

h = Data.h # unit cost of unused inventory
g = Data.g # unit cost of shortage 
I = Data.I # inventory of the center at the beginning
Yn = Data.Yn # inventory of city n at the beginning
demand = Data.demand # demand of city n under scenario k
prob = 1.0/len(scenarios) # probability of scenario k
