###Data Simulator

###Christian Rodriguez
###crodriguez0874@gmail.com
###06/15/19

###Summary: Using the Simular_Functions.R script, this script will handle the
###data wrangling and write training/test sets in csv format. Here, we have a
###a little bit of data merging, incorporating moveset, and writing the data.


###Analysis corresponds to data provided by Kaggle users Rounak Banik.
###Data uses info from the Gen VII pokedex on Serebii.
###Data downloaded on 05/30/19.

################################################################################
###Loading libraries, functions, and data
################################################################################

library(readr)
library(dplyr)
source('Simulator_Functions.R')

pokemon <- read_csv("data/pokemon.csv")
pokemon2 <- read_csv("data/pokemon2.csv")
type <- read_csv("data/type-chart.csv")
natures <- read_csv("data/natures.csv")
movesets <- read_csv("data/movesets.csv")

################################################################################
###Data wrangling special cases and legendary status
################################################################################

###The other forms of Zygarde are missing their abilities (one 10% and 50%)
###We assume they have the same potential abilities as Zygarde 10% forme
###with id 1013.
pokemon2$ability1[c(1013, 1014)] <- pokemon2$ability1[c(1012)]
pokemon2$ability2[c(1013, 1014)] <- pokemon2$ability2[c(1012)]
pokemon2$abilityH[c(1013, 1014)] <- pokemon2$abilityH[c(1012)]

###Using pokemon.csv to extract legendary status
legendary_pokemon <- pokemon[, c('pokedex_number', 'is_legendary')]
###Updating legendary roster
new_legendary_pokemon  <- data.frame(pokedex_number=c(802:809), is_legendary=1)
legendary_pokemon[802:809,] <- new_legendary_pokemon

###Legendary status
legendary_dictionary <- legendary_pokemon
colnames(legendary_dictionary)[1] <- 'ndex'
pokemon2 <- left_join(pokemon2, legendary_dictionary, by='ndex')

################################################################################
###Giving every pokemon their moveset.
################################################################################

###Applying the function to each observation in 'moveset', and creating a new
###data frame that we will attach to the pokemon2 data frame. Also, we are
###stripping the 'Start - ' and 'L## - ' components.
four_moves <- apply(X=matrix(data=c(1:nrow(movesets)), nrow=nrow(movesets)),
                    FUN=final_four_moves,
                    MARGIN=1)
four_moves <- matrix(unlist(four_moves),
                     nrow=nrow(movesets),
                     ncol=4,
                     byrow=TRUE)
four_moves <- gsub(x=four_moves, pattern='L[0-9]{1,3} - ', replacement='')
four_moves <- gsub(x=four_moves, pattern='Start - ', replacement='')
colnames(four_moves) <- c('Move 1', 'Move 2', 'Move 3', 'Move 4')
four_moves <- data.frame(movesets[, c(1:3)], four_moves)
###Removes strange empty rows present from the moveset data.
four_moves <- four_moves[!is.na(four_moves$ndex) ,]

###Attaching the new moveset info to the pokemon2 data. Will create another 
###data frame instead to play it safe until we are sure we have what we want.
pokemon2 <- left_join(pokemon2, four_moves, by=c('ndex', 'species', 'forme'))

###There are certain rows that were missed by the join proces. Actually,
###the 'species' and 'forme' data does not match 1-to-1 between the 'pokemon'
### (or referred to as 'pokemon2' in this clean up process) and 'moveset' data. 
exceptions <- which(is.na(pokemon2$Move.1) &
                      is.na(pokemon2$Move.2) &
                      is.na(pokemon2$Move.3) & 
                      is.na(pokemon2$Move.4))

###A major source of trouble is naming inconsistencies and when the 'moveset'
###data only has one listing for pokemon with multiple forms in the 'pokemon2'
###data (likely due to the fact) the certain pokemon learn the same moves
###regardless of the difference forms. This for-loop attempts to find the prior
###mentioned exceptions.
for (i in exceptions){
  
  ###If the pokemon is only listed once (regardless of form) in the 'moveset'
  ###data, then the exception will be filled in with that info. Here, we
  ###assume the pokemon learns the same moves regardless of forms.
  if (sum(four_moves$ndex == pokemon2$ndex[i]) == 1){
    
    ID <- which(four_moves$ndex == pokemon2$ndex[i])
    
    pokemon2$Move.1[i] <- four_moves$Move.1[ID]
    pokemon2$Move.2[i] <- four_moves$Move.2[ID]
    pokemon2$Move.3[i] <- four_moves$Move.3[ID]
    pokemon2$Move.4[i] <- four_moves$Move.4[ID]
  }
}

###Still missed a few. This time, we'll filter out forms that are listed and
###learn the same final four moves by Lv.100.
four_moves_cleaned <- unique(four_moves[, c('ndex', 'species', 'Move.1',
                                            'Move.2', 'Move.3', 'Move.4')])

exceptions <- which(is.na(pokemon2$Move.1) &
                      is.na(pokemon2$Move.2) &
                      is.na(pokemon2$Move.3) & 
                      is.na(pokemon2$Move.4))

###The filting process is similar as before, but with duplicates filtered out.
for (i in exceptions){
  
  if (sum(four_moves_cleaned$ndex == pokemon2$ndex[i]) == 1){
    
    ID <- which(four_moves_cleaned$ndex == pokemon2$ndex[i])
    
    pokemon2$Move.1[i] <- four_moves_cleaned$Move.1[ID]
    pokemon2$Move.2[i] <- four_moves_cleaned$Move.2[ID]
    pokemon2$Move.3[i] <- four_moves_cleaned$Move.3[ID]
    pokemon2$Move.4[i] <- four_moves_cleaned$Move.4[ID]
  }
  
}

###Again, missed a few, but there are only three and we can simply.
###hard code those in
exceptions <- which(is.na(pokemon2$Move.1) &
                      is.na(pokemon2$Move.2) &
                      is.na(pokemon2$Move.3) & 
                      is.na(pokemon2$Move.4))

###Will fill in Floette (Red Flower) with moveset from the other.
###non-Eternal Floette's
pokemon2$Move.1[exceptions[1]] <- 'Misty Terrain'
pokemon2$Move.2[exceptions[1]] <- 'Moonblast'
pokemon2$Move.3[exceptions[1]] <- 'Petal Dance'
pokemon2$Move.4[exceptions[1]] <- 'Solar Beam'

###Will fill Aegislash with four arbitrarily chosen moves from one of its forms.
pokemon2$Move.1[exceptions[2]] <- 'Iron Defense'
pokemon2$Move.2[exceptions[2]] <- 'Slash'
pokemon2$Move.3[exceptions[2]] <- 'Head Smash'
pokemon2$Move.4[exceptions[2]] <- 'Night Slash'

###Will fill in Hoopa (Hoopa Confined) info with Hoopa's info
pokemon2$Move.1[exceptions[3]] <- 'Shadow Ball'
pokemon2$Move.2[exceptions[3]] <- 'Nasty Plot'
pokemon2$Move.3[exceptions[3]] <- 'Psychic'
pokemon2$Move.4[exceptions[3]] <- 'Hyperspace Hole'

################################################################################
###Taking care of Arceus
################################################################################

Arceues_dupes <- (pokemon2$species == 'Arceus' & pokemon2$type1 != 'Normal')
pokemon2 <- pokemon2[!Arceues_dupes ,]

################################################################################
###Simulating data, and outputting a training and test set.
################################################################################

training_set <- pokemon_simulator(800)
test_set <- pokemon_simulator(400)

write.csv(x=training_set, file='pokemon_training.csv', row.names=FALSE)
write.csv(x=test_set, file='pokemon_test.csv', row.names=FALSE)