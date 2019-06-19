###Simulator Functions

###Christian Rodriguez
###crodriguez0874@gmail.com
###06/15/19

###Summary: This script runs the functions needed to simulate and generate the
###pokemon data that we plan to use for our analysis. For each pokemon, we will
###have its pokedex number, species (name), forme, legendary status, gender,
###types, abilities, moveset, height, weight, nature, and
###base stats (IVs included). This script has a master function that generates
###the data, with sub-functions that each focus on one of the described
###characteristics of the pokemon. Since this will not be put into corporate
###production, we will not worry about unit testing for now.

################################################################################
###Simulator Sub-functions
################################################################################

###Chooses ability of pokemon
ability_generator <- function(pokemon_data){
  
  ###Initialize the vector.
  abilities <- c(0)
  
  for (i in 1:nrow(pokemon_data)){
    
    ###Make a set of possible abilities to sample from. Choose one randomly
    ###with equal probability
    ability_set <- c(pokemon_data$ability1[i],
                     pokemon_data$ability2[i],
                     pokemon_data$abilityH[i])
    abilities[i] <- sample(x=ability_set[!is.na(ability_set)], size=1)
    
  }
  
  return(abilities)
}

###Calculates HP of the pokemon
###Can do whole vectors at a time
hp_calculator <- function(Base_HP, IV_HP, EV_HP=0, level=100){
  
  ###Meta-equation for a pokemon's HP.
  HP <- (((2*Base_HP + IV_HP + (sqrt(EV_HP)/4))*level/100) + level + 10)
  return(HP)
  
}

###Determines IV's of pokemon
###Input twice - once for the legendaries and once for the non-legendaries
###Legendary pokemon are guranteed 3 perfect IVs (or 3 31 IVs)
IV_generator <- function(num_pokemon, legendary){
  
  if (legendary == FALSE){
    
    ###Can simply draw a number between 0-31 for each stat and for each
    ###Pokemon.
    IVs <- matrix(data=sample(x=c(0:31), size=num_pokemon*6, replace=TRUE),
                  nrow=num_pokemon, ncol=6, byrow=TRUE)
    return(IVs)
    
  } else {
    
    ###First start with a vector of NAs. This will be filled in with stats one
    ###Pokemon at a time.
    IVs <- rep(NA, 6*num_pokemon)
    
    ###This perfect IVs IDs vector will tell us which stats of each legendary
    ###Pokemon will be perfect.
    perfect_IVs_IDs <- c()
    
    ###We will choose which three of six IVs are perfect for each legendary
    ###Pokemon one at a time.
    for (i in 1:num_pokemon){
      
      ###Creates intervals of lengths of 6's to draw ID numbers from.
      start <- (1 + 6*(i-1))
      end <- (6*i)
      
      ###Draw three numbers from 6 consecutive numbers.
      ###If an ID number is drawn,
      ###then that mean there will be a 31 placed there.
      ###1-6 refers to the 6 stats (HP, Att, Def, Sp, Sp.Att, Sp.Def)
      ###of the first Pokemon
      ###If 3, 5, and 6 are drawn, then Defense, Sp.Att, and Sp. Def of the
      ###first pokemon will have perfect IVs.
      ###7-12 refers to the 6 stats (HP, Att, Def, Sp, Sp.Att, Sp.Def)
      ###of the second Pokemon that will have perfect IVs.
      ###If 7, 8, and 10 are drawn, then HP, Att, and Sp of the second
      ###pokemon will have perfect IVs.
      perfect_IVs_IDs <- c(perfect_IVs_IDs,
                           sample(size=3, replace=FALSE, x=c(start:end)))
      
    }
    
    ###Assigning the perfect IVs.
    IVs[perfect_IVs_IDs] <- 31
    
    ###Sample the non perfect IVs and assigning them.
    non_perfect_IVs <- sample(size=sum(is.na(IVs)), replace=TRUE, x=c(0:31))
    IVs[is.na(IVs)] <- non_perfect_IVs
    
    ###Turning the IV vector into a matrix.
    IVs <- matrix(nrow=num_pokemon, ncol=6, byrow=TRUE, data=IVs)
    
    return(IVs)
  }
}

###Calculates non-HP stats of the pokemon
###Can do whole matrices of stats at a time.
stats_calculator <- function(base_stats, IVs, EVs=0, level=100, N_modifier){
  
  ###Meta-equation of pokemon's non-hp stats.
  final_stats <- (((2*base_stats + IVs + (sqrt(EVs)/4))*level/100) + 5)
  final_stats <- final_stats*N_modifier 
  return(final_stats)
  
}

###Chooses gender of pokemon
gender_generator <- function(percent_male){
  
  if (is.na(percent_male)){
    
    ###Pokemon without a percent male entry are always genderless.
    return(NA)
    
  } else {
    
    ###Draw the gender of the pokemon via bernoulli trials.
    gender <- sample(x=c(0,1), size=1, prob=c(1-percent_male, percent_male))
    return(gender)
    
  }
}

###Generates the weight of the specified pokemon from a normal distribution
###with the mean as the listed weight and a standard deviation of 10% of the 
###listed weight.
weight_generator <- function(pokemon_data){
  
  ###Strip unwanted character strings and converts lbs into kg.
  weights <- gsub(x=pokemon_data$weight, pattern='.{5}$', replacement='')
  weights <- as.numeric(weights) * 0.453592
  
  ###For each pokemon, draw the pokemon's weight from a normal distribution.
  for (i in 1:nrow(pokemon_data)){
    
    ###Gurnatees there are no negative weights. If a negative weight is drawn,
    ###then we redraw the weight.
    weightt <- -1
    
    while(weightt < 0){
      
      weightt <- round(rnorm(n=1, mean=weights[i], sd=sqrt(0.10 * weights[i])),
                       digits=2)
      
    }
    
    weights[i] <- weightt
    
  }
  
  return(weights)
}

###Generates the height of the specified pokemon from a normal distribution
###with the mean as the listed height and a standard deviation of 10% of the 
###listed height.
height_generator <- function(pokemon_data){
  
  ###Extracts number of feet from the character strings.
  feet <- as.numeric(gsub(x=pokemon_data$height,
                          pattern='\\\'[0-9]{2}\"',
                          replacement=''))
  
  ###Extracts number of inches.
  inches <- gsub(x=pokemon_data$height, pattern='[0-9]*\'', replacement='')
  inches <- as.numeric(gsub(x=inches, pattern='\\\"', replacement=''))
  
  ###Calculates the pokemons' height in inches, and converts it into meters.
  height <- (12*feet + inches)*0.0254
  
  ###For each pokemon, draws their height from a normal distribution.
  for (i in 1:nrow(pokemon_data)){
    
    ###Gurantees we do not get negative height.
    heightt <- -1
    
    while (heightt < 0){
      
      heightt <- rnorm(n=1, mean=height[i], sd=sqrt(0.10*height[i]))
    }
    
    height[i] <- round(heightt, digits=2)
    
  }
  
  return(height)
}

###Makes the stats of the pokemon after considering base stats, IVs, and natures.
###Utilizes pokemon2 data.
finalStats_calculator <- function(stat_data, non_stat_data){
  
  nature_modifiers <- data.frame(nature=non_stat_data[, c('nature')])
  nature_modifiers <- left_join(nature_modifiers, natures, by=c('nature'))
  
  ###Initialize Stats
  hp <- rep(NA, nrow(stat_data))
  attack <- rep(NA, nrow(stat_data))
  defense <- rep(NA, nrow(stat_data))
  spattack <- rep(NA, nrow(stat_data))
  spdefense <- rep(NA, nrow(stat_data))
  speed <- rep(NA, nrow(stat_data))
  
  IVs <- cbind(non_stat_data$is_legendary, hp, attack, defense,
               spattack, spdefense, speed)
  
  ###Three Cases
  IVs[(IVs[, 1] == 0), c(2:7)] <- IV_generator(sum(IVs[, 1] == 0),
                                               legendary=FALSE)
  IVs[(IVs[, 1] == 1), c(2:7)] <- IV_generator(sum(IVs[, 1] == 1),
                                               legendary=TRUE)
  
  ###HP stats
  hp <- hp_calculator(Base_HP=stat_data[, 1], IV_HP=IVs[, 2])
  
  ###non_HP stats
  non_hp <- floor(stats_calculator(base_stats=stat_data[, 2:6],
                                   IVs=IVs[, 3:7],
                                   N_modifier=nature_modifiers[, 2:6]))
  
  ###Return Final Data Frame
  colnames(non_hp) <- c('attack', 'defense', 'spattack', 'spdefense', 'speed')
  resulting_stats <- cbind(hp, non_hp)
  
  return(resulting_stats)
  
}

###Generates the final four moves a pokemon should have at Lv.100
###Uses the 'moveset' data
final_four_moves <- function(observation_ID){
  
  ###Gets the last four moves from leveling up
  moves_column_IDs <- tail(grep(movesets[c(observation_ID) ,],
                                pattern='L[0-9]{1,3}'), n=4)
  ###Pokemon do not always learn 4 moves when leveling up to Lv.100
  if (length(moves_column_IDs) < 4) {
    
    ###We will use moves a Pokemon learned from the Start to fill in the moves.
    startmoves_column_IDs <- grep(movesets[c(observation_ID) ,],
                                  pattern='Start')
    
    if ((length(moves_column_IDs) + length(startmoves_column_IDs)) >= 4) {
      
      ###Figuring out how many moves to fill in after considering Lv. moves
      ###And choosing the Start moves at random
      moveslots_leftover <- (4-length(moves_column_IDs))
      startmoves_column_IDs <- sample(x=startmoves_column_IDs,
                                      replace=FALSE,
                                      size=moveslots_leftover)
      
      moves <- as.vector(movesets[observation_ID , c(moves_column_IDs,
                                                     startmoves_column_IDs)])
      
    } else {
      
      ###Even then, certain pokemon aren't able to learn at least four total Lv.
      ###and start moves.
      moveslots_leftover <- (4 - length(moves_column_IDs) - length(startmoves_column_IDs))
      
      moves <- as.vector(c(movesets[observation_ID, c(moves_column_IDs,
                                                      startmoves_column_IDs)],
                           rep(NA, moveslots_leftover)))
    }
    
  } else {
    
    ###The simple case when a pokemon can learn at least four moves
    ###by leveling up.
    moves <- as.vector(movesets[observation_ID, moves_column_IDs])
    
  }
  
  return(moves)
}

################################################################################
###Master Simulator Function
################################################################################

pokemon_simulator <- function(sample_size){
  
  ###Identity columns
  IDs <- sample(x=c(1:nrow(pokemon2)), size=sample_size, replace=TRUE)
  pokemon_identity <- pokemon2[IDs , c(2,3,4)]
  
  ###Legendary Status
  is_legendary <- pokemon2$is_legendary[IDs]
  
  ###Types
  type1 <- pokemon2$type1[IDs]
  type2 <- pokemon2$type2[IDs]
  
  ###Pokemons' Weight and Height
  weight_kg <- weight_generator(pokemon2[IDs ,])
  height_m <- height_generator(pokemon2[IDs ,])
  
  ###Ability
  ability <- ability_generator(pokemon2[IDs ,])
  
  ###Gender
  pokemon_maleRate <- pokemon2$`percent-male`[IDs]
  gender <- apply(FUN=gender_generator,
                  X=matrix(ncol=1, data=pokemon_maleRate),
                  MARGIN=1)
  
  ###Nature
  nature <- sample(natures$nature, size=nrow(pokemon2[IDs ,]), replace=TRUE)
  
  
  pokemon_data1 <- cbind(is_legendary, pokemon_identity, gender, type1, type2,
                         ability, weight_kg, height_m, nature)
  
  ###Base Stats
  pokemon_baseStats <- pokemon2[IDs, 10:15]
  
  ###Base Stats (IVs and Natures Included)
  resulting_stats <- finalStats_calculator(stat_data=pokemon_baseStats,
                                           non_stat_data=pokemon_data1)
  
  ###Pokemon Movesets
  moveset <- pokemon2[IDs, c('Move.1', 'Move.2', 'Move.3', 'Move.4')]
  
  ###Making final data frame and returning it.
  simulated_pokemon <- cbind(pokemon_data1, resulting_stats, moveset)
  
  return(simulated_pokemon)
  
}


