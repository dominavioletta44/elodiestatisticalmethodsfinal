#---SETUP/CURSORY DATA---
library(dplyr)
library(stringr)
library(ggplot2)
library(tidyverse)
library(readxl)
library(lme4)
library(DHARMa)
library(autoimage)
setwd("/Users/User/OneDrive - University of Rochester/Documents/navajostuff/")
data <- read_excel('paradigms/full-joined-paradigms-wors-on-pc.xlsx')
write.csv(data, "paradigmscsv.csv")
data <- read.csv("paradigmscsv.csv")
k <- 5
set.seed(42)

cleandata <- data %>% #getting only the rows w/ classifier and stem 
  drop_na(Classifier, stem)
lesscleandata <- data %>% #only rows w/ classifiers, incl. those w/o ID'd stems
  drop_na(Classifier)
cleandata$fold <- sample(rep(1:k, length.out = nrow(cleandata))) #establishing fold indices

objpronouns <- c(" it", " me", " them", " him", " her", " us") #" the ", " a ") of course, this won't catch strings that have e.g. "meat" instead of "me"); also, the decision not to include strings with the/a was motivated by the fact that translations with 'the' seem to a) have the phrase 'into the fire' (where transitivity is already being specified by an 'it' or similar) or b) using English 'the' in a more generic sense than we would expect from transitive objects (like 'jump into the air', 'be on the warpath')
objpattern <- str_c(objpronouns, collapse = "|")#GPT helped with the syntax here
cleandata$hasobject=str_detect(cleandata$glose, objpattern) #giving each row a transitivity-boolean
lesscleandata$hasobject=str_detect(lesscleandata$glose, objpattern)

cleandata$purestem=str_sub(cleandata$stem, start = 2, end = -1) #the 'stem' column has the classifier "prefixed" to the stem in the original, so this is just chopping that off
cleandata$purestem <- factor(cleandata$purestem)
cleandata$Classifier <- factor(cleandata$Classifier)
cleandata$Classifier <- relevel(cleandata$Classifier, ref = "ø")

lesscleandata$purestem=str_sub(lesscleandata$stem, start = 2, end = -1) #same idea as above--it only occurred to me now that I could have just established all this for lesscleandata, then built cleandata off of that, but I suppose this is more hygienic anyways
lesscleandata$purestem <- factor(lesscleandata$purestem)
lesscleandata$Classifier <- factor(lesscleandata$Classifier)
lesscleandata$Classifier <- relevel(lesscleandata$Classifier, ref = "ø") 

table(cleandata$Classifier, cleandata$hasobject)
table(lesscleandata$Classifier, lesscleandata$hasobject) #getting an idea for information loss and general trends
length(unique(cleandata$purestem)) #for seeing how much overlap there is between rows
length(unique(cleandata$glose))
stemcounts <- table(cleandata$purestem)
rarestems <- names(stemcounts[stemcounts < 3]) #this bit was suggested by GPT after I got the errors from DHARMa about certain stem counts being too low for safe regression
#rarestems

#---ANALYSIS---

brierfolds1 <- c(0,0,0,0,0)
brierfolds2 <- c(0,0,0,0,0)
brierfolds3 <- c(0,0,0,0,0)

model1 <- glm(hasobject ~ Classifier, family = binomial, data = cleandata) #calling glm() with family = binomial gives a logistic regression
model2 <- glmer(hasobject ~ (1 | purestem), family = binomial, data = cleandata) #glmer() just allows for random effects modelling: "1 | stem" just means "a baseline intercept adjusted for all stems"
model3 <- glmer(hasobject ~ Classifier + (1 | purestem), family = binomial, data = cleandata)

for (fold in c(1:k)){ #for all folds
  testvals <- (filter(cleandata[cleandata$fold == fold, ])) #get and model values for train and test folds
  trainvals <- (filter(cleandata[cleandata$fold != fold, ]))
  trainvals$purestem <- factor(trainvals$purestem)
  testvals$purestem  <- factor(testvals$purestem)
  mod1 <- glm(hasobject ~ Classifier, data = trainvals, family = binomial)
  mod2 <- glmer(hasobject ~ (1 | purestem), data = trainvals, family = binomial)
  mod3 <- glmer(hasobject ~ Classifier + (1 | purestem), data = trainvals, family = binomial)
  pred1 <- predict(mod1, newdata = testvals, type = "response") #get prediction values for all models
  pred2 <- predict(mod2, newdata = testvals, type = "response", allow.new.levels = TRUE) #allow.new.levels just means that if there's a stem that's in the test data that wasn't in the training data, that stem's effect is set to zero instead of throwing an error
  pred3 <- predict(mod3, newdata = testvals, type = "response", allow.new.levels = TRUE)
  actuals <- testvals$hasobject
  brierfolds1[fold] <- mean((pred1 - actuals)^2) #getting "MSE's" for each fold and model
  brierfolds2[fold] <- mean((pred2 - actuals)^2)
  brierfolds3[fold] <- mean((pred3 - actuals)^2)
  }

CV1 <- mean(brierfolds1)
CV2 <- mean(brierfolds2)
CV3 <- mean(brierfolds3)

cat("CVE 1:", CV1, "  \nCVE 2:", CV2, "\nCVE 3:", CV3)

foldnum <- c(1:5) #putting the "MSE" values into a dataframe for easier plotting
folddf <- data.frame(foldnum, brierfolds1, brierfolds2, brierfolds3)
folddf_long <- pivot_longer(
  folddf,
  cols = c(brierfolds1, brierfolds2, brierfolds3),
  names_to = "Model",
  values_to = "MSE")
ggplot() +
  geom_bar(data = folddf_long, aes(x = factor(foldnum), y = MSE, fill = Model), stat = "identity", position = "dodge") + #the conversion of folds into factors was also a GPTism
  labs(x = "Fold", y = "Brier Scores", title = "Cross-validated Brier scores per fold by model") +
  scale_fill_discrete(name="Model", labels=c("1 (classifiers only)", "2 (stems only)", "3 (classifiers + stems)")) 

diff12 <- brierfolds1 - brierfolds2 #t-tests for cross-validation: all three models significantly differ in this respect (yay), and the one with both classifier and stem performs the best (double yay)
diff13 <- brierfolds1 - brierfolds3
diff23 <- brierfolds2 - brierfolds3
t.test(diff12)
t.test(diff13)
t.test(diff23)
summary(model1)
summary(model2)
summary(model3)
confint(model1)
confint(model2)
confint(model3)

#---PLOTTING---

par(mfrow = c(2,2))
simres3 <- simulateResiduals(fittedModel = model3, n = 1000) #as a sanity check, seeing if model3 has the properties we expect (no huge outliers, no zero dispersion, etc.); using DHARMa was suggested to me by GPT after I prompted it for better ways to plot a logistic regression--the actual implementation comes from it and the DHARMa documentation
plot(simres3)
testCategorical(simres3, catPred = cleandata$Classifier)
testDispersion(simres3)
title(sub = "= 0.142")
testZeroInflation(simres3)
title(sub = "= 0.036")
testOutliers(simres3)
model3data <- model.frame(model3)
idx <- model3data$Classifier == "ø"
simnull <- recalculateResiduals(simres3, sel = idx)#the call to model.frame in order to get these plots is from ChatGPT--the prompt was simply to extract the null classifier for the relevant plotting
plot(simnull)
plot(model3)
plot(model2)
plot(model1)
reset.par()
