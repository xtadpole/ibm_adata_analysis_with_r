# INSTALL TIDYMODELS
install.packages("rlang")
install.packages("tidymodels")

# load packages
library(tidymodels)
library(tidyverse)

# 1 DOWNLOAD DATA SET AND SETUP FOR USE
url <- "https://dax-cdn.cdn.appdomain.cloud/dax-noaa-weather-data-jfk-airport/1.1.4/noaa-weather-sample-data.tar.gz"
download.file(url, destfile = "noaa_weather_sample_data.tar.gz")

# untar the file
untar("noaa_weather_sample_data.tar.gz")

# 2 EXTRACT AND READ THE SAMPLE
noaa_sample <- read_csv("noaa-weather-sample-data/jfk_weather_sample.csv")
head(noaa_sample)

# use glimpse to see data types
glimpse(noaa_sample)

# 3 SELECT SUBSET OF COLUMNS
hourly_data <- select(noaa_sample,
                      HOURLYRelativeHumidity,
                      HOURLYDRYBULBTEMPF,
                      HOURLYPrecip,
                      HOURLYWindSpeed,
                      HOURLYStationPressure)

# show the first 10 rows
head(hourly_data, n = 10)

# 4 CLEAN COLUMNS
# use unique for HOURLY Precip
unique(hourly_data$HOURLYPrecip)

# replace T values with 0.0 and remove s values (also removed na values)
hourly_data$HOURLYPrecip <- str_remove(hourly_data$HOURLYPrecip, "s$")
hourly_data2 <- hourly_data %>%
  mutate(HOURLYPrecip = ifelse(HOURLYPrecip == "T", 0.00, HOURLYPrecip))
hourly_data2 <- na.omit(hourly_data2)
head(hourly_data2, n = 10)

# 5 CONVERT COLUMNS TO NUMERICAL TYPES
glimpse(hourly_data2)

# convert HOURLYPrecip to numeric
hourly_data3 <- hourly_data2 %>%
  mutate(HOURLYPrecip = as.numeric(HOURLYPrecip))
glimpse(hourly_data3)
summary(hourly_data3)

# 6 RENAME COLUMNS
hourly_data4 <- hourly_data3 %>%
  rename(relative_humidity = HOURLYRelativeHumidity,
         dry_bulb_temp_f = HOURLYDRYBULBTEMPF,
         precip = HOURLYPrecip,
         wind_speed = HOURLYWindSpeed,
         station_pressure = HOURLYStationPressure)
glimpse(hourly_data4)

# 7 EXPLORATORY DATA ANALYSIS
# set random seed data
set.seed(1234)
weather_split <- initial_split(hourly_data4, prop = 0.8)
train_data <- training(weather_split)
test_data <- testing(weather_split)
head(train_data)

# plot boxplots of training data only
ggplot(train_data, aes(x = relative_humidity, y = precip)) +
  geom_boxplot(fill = "blue", color = "red", alpha = 0.5) +
  geom_jitter(color = "black", alpha = 0.3)
ggplot(train_data, aes(x = dry_bulb_temp_f, y = precip)) +
  geom_boxplot(fill = "blue", color = "red", alpha = 0.5) +
  geom_jitter(color = "black", alpha = 0.3)
ggplot(train_data, aes(x = wind_speed, y = precip)) +
  geom_boxplot(fill = "blue", color = "red", alpha = 0.5) +
  geom_jitter(color = "black", alpha = 0.3)
ggplot(train_data, aes(x = station_pressure, y = precip)) +
  geom_boxplot(fill = "blue", color = "red", alpha = 0.5) +
  geom_jitter(color = "black", alpha = 0.3)

# 8 LINEAR REGRESSION
# create simple linear regression model and scatterplot for each response variable
relative_humidity_lm <- lm(precip ~ relative_humidity, data = train_data)
ggplot(relative_humidity_lm, mapping = aes(x = relative_humidity, y = precip)) +
  geom_point(fill = "blue", color = "red", alpha = 0.5) +
  geom_smooth(method = lm, color = "black")
dry_bulb_temp_f_lm <- lm(precip ~ dry_bulb_temp_f, data = train_data)
ggplot(dry_bulb_temp_f_lm, mapping = aes(x = dry_bulb_temp_f, y = precip)) +
  geom_point(fill = "blue", color = "red", alpha = 0.5) +
  geom_smooth(method = lm, color = "black")
wind_speed_lm <- lm(precip ~ wind_speed, data = train_data)
ggplot(wind_speed_lm, mapping = aes(x = wind_speed, y = precip)) +
  geom_point(fill = "blue", color = "red", alpha = 0.5) +
  geom_smooth(method = lm, color = "black")
station_pressure_lm <- lm(precip ~ station_pressure, data = train_data)
ggplot(station_pressure_lm, mapping = aes(x = station_pressure, y = precip)) +
  geom_point(fill = "blue", color = "red", alpha = 0.5) +
  geom_smooth(method = lm, color = "black")

# 9 IMPROVE THE MODEL
#ridge (L2) regularization
weather_recipe <-
  recipe(precip ~ ., data = train_data)
weather_ridge <- linear_reg(penalty = 0.1, mixture = 0) %>%
  set_engine("glmnet")
ridge_wf <- workflow() %>%
  add_recipe(weather_recipe)
ridge_fit <- ridge_wf %>%
  add_model(weather_ridge) %>%
  fit(data = train_data)
ridge_fit %>%
  pull_workflow_fit() %>%
  tidy()

#lasso (L1) regularization
weather_lasso <- linear_reg(penalty = 0.1, mixture = 1) %>%
  set_engine("glmnet")
lasso_wf <- workflow() %>%
  add_recipe(weather_recipe)
lasso_fit <- lasso_wf %>%
  add_model(weather_lasso) %>%
  fit(data = train_data)
lasso_fit %>%
  pull_workflow_fit() %>%
  tidy()

#elastic (L1, L2) regularization
weather_elastic <- linear_reg(penalty = 0.1, mixture = 0.3) %>%
  set_engine("glmnet")
elastic_wf <- workflow () %>%
  add_recipe(weather_recipe)
elastic_fit <- elastic_wf %>%
  add_model(weather_elastic) %>%
  fit(data = train_data)
elastic_fit %>%
  pull_workflow_fit() %>%
  tidy()

#10 FIND THE BEST MODEL
model_names <- c(relative_humidity_lm, dry_bulb_temp_f_lm, wind_speed_lm, station_pressure_lm)
train_error <- c(relative_humidity_lm, dry_bulb_temp_f_lm, wind_speed_lm, station_pressure_lm)
test_error <- c(relative_humidity_lm, dry_bulb_temp_f_lm, wind_speed_lm, station_pressure_lm)
comparison_df <- data.frame(model_names, train_error, test_error)
