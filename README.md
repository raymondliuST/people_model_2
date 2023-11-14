# People Model
people model is a transformer encoder based model that predicts the masked attribute of user data

## Model Choice
Simplified Bert Model

## Example:
1. Each user has attributes -> [BrowserFamily, DeviceType, os, Country] 
2. Each Token has 80% chance to be masked
3. Predict the missing token