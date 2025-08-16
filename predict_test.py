import pandas as pd
from src.utils  import load_obj


transformer = load_obj("artifacts/preprocessor.pkl")
model = load_obj("artifacts/model.pkl")


## creating data frame
new_data = pd.DataFrame({
    'gender':['Male'],
    'race_ethnicity':['group C'],
    'parental_level_of_education':['high school'],
    'lunch':['standard'],
    'test_preparation_course':['none'],
    'reading_score':[70],
    'writing_score':[80]
})


new_data_transformer = transformer.transform(new_data)
predict = model.predict(new_data_transformer)

print(f"Predicted score: {predict[0]:.2f}")
