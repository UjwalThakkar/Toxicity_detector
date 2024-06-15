import pandas as pd
import sys

from test import toxicity_classifier
text = sys.argv[1]
# text = input('Enter Text: ')  
model = toxicity_classifier()
results = model.predict(text)

df = pd.DataFrame.from_dict(results, orient='index')
print(df)
sys.stdout.flush()
