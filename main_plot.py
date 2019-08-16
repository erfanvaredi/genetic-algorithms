import pandas as pd
import numpy as np
import utils as EGAUtils
import constants as EConstants
import matplotlib.pyplot as plt


# 'M%', 'P%', 'LA%', 'D', 'POP_SIZE', 'AAA%', 'AA%', 'A%', 'BBB%', 'BB%', 'ACCEPTED_CUSTOMERS','GENERATION_SIZE'
df_results = pd.read_csv('results.csv', names=EConstants.get_cols_result())

df_results['M%'] = df_results['M%'].apply(lambda x: int(x*100))
df_results['P%'] = df_results['P%'].apply(lambda x: int(x*100))
df_results['LA%'] = df_results['LA%'].apply(lambda x: int(x*100))
df_results['AAA%'] = df_results['AAA%'].apply(lambda x: int(x*100))
df_results['AA%'] = df_results['AA%'].apply(lambda x: int(x*100))
df_results['A%'] = df_results['A%'].apply(lambda x: int(x*100))
df_results['BB%'] = df_results['BB%'].apply(lambda x: int(x*100))
df_results['BBB%'] = df_results['BBB%'].apply(lambda x: int(x*100))
# df_results['BBB%'] = df_results['BBB%'].apply(lambda x: int(x*100))


print(df_results[['M%', 'P%', 'LA%', 'ACCEPTED_CUSTOMERS']])
print(df_results[['AAA%', 'AA%', 'A%', 'BBB%', 'BB%', 'ACCEPTED_CUSTOMERS']])

# df_results[['M%', 'P%', 'A%']].plot()
# plt.legend()
# plt.show()