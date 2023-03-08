import re
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Defining object sid
sid = SentimentIntensityAnalyzer()

# Function for data manipulation and Analysis
def Analyse_Data(chatdata):
   dataPattern = '\d{1,2}\/\d{2,4}\/\d{2,4},\s\d{1,2}:\d{1,2}\s\w{1,2}\s-\s'
   messages = re.split(dataPattern, chatdata)[1:]
   dates = re.findall(dataPattern, chatdata)
   df = pd.DataFrame({'user_message': messages, 'message_date': dates})

   users = []
   messages = []
   for message in df['user_message']:
      entry = re.split('([\w\W]+?):\s', message)
      if entry[1:]:
         users.append(entry[1])
         messages.append(entry[2])

      else:
         users.append('Group_Message')
         messages.append(entry[0])

   df['user'] = users
   df['message'] = messages
   df.drop(columns=['user_message'], inplace=True)

   df['formated_time'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %I:%M %p - ')

   df['year'] = df['formated_time'].dt.year
   df['month'] = df['formated_time'].dt.month_name()
   df['day'] = df['formated_time'].dt.day
   df['hour'] = df['formated_time'].dt.hour
   df['min'] = df['formated_time'].dt.minute
   df['month_num'] = df['formated_time'].dt.month
   df['daily'] = df['formated_time'].dt.date
   df['day_name'] = df['formated_time'].dt.day_name()
   
   df['scores'] = df['message'].apply(lambda message: sid.polarity_scores(message))
   df['neg'] = df['scores'].apply(lambda score_dict: score_dict['neg'])
   df['neu'] = df['scores'].apply(lambda score_dict: score_dict['neu'])
   df['pos'] = df['scores'].apply(lambda score_dict: score_dict['pos'])
   df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
   df['compare_score'] = df['compound'].apply(lambda c: 'pos' if c>=0 else 'neg')


   return df
