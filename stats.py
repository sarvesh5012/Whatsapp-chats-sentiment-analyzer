from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd 
from collections import Counter


#Object for Url extraction
extract = URLExtract()

#functions for stats fetching
def fetch_stats(selected_user, df):
    if selected_user != 'All Users':
        
        df = df[df['user'] == selected_user]#fetching number of messages
    number_of_messages = df.shape[0]
    
    number_of_words = []# fetching number of words
    for message in df['message']:
        number_of_words.extend(message.split())
        
    media_shared = df[df['message'] == '<Media omitted>\n'].shape[0]#Total media shared
    
    links = []# Total links shared
    for message in df['message']:
        links.extend(extract.find_urls(message))
    
    return number_of_messages, len(number_of_words), media_shared, len(links)


def Active_user(df):
    act_user = df['user'].value_counts().head()
    df = round((df['user'].value_counts()/df.shape[0])*100,2).reset_index().rename(columns={'index':'Name','user':'Percent'})
    return act_user, df


#function to create wordcloud

def cloud_of_words(selected_user,df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
        
    wcloud = WordCloud(width=350,height=250,min_font_size=8,background_color='white')
    df_wcloud = wcloud.generate(df['message'].str.cat(sep=' ')) 
    return df_wcloud


#Common words generally used in chat
def common_words(selected_user,df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
    
    rm_grp_noti = df[df['user'] != 'Group_Message']# Removing group notification
    rm_media_omi = rm_grp_noti[rm_grp_noti['message']!= '<Media omitted>\n']# removing <Media omitted>
    
    f = open('stopwords_hinglish.txt','r')
    stop_words = f.read()
    message_words = []
    for message in rm_media_omi['message']:
        for word in message.lower().split():
            if word not in stop_words:
                message_words.append(word)
                
    df_common = pd.DataFrame(Counter(message_words).most_common(20))
    return df_common

#Monthly messages from single user or Overall

def monthly_usage(selected_user,df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
        
    year_ana = df.groupby(['year','month_num','month']).count()['message'].reset_index()
    
    month_year = []
    for i in range(year_ana.shape[0]):
        month_year.append(year_ana['month'][i]+'-'+ str(year_ana['year'][i]))
    year_ana['Time'] = month_year
    return year_ana

# Daily usage by users
def daily_msgs(selected_user,df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
        
    daily_usage = df.groupby('daily').count()['message'].reset_index()
    return daily_usage

# Active users on a particular day in a week
def weekly_activity(selected_user,df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
        
    return df['day_name'].value_counts()

# Active users on a particular month

def month_avtivity(selected_user,df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
        
    return df['month'].value_counts()


#######SENTIMENT Analysis for the particular user or overall#####

def sentiment_func(selected_user,df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
        
    neg_total = df['neg'].sum()
    neu_total = df['neu'].sum()
    pos_total = df['pos'].sum()
    comp_total = df['compound'].sum()
    sizes = [neg_total, neu_total, pos_total, comp_total]
    
    return sizes

def sentiment_neg_pos(selected_user,df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
        
    scr_count = df['compare_score'].value_counts()
    pos_count = scr_count['pos']
    neg_count = scr_count['neg']
    
    return pos_count, neg_count
    

    
    
    
    
    

            
