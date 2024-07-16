import os
import sys
from datetime import datetime as d_time
import random
import pred_config

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class FeatureGenerator(object):
  #date_format = "%Y-%m-%d %H:%M:%S"
  date_format = "%Y.%m.%d %H:%M"

  def train_test_split(self, df, train_ratio, test_ratio):
    caseid = list(set(df['id']))
    num_train = int(len(caseid)*train_ratio)
    num_test = len(caseid)*test_ratio
    train_caseid = list(random.sample(caseid, num_train))
    test_caseid = [x for x in caseid if x not in train_caseid]
    train = df.loc[df['id'].isin(train_caseid)]
    test = df.loc[df['id'].isin(test_caseid)]
    return train, test

  def create_initial_log(self, path,config):
      df = self.read_into_panda_from_csv(path,config)
      return df

  def add_dur(self, df):
    def to_minute(x):
      return int(x.total_seconds() / 60)
    df['dur'] = (df['complete_timestamp'] - df['start_timestamp']).apply(to_minute)
    return df

  def read_into_panda_from_csv(self, path, config, sep=','):
    df_log = pd.read_csv(filepath_or_buffer=path, header=0, sep=sep)#, index_col=0)
    columns = list()
    rename_columns = list()
    columns.append("case:concept:name")
    rename_columns.append("id")
    if config["control_flow_p"]==True:
      columns.append("concept:name")
      rename_columns.append("activity")
    if config["resource_p"]==True:
      columns.append("org:resource")
      rename_columns.append("resource")
    if config["time_p"]==True:
      if config["transition"]==True:
        columns.append("StartTimestamp")
        rename_columns.append("start_timestamp")
        columns.append("time:timestamp")
        rename_columns.append("complete_timestamp")
      else:
        columns.append("time:timestamp")
        rename_columns.append("complete_timestamp")
    if config["data_p"]==True:
      pass
    df_log = df_log[columns]
    # rename columns:
    df_log.columns = rename_columns
    for col in df_log.columns:
      if "timestamp" in col:
          df_log[col] = pd.to_datetime(df_log[col])
    if config["data_p"]==True:
      df_log = df_log.sort_values(['id', 'complete_timestamp'], ascending=True)
      df_log = df_log.reset_index(drop=True)
    return df_log

  def add_next_activity(self, df):
    df['next_activity'] = ''
    df['next_time'] = 0
    num_rows = len(df)
    for i in range(0, num_rows - 1):
      #print(str(i) + ' out of ' + str(num_rows))

      if df.at[i, 'id'] == df.at[i + 1, 'id']:
        df.at[i, 'next_activity'] = df.at[i + 1, 'activity']
        if(pred_config.load().time_p):
          df.at[i, 'next_time'] = df.at[i + 1, 'complete_timestamp']
      else:
        df.at[i, 'next_activity'] = '!'
        if(pred_config.load().time_p):
          df.at[i, 'next_time'] = df.at[i, 'complete_timestamp']
    df.at[num_rows-1, 'next_activity'] = '!'
    if(pred_config.load().time_p):
      df.at[num_rows-1, 'next_time'] = df.at[num_rows-1, 'complete_timestamp']

    return df

  def add_next_dur(self,df):
    df['next_dur'] = 0
    num_rows = len(df)
    for i in range(0, num_rows - 1):
        #print(str(i) + ' out of ' + str(num_rows))
        if df.at[i, 'id'] == df.at[i + 1, 'id']:
            df.at[i, 'next_dur'] = int((df.at[i + 1, 'complete_timestamp'] - df.at[i+1, 'start_timestamp']).total_seconds() / 60 )
        else:
            df.at[i, 'next_dur'] = 0
    df.at[num_rows-1, 'next_dur'] = 0

    return df


  def add_next_resource(self, df):
      df['next_resource'] = ''
      num_rows = len(df)
      for i in range(0, num_rows - 1):
          #print(str(i) + ' out of ' + str(num_rows))

          if df.at[i, 'id'] == df.at[i + 1, 'id']:
              df.at[i, 'next_resource'] = df.at[i + 1, 'activity']
          else:
              df.at[i, 'next_resource'] = '!'
      df.at[num_rows-1, 'next_resource'] = '!'

      return df

  # def encode_sequence(df):
  def write_pandas_to_csv(self, df, version, out):
      # df = df.reset_index(drop=True)
      filename = ''
      if out == False:
          filename = 'training-data' + str(version) + '.csv'
          df.to_csv(filename,sep=',')
      else:
          filename = 'Results/' + str(version) + '.csv'
          df.to_csv('Results/' + str(version) + '.csv', sep=',')
      return filename

  
  def add_activity_history(self, df):
    df['activity_history'] = ""
    ids = []
    num_rows = len(df)
    prefix = str(df.at[0, 'activity'])
    df.at[0, 'activity_history'] = prefix

    for i in range(1, num_rows):
      if df.at[i, 'id'] == df.at[i - 1, 'id']:
        prefix = prefix + '+' + str(df.at[i, 'activity'])
        df.at[i, 'activity_history'] = prefix
      else:
        ids.append(df.at[i - 1, 'id'])
        prefix = str(df.at[i, 'activity'])
        df.at[i, 'activity_history'] = prefix
    return df

  def add_resource_history(self, df):
    df['resource_history'] = ""
    ids = []
    num_rows = len(df)
    res_prefix = str(df.at[0, 'resource'])
    df.at[0, 'resource_history'] = res_prefix

    for i in range(1, num_rows):
      if df.at[i, 'id'] == df.at[i - 1, 'id']:
        res_prefix = res_prefix + '+' + str(df.at[i, 'resource'])
        df.at[i, 'resource_history'] = res_prefix
      else:
        ids.append(df.at[i - 1, 'id'])
        res_prefix = str(df.at[i, 'resource'])
        df.at[i, 'resource_history'] = res_prefix
    return df


  def add_query_remaining(self, df):
    df['elapsed_time'] = 0
    df['total_time'] = 0
    df['remaining_time'] = 0
    ids = []
    total_Times = []
    num_rows = len(df)
    temp_elapsed = 0

    for i in range(1, num_rows):
      if df.at[i, 'id'] == df.at[i - 1, 'id']:
        sojourn_time = df.at[i - 1, 'next_time'] - df.at[i - 1, 'complete_timestamp']
        temp_elapsed += sojourn_time.total_seconds()
        df.at[i, 'elapsed_time'] = temp_elapsed
      else:
        ids.append(df.at[i - 1, 'id'])
        total_Times.append(temp_elapsed)
        temp_elapsed = 0

    ids.append(df.at[num_rows - 1, 'id'])
    total_Times.append(df.at[num_rows - 1, 'elapsed_time'])
    df.at[num_rows-1,'elapsed_time'] = temp_elapsed
    for i in range(0, num_rows):
        try:
            ind = ids.index(df.at[i, 'id'])
            total_ = total_Times[ind]
            df.at[i, 'total_time'] = total_
            df.at[i, 'remaining_time'] = total_ - df.at[i, 'elapsed_time']
        except ValueError:
            return ValueError
    return df

  def read_from_query(self, path):
      df = pd.read_csv(filepath_or_buffer=path, header=0, index_col=0)  # ,nrows = 1000)
      # List = range(0,len(df))
      # df = df.ix[List]

      return df

  def queue_level(self, df):
      #df = self.read_from_query(path_query)
      df = df.reset_index(drop=True)
      activity_list = self.get_activities(df)
      df = self.add_queues(df, activity_list)
      #version = "V_events_3"+name
      #filename = self.write_pandas_to_csv(df, version, False)
      return df

  def get_activities(self, df):
    """
    activity_list = []
    for i in range(0, len(df)):
      pair = df.at[i, 'activity']
      try:
        ind = activity_list.index(pair)
      except ValueError:
        activity_list.append(pair)
    return sorted(activity_list)
    """
    return sorted(list(set(df['activity'])))

  def update_event_queue(self, event_queue, cur_time):
      remove_indices = []
      rem_ind = []

      # going over the different activities and getting the rates
      for i, e in enumerate(event_queue):
          for j, q in enumerate(event_queue[i]):
              if q[1] <= cur_time:
                  rem_ind.append(j)
          remove_indices.append(rem_ind)

          # print 'count remove: ' + str(count_remove)
          count_remove = 0
          if len(remove_indices[i]) > 0:
              for index in sorted(remove_indices[i], reverse=True):
                  del event_queue[i][index]
          rem_ind = []
      return

  def add_queues(self, df, activity_list):
    event_queue = []
    tuple = []
    df['total_q'] = 0

    for s in activity_list:
      col_name = 'queue' + '_' + str(s)
      df[col_name] = 0
      event_queue.append(tuple)
      tuple = []

    num_rows = len(df)
    for i in range(0, num_rows):
      # print (str(i) + ' queueing calculation')
      cur_time = df.at[i, 'complete_timestamp']
      next_time = df.at[i, 'next_time']
      cur_activity = df.at[i, 'activity']
      ind = activity_list.index(cur_activity)
      tuple = [cur_time, next_time]
      event_queue[ind].append(tuple)
      self.update_event_queue(event_queue, cur_time)


      total_q = 0
      for j, s in enumerate(activity_list):
        col_name1 = 'queue' + '_' + str(s)
        ind = activity_list.index(s)
        x = self.find_q_len_ttiq(event_queue[ind], cur_time)
        df.at[i, col_name1] = x
        total_q += x
      df.at[i,'total_q'] = total_q

    return df

  def find_q_len_ttiq(self, event_queue, cur_time):
      q_len = len(event_queue)
      return q_len

  def save_act_and_res(self, df, checkpoint, data_set):
    activities = sorted(list(set(df['activity'])))
    resources = sorted(list(set(df['resource'])))
    with open("%s%s_activities" % (checkpoint, data_set)) as f:
      pickle.dump(activities, f)
    with open("%s%s_resources" % (checkpoint, data_set)) as f:
      pickle.dump(resources, f)

  # def one_hot_encode_history(self, df, dict_dir):
  #   activities = sorted(list(set(df['activity'])))
  #   activities.append('!')
  #   with open("%s_activities.pkl" % (dict_dir), 'wb') as f:
  #     pickle.dump(activities, f)
  #   num_act = len(activities)

  #   # resources = sorted(list(set(df['resource'])))
  #   # with open("%s_resources.pkl" % (dict_dir), 'wb') as f:
  #   #   pickle.dump(resources, f)
  #   # num_res = len(resources)

  #   # define a mapping of chars to integers
  #   act_char_to_int = dict((str(c), i) for i, c in enumerate(activities))
  #   act_int_to_char = dict((i, c) for i, c in enumerate(activities))

  #   # res_char_to_int = dict((str(c), i) for i, c in enumerate(resources))
  #   # res_int_to_char = dict((i, c) for i, c in enumerate(resources))
  #   print("act_char_to_int: {}".format(act_char_to_int))
  #   print("act_int_to_char: {}".format(act_int_to_char))
  #   # print("res_char_to_int: {}".format(res_char_to_int))
  #   # print("res_int_to_char: {}".format(res_int_to_char))
  #   """
  #   with open("%s%s_act_char_to_int") as f:
  #     pickle.dump(act_char_to_int, f)
  #   with open("%s%s_act_int_to_char") as f:
  #     pickle.dump(act_int_to_char, f)

  #   with open("%s%s_res_char_to_int") as f:
  #     pickle.dump(res_char_to_int, f)
  #   with open("%s%s_res_int_to_char") as f:
  #     pickle.dump(res_int_to_char, f)
  #   """
  #   # integer encode input data
  #   X_train = list()
  #   y_a = list()
  #   y_t = list()
  #   df['history'] = df['history'].astype(str)
  #   # df['resource_history'] = df['resource_history'].astype(str)
  #   maxlen = max([len(str(x).split('_')) for x in df['history']])

  #   for i in range(0, len(df)):
  #     if str(df.at[i, 'history']) != "nan":
  #       parsed_hist = str(df.at[i, 'history']).split("_")
  #       # parsed_res_hist = str(df.at[i, 'resource_history']).split("_")

  #       int_encoded_act = [act_char_to_int[act] for act in parsed_hist]
  #       # int_encoded_res = [res_char_to_int[res] for res in parsed_res_hist]

  #       # one hot encode X
  #       onehot_encoded_X = list()
  #       for act_int, res_int in zip(int_encoded_act, int_encoded_res):
  #         onehot_encoded_act = [0 for _ in range(num_act)]
  #         onehot_encoded_act[act_int] = 1
  #         onehot_encoded_res = [0 for _ in range(num_res)]
  #         onehot_encoded_res[res_int] = 1
  #         onehot_encoded = onehot_encoded_act + onehot_encoded_res
  #         onehot_encoded_X.append(onehot_encoded)
  #       #zero-pad
  #       while len(onehot_encoded_X) != maxlen:
  #           onehot_encoded_X.insert(0, [0]*(num_act+num_res))
  #       if len(onehot_encoded_X) > maxlen:
  #           print(onehot_encoded_X)
  #       X_train.append(onehot_encoded_X)

  #       # one hot encode y
  #       next_act = str(df.at[i, 'next_activity'])
  #       current_duration = df.at[i, 'dur']
  #       int_encoded_next_act = act_char_to_int[next_act]
  #       onehot_encoded_next_act = [0 for _ in range(num_act)]
  #       onehot_encoded_next_act[int_encoded_next_act] = 1
  #       y_a.append(onehot_encoded_next_act)
  #       y_t.append(current_duration)

  #   X_train = np.asarray(X_train)
  #   y_a = np.asarray(y_a)
  #   y_t = np.asarray(y_t)
  #   print(X_train.shape, y_a.shape, y_t.shape)
  #   return X_train, y_a, y_t

  def generate_context_feature(self, df, activity_list):
      cols=list()
      for k,s in enumerate(activity_list):
        cols.append('queue'+'_'+str(s))
      df_numerical = df[cols]
      context_X = df_numerical.values.tolist()
      context_X = np.asarray(context_X)
      return context_X

if __name__ == '__main__':
  #level = req['level']
  level = 'Level1'
  #filename = req['name']
  filename = 'Production.xes'

  name = filename
  #filename = 'logdata/'+filename
  #l_views.prep_data(filename)
  filename = '../data/'+filename+'.csv'


  # encode the file -- level 0
  FG = FeatureGenerator()
  level0_file = FG.create_initial_log(filename ,name)

  #make the predictions
  df = {}
  activity_list = {}
  query_name = 'remaining_time'
  # # encode the file -- level 1 and 2
  # level1_2_file = FG.queue_level(level0_file_ordered, name)
  # df = FG.read_from_query(level1_2_file)
  activity_list = FG.get_activities(df)

  #FG.one_hot_encode_history(df)
  FG.generate_context_feature(df,activity_list)
  #train_df, test_df, train_list, test_list = FG.prep_data(df, activity_list, query_name, level)
  #print(train_df.col)

