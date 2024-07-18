# added comments to what is performed
import xml.etree.ElementTree as ET
import pm4py
import traceback
import pickle
import os.path
from Relevance_based_Sampling.SamplingAlgorithms import FeatureGuidedLogSampler, SequenceGuidedLogSampler
import pred_config
from core.feature_encoder import FeatureEncoder
from core.feature_generator import FeatureGenerator
from core.model import net
import time
import numpy as np
import csv
import random
from array import array
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, train_test_split

if __name__ == '__main__':
    args = pred_config.load()

    contextual_info = args.contextual_info
    status = args.status
    task = args.task
    control_flow_p = args.control_flow_p
    time_p = args.time_p
    resource_p = args.resource_p
    data_p = args.data_p
    transition = args.transition

    data_dir = args.data_dir
    data_set = args.data_set
    
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    checkpoint_dir = args.checkpoint_dir
    
    cross_number = args.cross_number

    exp_name = '%s_%s_%s_%s_%s_%s_%s_%s_LSTM' % (data_set, task, control_flow_p, time_p, resource_p, data_p, num_epochs, batch_size)

    load_saved_data = args.load_saved_data
    load_saved_test_data = args.load_saved_test_data
    save_data = args.save_data

    total_results_file = args.result_dir + exp_name + '.csv'
    file_exists = os.path.isfile(total_results_file)

    sampling_technique = args.sampling_technique
    sample_size = args.sample_size
    sample_size = float(sample_size)

    # create or open file to store the results
    with open (total_results_file, 'a') as csvfile:
        headers = ['Dataset', 'Sampling Method', 'Sample size' 'Accuracy', 'Example Leakage', 'Running_Time', 'F1_Score', 'Precisions', 'Recalls', 'FeatureEncoder_Time',
              'TruePositive', 'FalsePositive', 'FalseNegative', 'TrueNegative']
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
        print("File created")  
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

    # get data set
    filename = data_dir + data_set + ".csv" 
    model_name = '%s_%s_%s_%s_%s_%s_%s_%s' % (data_set, task, control_flow_p, time_p, resource_p, data_p, num_epochs,
                batch_size)

    feature_name = '%s_%s_%s_%s_%s_%s' % (
                    data_set, task, control_flow_p, time_p, resource_p, data_p)

    log_config = {"control_flow_p": control_flow_p, "time_p": time_p, "resource_p": resource_p, "data_p": data_p, 'transition': transition}

    # feature generation and split data set
    print("flag: loading training data")
    Start_FeatureGenerator = time.time()
    fg = FeatureGenerator()
    df = fg.create_initial_log(filename, log_config)

    #SAMPLING + splitting
    print("Sampling and splitting data in train/test set")
    print("Sampling: ", sampling_technique)
    
    #Random Sampling (own implementation)
    if(sampling_technique=="random"):
        list_ids = list(df['id'].unique())
        list_ids = [int(id) for id in list_ids]
        list_ids = array("i", list_ids)
        sample_size = int(len(list_ids)*sample_size)
        sampled_ids = random.sample(list_ids, sample_size)

        # train/test split
        train_ids_size = int(len(sampled_ids)*0.8)
        train_ids = random.sample(sampled_ids, train_ids_size)
        test_ids= []
        for id in sampled_ids:
            if (id not in train_ids):
                test_ids.append(id)
        train_df =  df[df['id'].isin(train_ids)]
        test_df = df[df['id'].isin(test_ids)]
    

    # Relevance-based sampling (based on the implemention of Kabierski et al., 2020)
    elif(sampling_technique=="relevance"):
        xes_file_path = data_dir + data_set + ".xes"
        xes = pm4py.read_xes(xes_file_path) 

        xes = pm4py.convert_to_event_log(xes, case_id_key="case:concept:name")
        sample_size = int(len(xes) * 0.1)

        petrinet, initial_marking, final_marking = pm4py.read_pnml(data_dir + data_set + ".pnml")
        sampler = SequenceGuidedLogSampler(xes, batch_size=1)
        sample = sampler.construct_sample(xes, petrinet, initial_marking, final_marking, sample_size)
        ids = [trace.attributes['concept:name'] for trace in sample]
        ids = [int(id) for id in ids]
        relevance_df = df[df['id'].isin(ids)]
        
        # train/test split
        sampled_ids = list(relevance_df['id'].unique())
        sampled_ids = [int(id) for id in sampled_ids]
        sampled_ids = array("i", sampled_ids)
        train_ids_size = int(len(sampled_ids)*0.8)
        train_ids = random.sample(sampled_ids, train_ids_size)
        test_ids= []
        for id in sampled_ids:
            if (id not in train_ids):
                test_ids.append(id)
        train_df =  df[df['id'].isin(train_ids)]
        test_df = df[df['id'].isin(test_ids)]


    # Variant-based sampling:
    # The variant-based sampling was done using a plug-in in ProM, 
    # this code simply performs a train/test split of the sampled event log 
    elif(sampling_technique=="variant"):
        sample_size = int(float(sample_size)*100)
        xes_file_path = data_dir + data_set + "_variant_" + str(sample_size) + ".xes"
        print(xes_file_path)
        xes = pm4py.read_xes(xes_file_path) 
        
        variant_df = pm4py.convert_to_dataframe(xes)
        
        sampled_ids = list(variant_df['case:concept:name'].unique())
        sampled_ids = [int(id) for id in sampled_ids]
        sampled_ids = array("i", sampled_ids)
        train_ids_size = int(len(sampled_ids)*0.8)
        train_ids = random.sample(sampled_ids, train_ids_size)
        test_ids= []
        for id in sampled_ids:
            if (id not in train_ids):
                test_ids.append(id)
        train_df =  df[df['id'].isin(train_ids)]
        test_df = df[df['id'].isin(test_ids)]

    # No samping, just train-test split (80/20)
    else: 
        #split
        list_ids = list(df['id'].unique())
        train_ids_size = int(len(list_ids)*0.8)
        train_ids = random.sample(list_ids, train_ids_size)
        train_df = df[df['id'].isin(train_ids)]
        test_df = df[~df['id'].isin(train_ids)]

    df[df['id'].isin(sampled_ids)].to_csv(data_set + sampling_technique + str(sample_size) + 'sample.csv', index=False)
   

    columns = ['concept:name', 'concept:name']
    if(args.time_p):
        columns = ['concept:name', 'concept:name', 'time:timestamp']
    header_row = pd.DataFrame([columns], columns=test_df.columns)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)    

    print("Train + test set: ")
    print(train_df.shape)
    print(test_df.shape)

    FeatureGenerator_Time = time.time() - Start_FeatureGenerator
    print("FeatureGenerator done")

    num_events = len(df)
    num_cases = len(set(df["id"]))

    print("flag: generating features")
    if task == 'next_activity':
        loss = 'categorical_crossentropy'
        regression = False
        feature_type_list = ["activity_history"]
        train_df = fg.add_activity_history(train_df)
        train_df = fg.add_next_activity(train_df)

    # training set generation
    print("flag: encoding features")
    Start_FeatureEncoder = time.time()
    fe = FeatureEncoder()
    train_X, train_y = fe.original_one_hot_encode(train_df, feature_type_list, task, feature_name)
    FeatureEncoder_Time = time.time() - Start_FeatureEncoder
    print("done")

    #if contextual_info:
        # train_df = fg.queue_level(train_df)
        # activity_list = fg.get_activities(train_df)
        # train_context_X = fg.generate_context_feature(train_df, activity_list)
        # model = net()
        # if task == 'next_timestamp':
        #     model.train(train_X, train_y, regression, loss, n_epochs=num_epochs, batch_size=batch_size,
        #                 model_name=model_name, checkpoint_dir=checkpoint_dir,
        #                 X_train_ctx=train_context_X)
        # elif task == 'next_activity':
        #     model.train(train_X, train_y, regression, loss, n_epochs=num_epochs, batch_size=batch_size,
        #                 model_name=model_name, checkpoint_dir=checkpoint_dir,
        #                 X_train_ctx=train_context_X)
    #else:

    train_context_X = None
    model = net()

    if task == 'next_activity':
        model.train(train_X, train_y, regression, loss, n_epochs=num_epochs, batch_size=batch_size,
                    model_name=model_name, checkpoint_dir=checkpoint_dir,
                    context=contextual_info)

    Running_Time = model.running_time
    print("training is done")
    model.load(checkpoint_dir, model_name=model_name)

    ## loading test data
    print("testing")
    fg = FeatureGenerator()
    num_events = len(df)
    num_cases = len(set(df["id"]))

    # feature generation
    print("flag: generating features")
    if task == 'next_activity':
        test_df = fg.add_activity_history(test_df)
        test_df = fg.add_next_activity(test_df)
    print("done")

    # test set generation
    print("flag: encoding features")
    fe = FeatureEncoder()
    test_X, test_y = fe.preprocessed_one_hot_encode(test_df, feature_type_list, task, feature_name)
    print("done")
    
    exp_info = {"task": task, "filename": filename, "control_flow_p": control_flow_p, "time_p": time_p,
                        "resource_p": resource_p, "data_p": data_p, "num_epochs": num_epochs,
                        "batch_size": batch_size}


    # Evaluate the model on the test data using `evaluate`
    CF_matrix, report, Accuracy, F1_Score, Precision, Recall = model.evaluate(test_X, test_y, exp_info)

    TruePositive = sum(np.diag(CF_matrix))
    FalsePositive = 0
    for jj in range(CF_matrix.shape[0]):
        FalsePositive += sum(CF_matrix[:, jj]) - CF_matrix[jj, jj]
    FalseNegative = 0
    for ii in range(CF_matrix.shape[0]):
        FalseNegative += sum(CF_matrix[ii, :]) - CF_matrix[ii, ii]
    TrueNegative = 0
    for kk in range(CF_matrix.shape[0]):
        temp = np.delete(CF_matrix, kk, 0)
        temp = np.delete(temp, kk, 1)
        TrueNegative += sum(sum(temp))

    # Calculate example leakage (based on the implementation of Abb et al., 2023)
    test_prefix_list = [row['activity_history'] for _, row in test_df.iterrows()]
    train_prefix_list = [row['activity_history'] for _, row in train_df.iterrows()]
    leaked = 0
    for activity_history in test_prefix_list:
        if (activity_history in train_prefix_list):
            leaked += 1
    leakage_percent = leaked / len(test_prefix_list)
    
    print("Leaked: ",leaked)
    print("Out of ", len(test_prefix_list))
    print(leakage_percent)

    Results_data = [data_set, sampling_technique, sample_size, Accuracy, leakage_percent, Running_Time, F1_Score, Precision, Recall, FeatureGenerator_Time,
                    TruePositive, FalsePositive, FalseNegative, TrueNegative]

    print("flag: saving output")
    with open(total_results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(Results_data)
    print("done")

    


 
          


