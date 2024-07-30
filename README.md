# EVENT LOG SAMPLING FOR PREDICTIVE PROCESS MONITORING
## HOW DOES SAMPLING AFFECT THE RESULTS OF NEXT ACTIVITY PREDICTION IN PREDICTIVE PROCESS MONITORING?

This repositry contains the implementation and data used for my master thesis. 

The goal was improve the generalization of prediction models in the context of predictive process monitoring. Therefore, three different sampling techniques were tested on five event logs, aiming to decrase the example leakage between train and test log. 

The code contains implementations from different sources. The implementations are closely following the original code with small adaptions, so that the different parts work together.

Detailed results can be found in the folder "results", including a xlsx-file summarizing all executions.


### How to run the code

1. Clone repository
   - ```git clone ```

2. Install dependencies (Python 3.8.8)
   - ```pip install -r requirements.txt```

3. Unzip the sample event logs

4. How to:
    1. Run the script lstm.py with the following flags
        - *--task*: "next_activity", 
        - *--contextual_info*: "True" if you want to exploit contextual features, "False" otherwise (default: False),
        - *--transition*: "True" if the dataset contains transition information, "False" otherwise (default: False),
        - *--learning_rate*: learning rate of optimization algorithm (default: 0.002),
        - *--num_folds*: number of folds in validation (default: 10),
        - *--batch_size*: training batch size of models (default: 256),
        - *--data_dir*: directory where the dataset is located,
        - *--data_set*: name of the dataset,
        - *--checkpoint*: directory where the models are saved (default: "./checkpoints/"),
        - *--control_flow_p*: "True" if you want to use control-flows as features, "False" otherwise (default: True),
        - *--time_p*: "True" if you want to use time as features, "False" otherwise (default: True),
        - *--resource_p*: "True" if you want to use resource as features, "False" otherwise (default: False),
        - *--data_p*: "True" if you want to use data as features, "False" otherwise (default: False),
        - *--result_dir*: directory where the experimental results are saved
        - *--sampling_technique*: the desired sampling technique ("random", "relevance", "variant", "none", default = "none").
                                  - If you want to use variant-based sampling please provide the sampled event log with the following name: *dataset*_variant_samplesize.xes, e.g., "bpi2012_variant_50.xes"
                                  - If you want to use relevance-based sampling please provide the evetn log as a xes-file and a petri net of your event log, 
                                  e.g., bpi2012.xes and bpi2012.pnml
         - *--sample_size*: the desired sample size (default=1.0)              
      - e.g., 
        - ```python lstm.py --task "next_activity" --contextual_info False --learning_rate 0.002 --num_epochs 15 --batch_size 128 --data_dir "./samples/" --data_set "helpdesk" --checkpoint_dir "./checkpoints/" --control_flow_p True --time_p False --resource_p False --data_p False --transition False --result_dir "./result/" --sampling_technique "relevance"  --sample_size 0.5"``


### Citations:
Implementation of the LSTM neural networks:
The fundamental implementation and structure is from Fani Sani et al. This includes all folder unless dated otherwise.
The file lstm.py is based on next_activity_prediction_with_neural_networks.py and code was incorporated to include the chosen sampling methods and the calculation of the example leakage.
Fani Sani, M., Vazifehdoostirani, M., Park, G., Pegoraro, M., Van Zelst, S. J., & Van Der Aalst, W. M. P. (2023). Performance-preserving event log sampling for predictive monitoring. J Intell Inf Syst, 61(1), 53–82. https://doi.org/10.1007/s10844-022-00775-9

Relevance-based Sampling:
The folder "Relevance_based_sampling" contains the necessary code for this sampling technique. 
All classes were implemented by:
Kabierski, M., Nguyen, H. L., Grunske, L., & Weidlich, M. (2021). Sampling What Matters: Relevance-guided Sampling of Event Logs. 2021 3rd International Conference on Process Mining (ICPM), 64–71. https://doi.org/10.1109/ICPM53251.
2021.9576875

Variant-peserving Sampling:
To sample the event logs based on their variants a Plug-In in ProM was used, which was developed by Sani et al..
ProM - Package LogFiltering - action: Variant Sampling
Fani Sani, M., Vazifehdoostirani, M., Park, G., Pegoraro, M., van Zelst, S. J., & van der Aalst, W. M. (2021). Event log sampling for predictive monitoring. International Conference on Process Mining, 154–166. Retrieved December 18, 2023, from https://library.oapen.org/bitstream/handle/20.500.12657/54026/978- 3- 030-98581-3.pdf?sequence=1#page=167

Example leakage calculations:
THe calculation of the example leakage was implemented based on the code of:
Abb, L., Pfeiffer, P., Fettke, P., & Rehse, J.-R. (2023). A Discussion on Generalization in Next-Activity Prediction [arXiv:2309.09618 [cs]]. Retrieved December 11, 2023, from http://arxiv.org/abs/2309.09618


Event logs:
- bpi2012: van Dongen, B.: BPI Challenge 2012. https://data.4tu.nl/articles/_/12689204/1
- bpi2013: Steeman, W.: BPI Challenge 2013, incidents. https://data.4tu.nl/articles/_/12693914/1
- bpi2017: van Dongen, B.: BPI Challenge 2017. https://data.4tu.nl/articles/_/12696884/1
  bpi2017 contains the half of the original event log, because too much memory space is needed otherwise
- helpdesk: Verenich, I.: Helpdesk event log. https://doi.org/10.17632/39bp3vv62t.1
- mobis2019: Scheid, M., Rehse, J.R., Houy, C., Fettke, P.: Data set for mobis challenge 2019 (2018). https://doi.org/10.13140/RG.2.2.11870.28487

Note: Sampling and model training may take several hours for the event logs bpi2012 and bpi2017.


### Requirements

- Datetime format: "%Y.%m.%d %H:%M".
- The input csv file contains the columns named after the xes elements, e.g., concept:name
- Csv format dataset where 'case:concept:name' denotes the caseid, 'concept:name' the activity,'org:resource' the resource, 'time:timestamp' the complete timestamp, 'event_@' the event attribute, and 'case_@' the case attribute.
- The caseid must be an integer
