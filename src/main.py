from utils import ModelEval, SignalPlot
from model import HybridModel, AutoGModel
from pipeline import LabelPipeline, InteractionPipeline, SignalAugmentationPipeline
from plot import Plots
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

if __name__ == '__main__':
    SEED = 42
    cnn_params = {
    'input_length': 4000,
    'embedding_dim': 256,
    'kernel_sizes': [3, 3, 5, 5],  
    'num_filters': 64,         
    'drop_out': 0.2,            
    }
    classifier_params = {
        'n_estimators':500,         
        'max_depth': 7,             
        'learning_rate': 0.05,      
        'subsample': 0.8,           
        'colsample_bytree': 0.8,    
        'min_child_weight': 0.9,    
        'gamma': 0,                 
        'reg_alpha': 0.1,           
        'reg_lambda': 1.0,          
        'objective': 'binary:logistic',  
        'eval_metric': 'logloss'
    }
    train_df = pd.read_csv(r'data/traindata.csv')
    test_df = pd.read_csv(r'data/testdata.csv')
    lp = LabelPipeline(train_df)
    labels = np.loadtxt(r'data/unlabeled_predictions.csv', delimiter=',')
    
    labeled_signals, targets = lp.add_labels(labels, cutedge=(500, 1000))
    sap = SignalAugmentationPipeline(
        labeled_signals, targets,
        window_length=5, lag=5, noise_level=0.05, diff=False)
    labeled_signals, targets = sap.get_processed_data()
    print(f"Total labeled signals: {labeled_signals.shape[0]}, "
          f"Total targets: {targets.shape[0]}")
    

    X_train, X_val, y_train, y_val = train_test_split(
        labeled_signals, targets, test_size=0.2, random_state=SEED
    )
    model = HybridModel(cnn_params=cnn_params, 
                        classifier_params=classifier_params)
    model.train(X_train, y_train, learning_rate=0.00015, 
                num_epochs=200, batch_size=32, splits=10, early_stopping=50)
    model.train_process_plot(save=True, val_loss_log=False)
    result = model.evaluate(X_val, y_val)
    _, test_pred = model.predict(test_df)
    
    pd.DataFrame(test_pred).to_csv(r'data/test_predictions.csv', 
                                   index=False,
                                   header=False)# remove the first one if necessary

    print(result['f1'])



   