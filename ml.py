

import os
import pathlib

import numpy as np
import pandas as pd
import scipy as sp

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, log_loss

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

import matplotlib.pyplot as plt
from austen_plots.AustenPlot import AustenPlot

PLAYER_DATA = "data/player_risk_2019_20games.csv"
RANDOM_SEED = 35     
np.random.seed(RANDOM_SEED)

lb = LabelEncoder()

def cleanData(risks):
    """
    PARAMETERS
    ----------
    risks : pd.df
    
    RETURNS
    ----------
    pd.df

    """
    # make title a bool
    risks['title'] = risks['title'].apply(lambda s: 0 if s == "NULL" else 1)

    # make gender a bool (0 for 'M', 1 for 'F')
    risks['gender'] = risks['sex'].apply(lambda s: 0 if s == "M" else 1)
    risks = risks.drop('sex', axis=1) # 'sex' column is self reported

    # transform birthday into age
    risks['age'] = risks['birthday'].apply(lambda i: 2019 - i)
    risks = risks.drop('birthday', axis=1)
    # 13 players have no age, probably safe to drop
    risks = risks.dropna(subset=['age'])

    # label countries
    risks['country'] = lb.fit_transform(risks['country'])


    return risks

class Data:
    @classmethod
    def fetchData(clz, filename):
        """
        PARAMETERS
        ----------
        filename : str
        
        RETURNS
        ----------
        Data
        
        """
        
        
        risks = pd.read_csv(PLAYER_DATA, skipinitialspace=True)
        risks = cleanData(risks)
        
        treatment    = risks['gender']
        outcome      = risks['avrisk']
        confounders = risks.drop(columns=['gender', 'avrisk', 'fideid', 'name'])
        
        return clz(treatment, outcome, confounders)

    def __init__(self, treatment, outcome, confounders):
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders

    def copy(self):
        return Data(self.treatment.copy(), self.outcome.copy(), self.confounders.copy())
        
class NuisanceQModel:
    def __init__(self, model):
        self.model = model

    def getModel(self):
        return self.model

    def sanityCheck(self, data):
        model = self.getModel()

        X_w_treatment = data.confounders
        X_w_treatment['treatment'] = data.treatment

        X_train, X_test, y_train, y_test = train_test_split(X_w_treatment, data.outcome, test_size=0.2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_mse = mean_squared_error(y_pred, y_test)
        baseline_mse = mean_squared_error(y_train.mean()*np.ones_like(y_test), y_test)

        return test_mse, baseline_mse

 
class NuisanceGModel:
    def __init__(self, model):
        self.model = model

    def getModel(self):
        return self.model

    def sanityCheck(self, data):

        model = self.getModel()

        X_train, X_test, a_train, a_test = train_test_split(data.confounders, data.treatment, test_size=0.2)
        model.fit(X_train, a_train)
        a_pred = model.predict_proba(X_test)[:,1]

        test_ce = log_loss(a_test, a_pred)
        baseline_ce = log_loss(a_test, a_train.mean()*np.ones_like(a_test))

        return test_ce, baseline_ce

def treatment_k_fold_fit_and_predict(make_model, data, n_splits):
    """
    Implements K fold cross-fitting for the model predicting the treatment A. 

    """

    A = data.treatment
    X = data.confounders

    predictions = np.full_like(A, np.nan, dtype=float)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    g_ces = []
    ce_baselines = []
    
    for train_index, test_index in kf.split(X, A):
        X_train = X.iloc[train_index]
        A_train = A.iloc[train_index]
        a_test = A.iloc[test_index]
        a_train = A_train
        g = make_model()
        g.fit(X_train, A_train)

        # get predictions for split
        predictions[test_index] = g.predict_proba(X.iloc[test_index])[:, 1]
        a_pred = predictions[test_index]

        # ce
        ce = log_loss(a_test, a_pred)
        baseline_ce = log_loss(a_test, a_train.mean()*np.ones_like(a_test))
        g_ces.append(ce)
        ce_baselines.append(baseline_ce)



    assert np.isnan(predictions).sum() == 0
    return predictions, np.mean(g_ces), np.mean(ce_baselines)


def outcome_k_fold_fit_and_predict(make_model, data, n_splits, output_type):
    """
    Implements K fold cross-fitting for the model predicting the outcome Y. 
    """
    A = data.treatment
    X = data.confounders
    y = data.outcome

    predictions0 = np.full_like(A, np.nan, dtype=float)
    predictions1 = np.full_like(y, np.nan, dtype=float)
    if output_type == 'binary':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    elif output_type == 'continuous':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    # include the treatment as input feature
    X_w_treatment = X.copy()
    X_w_treatment["A"] = A

    # for predicting effect under treatment / control status for each data point 
    X0 = X_w_treatment.copy()
    X0["A"] = 0
    X1 = X_w_treatment.copy()
    X1["A"] = 1

    Q_mses = []
    mse_baselines = []
    
    for train_index, test_index in kf.split(X_w_treatment, y):
        X_train = X_w_treatment.iloc[train_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        q = make_model()
        q.fit(X_train, y_train)

        if output_type =='binary':
            predictions0[test_index] = q.predict_proba(X0.iloc[test_index])[:, 1]
            predictions1[test_index] = q.predict_proba(X1.iloc[test_index])[:, 1]
        elif output_type == 'continuous':
            predictions0[test_index] = q.predict(X0.iloc[test_index])
            predictions1[test_index] = q.predict(X1.iloc[test_index])

        
        X_train, X_test = X_w_treatment.iloc[train_index], X_w_treatment.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        q = make_model()
        q.fit(X_train, y_train)

        y_pred = q.predict(X_test)
        Q_mse = mean_squared_error(y_test, y_pred)
        baseline_mse = mean_squared_error(y_train.mean()*np.ones_like(y_test), y_test)
        Q_mses.append(Q_mse)
        mse_baselines.append(baseline_mse)
        
    assert np.isnan(predictions0).sum() == 0
    assert np.isnan(predictions1).sum() == 0
    return predictions0, predictions1, np.mean(Q_mses), np.mean(mse_baselines)

def ate_aiptw(Q0, Q1, g, data, prob_t=None):
  """
  # Double ML estimator for the ATE
  """
  A = data.treatment
  Y = data.outcome

  tau_hat = (Q1 - Q0 + A*(Y-Q1)/g - (1-A)*(Y-Q0)/(1-g)).mean()
  
  scores = Q1 - Q0 + A*(Y-Q1)/g - (1-A)*(Y-Q0)/(1-g) - tau_hat
  n = Y.shape[0] # number of observations
  std_hat = np.std(scores) / np.sqrt(n)

  return tau_hat, std_hat

def runModel(q_model, g_model, data, name):

    # k fold and train
    g, ce, cebas = treatment_k_fold_fit_and_predict(g_model.getModel, data, n_splits=10)
    print("CE:")
    print(ce, cebas)
    plt.figure()
    plt.hist(g, density=True)
    plt.suptitle(name)
    plt.xlabel("Propensity Score")
    plt.savefig(name)
    Q0, Q1, mse, msebas = outcome_k_fold_fit_and_predict(q_model.getModel, data, n_splits=10, output_type="continuous")
    print("MSE:")
    print(mse, msebas)

    # find ate and return
    return ate_aiptw(Q0, Q1, g, data)

def sensitivityAnal(target_bias, make_g_model, make_Q_model, data):
    confounders = data.confounders
    treatment = data.treatment
    outcome=data.outcome
    
    covariates = {'rating': ['rating'],
                  'country': ['country'],
                  'title': ['title'],
                  'k': ['k'],
                  'age': ['age'],
                  'gamecount': ['gamecount'],
                  'oprating': ['oprating ']}

    nuisance_estimates = {}
    for group, covs in covariates.items():
        remaining_confounders = confounders.drop(columns=covs)

        g, _, _ = treatment_k_fold_fit_and_predict(make_g_model, data, n_splits=10)
        Q0, Q1, _, _ = outcome_k_fold_fit_and_predict(make_Q_model, data, n_splits=10, output_type="continuous")

        data_and_nuisance_estimates = pd.DataFrame({'g': g, 'Q0': Q0, 'Q1': Q1, 'A': treatment, 'Y': outcome})
        nuisance_estimates[group] = data_and_nuisance_estimates

    data_and_nuisance_path = 'data_and_nuisance_estimates.csv'
    covariate_dir_path = 'covariates/'

    def _convert_to_austen_format(nuisance_estimate_df: pd.DataFrame):
        austen_df = pd.DataFrame()
        austen_df['y']=nuisance_estimate_df['Y']
        austen_df['t']=nuisance_estimate_df['A']
        austen_df['g']=nuisance_estimate_df['g']
        A = nuisance_estimate_df['A']
        austen_df['Q']=A*nuisance_estimate_df['Q1'] + (1-A)*nuisance_estimate_df['Q0'] # use Q1 when A=1, and Q0 when A=0

        return austen_df

    austen_data_and_nuisance = _convert_to_austen_format(data_and_nuisance_estimates)
    austen_data_and_nuisance.to_csv(data_and_nuisance_path, index=False)
    
    pathlib.Path(covariate_dir_path).mkdir(exist_ok=True)
    for group, nuisance_estimate in nuisance_estimates.items():
        austen_nuisance_estimate = _convert_to_austen_format(nuisance_estimate)
        austen_nuisance_estimate.to_csv(os.path.join(covariate_dir_path,group+".csv"), index=False)

    ap = AustenPlot(data_and_nuisance_path, covariate_dir_path)
    p, plot_coords, variable_coords = ap.fit(bias=target_bias) # recall we set target_bias=2.0
    p.save("austen_plot")
    

risk_data = Data.fetchData(PLAYER_DATA)

models = [ (NuisanceQModel(RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=500, max_depth=None)),
            NuisanceGModel(RandomForestClassifier(n_estimators=500, max_depth=None)), "RF (depth = max)"),
           (NuisanceQModel(RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=500, max_depth=10)),
            NuisanceGModel(RandomForestClassifier(n_estimators=500, max_depth=10)), "RF (depth = 10)"),
           (NuisanceQModel(XGBRegressor(objective="reg:squarederror", random_state=RANDOM_SEED, n_jobs=16)),
            NuisanceGModel(XGBClassifier(objective="binary:logistic", random_state=RANDOM_SEED, n_jobs=16)), "XGBoost"),
           (NuisanceQModel(LinearRegression()),
            NuisanceGModel(LogisticRegression(max_iter=10000)), "Lin\Log") ]

for q_model, g_model, name in models:
    #print(q_model.sanityCheck(risk_data.copy()))
    #print(g_model.sanityCheck(risk_data.copy()))

    #print("next model")
    #print(runModel(q_model, g_model, risk_data, name))
    pass


lin_q, lin_g, _ = models[-1]
sensitivityAnal(.45, lin_g.getModel, lin_q.getModel, risk_data)




    

    

        




