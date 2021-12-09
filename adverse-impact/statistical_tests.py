import warnings
import numpy as np
import pandas as pd

from scipy import stats

def impact_ratio(sr_min, sr_maj):
    '''Selection rate minority / selection rate majority'''
    return sr_min / sr_maj

def z_test(sr_min, sr_maj, sr_total, N, P):
    '''Compute Z-test scores (aka 2 standard deviation test)
    
    Parameters:
    ----------
    sr_min : float
        Minority selection rate; group applicants hired 
        divided by total number of group applicants
    sr_maj : float
        Majority selection rate; group applicants hired
        divided by total number of group applicants
    sr_total : float
        Selection rate regardless of group; total hired
        divided by total applied
    N : int
        Total number of applicants from all groups
    P : float
        Proportion of minorities computed as total minorities
        that applied divided by total applicants.
        
    Returns:
    -------
    z_d : float
        Pooled two-sample z-score test result
    '''
    z = (sr_min - sr_maj) / np.sqrt((sr_total * (1 - sr_total)) / (N * P * (1-P)))
    
    return z

def z_test_ir(sr_min, sr_maj, sr_total, N, P):
    '''Compute Z-test of the ratio of selection rates
    
    http://www.adverseimpact.org/CalculatingAdverseImpact/ZIR.htm
    
    Parameters:
    ----------
    sr_min : float
        Minority selection rate; group applicants hired 
        divided by total number of group applicants
    sr_maj : float
        Majority selection rate; group applicants hired
        divided by total number of group applicants
    sr_total : float
        Selection rate regardless of group; total hired
        divided by total applied
    N : int
        Total number of applicants from all groups
    P : float
        Proportion of minorities computed as total minorities
        that applied divided by total applicants.
        
    Returns:
    -------
    z_ir : float
        Z-test result of the ratio of selection
        rates
    '''
    z_ir = (np.log(sr_min / sr_maj)) / np.sqrt((1-sr_total) / (sr_total * N * P * (1-P)))
    
    return z_ir

def compute_standard_error(sr_min, sr_maj, N_min, N_maj):
    se = np.sqrt(((1-sr_min) / (N_min * sr_min)) + ((1-sr_maj) / (N_maj * sr_maj)))
    return se

def compute_fet(table):
    idx = table.index.tolist()
    cols = table.columns.tolist()
    assert idx == ['majority','minority'], "Must have majority/minority as index"
    assert cols == ['selected','not-selected'], "Must have selected/not-selected as columns"
    
    odds_ratio, p_value = stats.fisher_exact(table)
    return odds_ratio, p_value

def compute_chi2(table):
    idx = table.index.tolist()
    cols = table.columns.tolist()
    assert idx == ['majority','minority'], "Must have majority/minority as index"
    assert cols == ['selected','not-selected'], "Must have selected/not-selected as columns"
    
    try:
        chi2, p_chi2, dof, expected = stats.chi2_contingency(table, correction=False)
        return chi2, p_chi2, dof, expected
    except Exception as e:
        print(e)
        print(table)
        return None, None, None, None

def compute(table):
    '''
    Compute AI Statistical tests

    Parameters
    ----------
    table : pd.DataFrame
        The 2x2 table used for the analysis. Must
        be in the form with selected/not-selected column
        names and minority/majority index names

    Returns
    -------
    scores : dict
        A collection of all AI impact statistical scores.

    '''
    idx = table.index.tolist()
    cols = table.columns.tolist()
    assert idx == ['majority','minority'], "Must have majority/minority as index"
    assert cols == ['selected','not-selected'], "Must have selected/not-selected as columns"

    if table.loc[:,'not-selected'].sum() != 0:
        
        minority_group = table.loc['minority',:]
        majority_group = table.loc['majority',:]
        
        N = sum(majority_group) + sum(minority_group)     # Total population tested
        N_min = sum(minority_group)
        N_maj = sum(majority_group)
        min_selected = minority_group['selected']           # Minority population selected
        maj_selected = majority_group['selected']           # Majority population selected
        total_selected = min_selected + maj_selected
        
        sr_maj = maj_selected / float(sum(majority_group))      # majority selection rate
        sr_min = min_selected / float(sum(minority_group))      # minority selection rate
        sr_total = (min_selected + maj_selected) / float(N)     # total selection rate
    
        P = sum(minority_group) / float(N)                 # proportion of minorities
    
        # --
        # Impact Ratio
        # --
        ir = impact_ratio(sr_min, sr_maj)
        se = compute_standard_error(sr_min, sr_maj, N_min, N_maj)
        upper_ci = np.exp((np.log(ir) + (se * 1.96)))
        lower_ci = np.exp((np.log(ir) - (se * 1.96)))
        
        # --
        # Z tests
        # --
        z = z_test(sr_min, sr_maj, sr_total, N, P)
        z_ir = z_test_ir(sr_min, sr_maj, sr_total, N, P)
        
        # -- compute p-values
        p_z = 2 * (1 - stats.norm.cdf(np.abs(z)))
        p_z_ir = 2 * (1 - stats.norm.cdf(np.abs(z_ir)))
        
        # --
        # Fishers Exact
        # --
        fishers, p_fishers = compute_fet(table)
     
        # -- 
        # chi2
        # --
        chi2, p_chi2, dof, expected = compute_chi2(table)

        # -- possible sample size issues
        min_ef = min([val for sublist in expected for val in sublist]) 
        if  min_ef < 10:
            warnings.warn("Expected value must be greater than 10")
            
        scores = {
            'min_sr' : np.round(sr_min, 3),
            'maj_sr' : np.round(sr_maj,3),
            'ir' : np.round(ir, 3),
            'lower_ci' : np.round(lower_ci, 3),
            'upper_ci' : np.round(upper_ci, 3),
            'se' : np.round(se, 3),
            'z-test' : np.round(z, 3),
            'z-p' : np.round(p_z, 3),
            'z-test-ir' : np.round(z_ir, 3),
            'z-test-ir-p' : np.round(p_z_ir, 3),
            'expected_freq' : np.round(min_ef, 3),
            'fet-odds-ratio' : np.round(fishers, 3),
            'fet-p' : np.round(p_fishers, 3),
            'chi2' : np.round(chi2, 3),
            'chi2-p' : np.round(p_chi2, 3)
        }
        return scores
    else:
        print('Looks like everyone passed!!!')
        return {}
