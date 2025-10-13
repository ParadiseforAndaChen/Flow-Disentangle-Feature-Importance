import csv
import os

num_runs = 100 

txt_dir_loco_scpi = "Flow_Disentangle_Feature_Importance/Exp1/0.8/1000/phi_logs/loco_scpi_phi_logs" 
csv_path_loco_scpi = "Flow_Disentangle_Feature_Importance/Exp1/0.8/1000/summary/loco_scpi_summary.csv"

txt_dir_dfi = "Flow_Disentangle_Feature_Importance/Exp1/0.8/1000/phi_logs/dfi_phi_logs"
csv_path_dfi = "Flow_Disentangle_Feature_Importance/Exp1/0.8/1000/summary/dfi_summary.csv"

txt_dir_cpi = "Flow_Disentangle_Feature_Importance/Exp1/0.8/1000/phi_logs/cpi_phi_logs"
csv_path_cpi = "Flow_Disentangle_Feature_Importance/Exp1/0.8/1000/summary/cpi_summary.csv"

os.makedirs(txt_dir_loco_scpi, exist_ok=True)
os.makedirs(txt_dir_dfi, exist_ok=True)
os.makedirs(txt_dir_cpi, exist_ok=True)

with open(csv_path_loco_scpi, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_0_loco',  'auc_score_x_scpi',
        'power_0_loco', 'type1_0_loco', 'count_0_loco',
        'power_x_scpi', 'type1_x_scpi', 'count_x_scpi',
    ])

with open(csv_path_dfi, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_x_dfi',
        'power_x_dfi', 'type1_x_dfi', 'count_x_dfi',       
    ])   

with open(csv_path_cpi, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'run', 
        'auc_score_0_cpi',  'auc_score_x_fdfi',
        'power_0_cpi', 'type1_0_cpi', 'count_0_cpi', 
        'power_x_cpi', 'type1_x_cpi', 'count_x_cpi',        
    ]) 

for run in range(1, num_runs + 1):
    print(f"\n================  Run {run}  ================\n")
    
    #########################################################################################################################################################################
    #                                                                           Flow Matching                                                                               #
    #########################################################################################################################################################################
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    
    from Flow_Disentangle_Feature_Importance import Exp1, evaluate_importance

    from sklearn.ensemble import RandomForestRegressor
    
    seed = np.random.randint(0, 10000)  
    rho = 0.8
    X_full, y = Exp1().generate(3000, rho, seed)
    
    D=X_full.shape[1]
    
    n_jobs=42

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    from Flow_Disentangle_Feature_Importance import FlowMatchingModel
    import torch
    model = FlowMatchingModel(
        X=X_full,
        dim=D,
        device=device,
        hidden_dim=128,        
        time_embed_dim=64,     
        num_blocks=1,
        use_bn=False,
        seed=seed
    )
    model.fit(num_steps=15000, batch_size=256, lr=1e-3, show_plot=False)

    from scipy.stats import norm

    X_full, y = Exp1().generate(1000, rho, seed) 
    
    #########################################################################################################################################################################
    #                                                                           LOCO                                                                                        #
    #########################################################################################################################################################################
    
    from Flow_Disentangle_Feature_Importance import LOCOEstimator

    estimator1 = LOCOEstimator(
            regressor =  RandomForestRegressor(
                 n_estimators=500,
                 max_depth=None,
                 min_samples_leaf=5,
                 random_state=seed,
                 n_jobs=n_jobs
                 )
    )
    
    phi_0_loco, se_0_loco = estimator1.importance(X_full, y)
    
    print("Feature\tLOCO_0 φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_0_loco, se_0_loco)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"Sum of LOCO_0: {D* np.mean(phi_0_loco)}")
    
    phi_0_loco_test = phi_0_loco 

    se_0_loco_test = se_0_loco 
    
    z_score_0_loco = phi_0_loco_test / se_0_loco_test
    
    p_value_0_loco = 1 - norm.cdf(z_score_0_loco)

    rounded_p_value_0_loco = np.round(p_value_0_loco, 3)

    print(rounded_p_value_0_loco)

    #########################################################################################################################################################################
    #                                                                         FDFI(SCPI)                                                                                    #
    #########################################################################################################################################################################
    from Flow_Disentangle_Feature_Importance import SCPI_Flow_Model_Estimator
    estimator2 = SCPI_Flow_Model_Estimator(
        regressor =  RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=n_jobs
                ),
        flow_model=model
    )

    phi_x_scpi, se_x_scpi = estimator2.importance(X_full, y)

    print("Feature\tSCPI_X φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_x_scpi, se_x_scpi)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"Sum of SCPI_X: {D* np.mean(phi_x_scpi)}")
    
    phi_x_scpi_test = phi_x_scpi 

    se_x_scpi_test = se_x_scpi 
    
    z_score_x_scpi = phi_x_scpi_test / se_x_scpi_test
    
    p_value_x_scpi = 1 - norm.cdf(z_score_x_scpi)

    rounded_p_value_x_scpi = np.round(p_value_x_scpi, 3)
    
    print(rounded_p_value_x_scpi)
    
    #---------------------------------------------------------------------------------------#
    #                               Power&Type 1 Error&Auc                                  #
    #---------------------------------------------------------------------------------------#

    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)
    ignore_idx = list(range(5, 10))

    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False

    auc_score_0_loco = roc_auc_score(y_true[mask], np.array(phi_0_loco)[mask])
    auc_score_x_scpi = roc_auc_score(y_true[mask], np.array(phi_x_scpi)[mask])

    
    print(f"AUC for phi_0_loco: {auc_score_0_loco:.4f}")
    print(f"AUC for phi_x_scpi: {auc_score_x_scpi:.4f}")

    alpha = 0.05
    
    power_x_scpi, type1_x_scpi, count_x_scpi = evaluate_importance(p_value_x_scpi, y_true, alpha)
    power_0_loco, type1_0_loco, count_0_loco = evaluate_importance(p_value_0_loco, y_true, alpha)

        
    print(f"[X]    Power: {power_x_scpi:.3f}   Type I Error: {type1_x_scpi:.3f}   Count: {count_x_scpi}")
    print(f"[0]    Power: {power_0_loco:.3f}   Type I Error: {type1_0_loco:.3f}   Count: {count_0_loco}")


    with open(f"{txt_dir_loco_scpi}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_0\tse_0\tphi_x\tse_x\n")
        for i in range(len(phi_x_scpi)):
            txtfile.write(
                          f"{phi_0_loco[i]:.6f}\t{se_0_loco[i]:.6f}\t"
                          f"{phi_x_scpi[i]:.6f}\t{se_x_scpi[i]:.6f}\n")

    with open(csv_path_loco_scpi, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_0_loco, 6),  round(auc_score_x_scpi, 6),
            round(power_0_loco, 6), round(type1_0_loco, 6), int(count_0_loco),
            round(power_x_scpi, 6), round(type1_x_scpi, 6), int(count_x_scpi),
        ])

#########################################################################################################################################################################
#                                                                           DFI                                                                                         #
#########################################################################################################################################################################
    from Flow_Disentangle_Feature_Importance import DFIEstimator

    estimator3 = DFIEstimator(
        regressor =  RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=n_jobs
                )
    )

    phi_x_dfi, se_x_dfi = estimator3.importance(X_full, y)
    
    print("Feature\tDFI_X φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_x_dfi, se_x_dfi)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"Sum of DFI_X: {D* np.mean(phi_x_dfi)}")
    
    phi_x_dfi_test = phi_x_dfi 

    se_x_dfi_test = se_x_dfi 
    
    z_score_x_dfi = phi_x_dfi_test / se_x_dfi_test
    
    p_value_x_dfi = 1 - norm.cdf(z_score_x_dfi)

    rounded_p_value_x_dfi = np.round(p_value_x_dfi, 3)
    
    print(rounded_p_value_x_dfi)

    #---------------------------------------------------------------------------------------#
    #                               Power&Type 1 Error&Auc                                  #
    #---------------------------------------------------------------------------------------#
    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)
    ignore_idx = list(range(5, 10))

    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False

    auc_score_x_dfi = roc_auc_score(y_true[mask], np.array(phi_x_dfi)[mask])
    print(f"AUC for phi_x_dfi: {auc_score_x_dfi:.4f}")

    alpha = 0.05
    
    power_x_dfi, type1_x_dfi, count_x_dfi = evaluate_importance(p_value_x_dfi, y_true, alpha)
    print(f"[X]    Power: {power_x_dfi:.3f}   Type I Error: {type1_x_dfi:.3f}   Count: {count_x_dfi}")

    with open(f"{txt_dir_dfi}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_x\tse_x\n")
        for i in range(len(phi_x_dfi)):
            txtfile.write(
                          f"{phi_x_dfi[i]:.6f}\t{se_x_dfi[i]:.6f}\n")
            
    with open(csv_path_dfi, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_x_dfi, 6),
            round(power_x_dfi, 6), round(type1_x_dfi, 6), int(count_x_dfi),
        ])
#########################################################################################################################################################################
#                                                                       CPI                                                                                             #
#########################################################################################################################################################################

    from Flow_Disentangle_Feature_Importance import CPIEstimator, CPI_Flow_Model_Estimator

    #---------------------------------------------------------------------------------------#
    #                                       CPI_0                                           #
    #---------------------------------------------------------------------------------------#

    estimator6 = CPIEstimator(
            regressor =  RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=n_jobs
                )
    )

    phi_0_cpi, se_0_cpi = estimator6.importance(X_full, y)

    print("Feature\tCPI_0 φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_0_cpi, se_0_cpi)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"Sum of CPI_0: {D* np.mean(phi_0_cpi)}")
    
    phi_0_cpi_test = phi_0_cpi 

    se_0_cpi_test = se_0_cpi 

    z_score_0_cpi = phi_0_cpi_test / se_0_cpi_test
    
    p_value_0_cpi = 1 - norm.cdf(z_score_0_cpi)

    rounded_p_value_0_cpi = np.round(p_value_0_cpi, 3)

    print(rounded_p_value_0_cpi)

    #---------------------------------------------------------------------------------------#
    #                                      FDFI(CPI)                                        #
    #---------------------------------------------------------------------------------------#

    estimator8 = CPI_Flow_Model_Estimator(
        regressor =  RandomForestRegressor(
                 n_estimators=500,
                 max_depth=None,
                 min_samples_leaf=5,
                 random_state=seed,
                 n_jobs=n_jobs
                 ),
        flow_model=model
    
    )
    
    phi_x_cpi, se_x_cpi = estimator8.importance(X_full, y)
    
    print("Feature\tCPI_X φ\tStdError")
    for j, (phi_j, se_j) in enumerate(zip(phi_x_cpi, se_x_cpi)):
        print(f"{j:>3d}\t{phi_j: .4f}\t{se_j: .4f}")
    print(f"Sum of CPI_X: {D* np.mean(phi_x_cpi)}")
    
    phi_x_cpi_test = phi_x_cpi 

    se_x_cpi_test = se_x_cpi 
    
    z_score_x_cpi = phi_x_cpi_test / se_x_cpi_test
    
    
    p_value_x_cpi = 1 - norm.cdf(z_score_x_cpi)

    rounded_p_value_x_cpi = np.round(p_value_x_cpi, 3)
    
    print(rounded_p_value_x_cpi)

#---------------------------------------------------------------------------------------#
#                               Power&Type 1 Error&Auc                                  #
#---------------------------------------------------------------------------------------#
    from sklearn.metrics import roc_auc_score

    y_true = np.array([1]*5 + [0]*45)
    ignore_idx = list(range(5, 10))

    mask = np.ones_like(y_true, dtype=bool)
    mask[ignore_idx] = False

    auc_score_0_cpi = roc_auc_score(y_true[mask], np.array(phi_0_cpi)[mask])
    auc_score_x_cpi = roc_auc_score(y_true[mask], np.array(phi_x_cpi)[mask])
        
    print(f"AUC for phi_0_cpi: {auc_score_0_cpi:.4f}")
    print(f"AUC for phi_x_cpi: {auc_score_x_cpi:.4f}")

    alpha = 0.05
    
    power_x_cpi, type1_x_cpi, count_x_cpi = evaluate_importance(p_value_x_cpi, y_true, alpha)
    power_0_cpi, type1_0_cpi, count_0_cpi = evaluate_importance(p_value_0_cpi, y_true, alpha)
       
    print(f"[X]    Power: {power_x_cpi:.3f}   Type I Error: {type1_x_cpi:.3f}   Count: {count_x_cpi}")
    print(f"[0]    Power: {power_0_cpi:.3f}   Type I Error: {type1_0_cpi:.3f}   Count: {count_0_cpi}")
    
    with open(f"{txt_dir_cpi}/phi_values_all_runs.txt", 'a') as txtfile:
        txtfile.write(f"\n### Run {run} ###\n")
        txtfile.write("Feature\tphi_0\tse_0\tphi_x\tse_x\n")
        for i in range(len(phi_x_cpi)):
            txtfile.write(
                          f"{phi_0_cpi[i]:.6f}\t{se_0_cpi[i]:.6f}\t"
                          f"{phi_x_cpi[i]:.6f}\t{se_x_cpi[i]:.6f}\n")
            
    with open(csv_path_cpi, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            round(auc_score_0_cpi, 6),  round(auc_score_x_cpi, 6),
            round(power_0_cpi, 6), round(type1_0_cpi, 6), int(count_0_cpi),
            round(power_x_cpi, 6), round(type1_x_cpi, 6), int(count_x_cpi),
        ])
