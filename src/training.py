from tqdm import tqdm
import torch.nn.functional as F
import torch
from distributions import StandardNormalSampler
from embeddings import one_hot_encoder

def pretrain_parabola(model, DIM, NUM, embedder,device, BATCH_SIZE = 1024):
    X0_sampler = StandardNormalSampler(dim=DIM, requires_grad=True, device=device)
    model_opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train(True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_opt, mode='min', factor=0.5, patience=1400)

    iteration_count = tqdm(range(20_000))
    for _ in iteration_count:
        X = (X0_sampler.sample(BATCH_SIZE)).detach() * 4
        X.requires_grad_(True)
        enc = embedder.encode(torch.randint(0,NUM,(BATCH_SIZE,),device = X.device,requires_grad=True,dtype=float))
        output = model.push(X,enc)
        loss = F.mse_loss(output, X)# + LAMBDA_REG_POS * D.positive_constraint_loss()
        model_opt.zero_grad()
        loss.backward()
        model_opt.step()
        scheduler.step(loss)
        # D.enforce_positive_weights()
        iteration_count.set_postfix(loss = loss.item())
        
        if loss.item() < 1e-2:
            break

    print('Final Loss:', loss.item())

def calculate_val_loss(D, D_conj, encoder, samplers_valid, proposed_sampler, alphas, LAMBDA, R_LAMBDA, BATCH_SIZE, NUM, device):
    D.train(False); D.train(False); 
    loss = 0.
    pretrain_encoder = one_hot_encoder(NUM)

    for n in range(NUM):
        X,S = samplers_valid[n].sample(BATCH_SIZE)
        X = X.to(device)
        S = S.to(device).flatten()
        X.requires_grad_(True)
        n_enc = encoder.encode(S).to(device)
        X_inv = D.push(X, n_enc).detach()
        loss += alphas[n] * ((X_inv * X).sum(dim=1).reshape(-1, 1) - D_conj(X_inv, n_enc)).mean()
    
    # Cycle Loss
    cycle_loss = 0.
    
    for n in range(NUM):
        X,S = samplers_valid[n].sample(BATCH_SIZE)
        X = X.to(device)
        S = S.to(device).flatten()
        X.requires_grad_(True)
        # Cycle loss
        n_enc = encoder.encode(S).to(device)
        X_inv = D.push(X,n_enc)
        cycle_loss += alphas[n] * ((D_conj.push(X_inv,n_enc) - X.detach()) ** 2).mean() 
    
    loss += LAMBDA * cycle_loss
    
    # Congruence Regularization Loss
    reg_loss = 0.

    Y = torch.cat(
        [proposed_sampler.sample(BATCH_SIZE).detach()] +
        [D.push(samplers_valid[n].sample(BATCH_SIZE)[0].to(device).requires_grad_(True),encoder.encode(samplers_valid[n].sample(BATCH_SIZE)[1].flatten()).to(device)).detach() for n in range(NUM)],
        dim=0
    ).detach()

    for n in range(NUM):
        reg_loss += alphas[n] * D_conj(Y, pretrain_encoder.encode(n*torch.ones(len(Y),device=X.device)))
    reg_loss -= ((Y ** 2).sum(dim=1) / 2).reshape(-1, 1)
    reg_loss = torch.relu(reg_loss).mean()
    
    loss += R_LAMBDA * reg_loss

    return loss


##### delete this
import numpy as np
import pandas as pd
from distributions import DatasetSampler
from embeddings import crime_encoder
from networks import PICNN
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from fairness_metrics import compute_KS
from utils import split_data


def pipeline(trial_n, X_all, S_all_data, Y_all, test_OT_size, test_models_size, ot_valid_size, feature_dim, feature_s_dim, num_layers, D_LR, MAX_ITER, LAMBDA, R_LAMBDA, alpha_ridge, n_estimators, max_depth, gamma_RKLS, score_freq, device, BATCH_SIZE):

    random_state = trial_n
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    print("------------------------------------")
    print("Trial number: ", trial_n)

    X_train_OT, X_test_OT, X_valid_OT, S_train_OT, S_test_OT,S_valid_OT, Y_train_OT, Y_test_OT, X_train_unfair, X_test_unfair, S_train_unfair, S_test_unfair, Y_train_unfair, Y_test_unfair, X_train_fair, X_test_fair, S_train_fair, S_test_fair, Y_train_fair, Y_test_fair = split_data(X_all, S_all_data['sensitive'], Y_all, test_OT_size=test_OT_size, test_models_size=test_models_size, ot_valid_size=ot_valid_size, random_state=random_state)

    # Create an instance of the sampler
    samplers = []
    for S_val in np.unique(S_train_OT): 
        index_val = (S_train_OT == S_val).squeeze()
        print("instances of s = ", S_val, ":", sum(index_val))
        samplers.append(DatasetSampler(X_train_OT.loc[index_val], S_train_OT.loc[index_val], device=device))

    DIM = samplers[0].X_dim
    NUM = len(samplers)
        
    encoder = crime_encoder(NUM, np.unique(S_train_OT))

    D = PICNN(input_x_dim=DIM,input_s_dim=NUM,feature_dim=feature_dim, feature_s_dim=feature_s_dim, out_dim= 1,num_layers=num_layers).to(device)
    print(np.sum([np.prod(list(p.shape)) for p in D.parameters()]), 'parameters in Discriminative Network')

    D.initialize_weights('gaussian',device)

    pretrain_encoder = one_hot_encoder(NUM) 
    pretrain_parabola(D, DIM, NUM, pretrain_encoder, device)

    D_conj = deepcopy(D)

    D_opt = torch.optim.Adam(D.parameters(), lr=D_LR)
    D_conj_opt = torch.optim.Adam(D_conj.parameters(), lr=D_LR)

    # initialize validation samplers
    samplers_valid = []
    for S_val in np.unique(S_valid_OT):
        index_val = (S_valid_OT == S_val).squeeze()
        print("instances of s = ", S_val, ":", sum(index_val))
        samplers_valid.append(DatasetSampler(X_valid_OT.loc[index_val], S_valid_OT.loc[index_val], device=device))

    alphas = [1/NUM]*NUM
    proposed_sampler =  StandardNormalSampler(dim=DIM, requires_grad=True, device=device)

    it = 0
    loss_history = []
    loss_validate_history = []
    full_log = {}

    iters = tqdm(range(MAX_ITER))
    for _ in iters:
        it += 1
        D.train(True); D.train(True); 
        loss = 0.

        for n in range(NUM):
            X,S = samplers[n].sample(BATCH_SIZE)
            X = X.to(device)
            X.requires_grad_(True)
            n_enc = encoder.encode(S.flatten())
            n_enc = n_enc.to(device)
            X_inv = D.push(X, n_enc).detach()
            loss += alphas[n] * ((X_inv * X).sum(dim=1).reshape(-1, 1) - D_conj(X_inv, n_enc)).mean()
        log = {'pure_loss' : loss.item()}
        
        # Cycle Loss
        cycle_loss = 0.
        
        for n in range(NUM):
            X,S = samplers[n].sample(BATCH_SIZE)
            X = X.to(device)
            X.requires_grad_(True)
            # Cycle loss
            n_enc = encoder.encode(S.flatten())
            n_enc = n_enc.to(device)
            X_inv = D.push(X,n_enc)
            cycle_loss += alphas[n] * ((D_conj.push(X_inv,n_enc) - X.detach()) ** 2).mean()
        
        loss += LAMBDA * cycle_loss
        
        # Congruence Regularization Loss
        reg_loss = 0.

        Y = torch.cat(
            [proposed_sampler.sample(BATCH_SIZE).detach()] +
            [D.push(samplers[n].sample(BATCH_SIZE)[0].to(device).requires_grad_(True),
                    encoder.encode(samplers[n].sample(BATCH_SIZE)[1].flatten().to(device))
                    ).detach() for n in range(NUM)],
            dim=0
        ).detach()

        for n in range(NUM):
            reg_loss += alphas[n] * D_conj(Y, pretrain_encoder.encode(n*torch.ones(len(Y),device=X.device))) 
        reg_loss -= ((Y ** 2).sum(dim=1) / 2).reshape(-1, 1)
        reg_loss = torch.relu(reg_loss).mean()
        
        loss += R_LAMBDA * reg_loss
        
        # # Positive constraint ! add this after change PICNN
        # loss += LAMBDA_REG_POS * (D.positive_constraint_loss()+D_conj.positive_constraint_loss())
        
        loss_history.append(loss.item())
        loss.backward()
        D_opt.step(); D_conj_opt.step()
        D_opt.zero_grad(); D_conj_opt.zero_grad(); 
        
        # for n in range(benchmark.num):
        #     D_conj.enforce_positive_weights()
        #     D.enforce_positive_weights()

        loss_validate_history.append(calculate_val_loss(D, D_conj, encoder, samplers_valid, 
                                                            proposed_sampler, alphas, LAMBDA, R_LAMBDA, 
                                                            BATCH_SIZE = 10, NUM= NUM, device=device).item())
        
        log = {
            **log,
            'full_loss' : loss.item(),
            'cycle_loss_XYX' : cycle_loss.item(),
            'reg_loss' : reg_loss.item(),
            'loss_validate' : loss_validate_history[-1],
        }        
        iters.set_postfix(loss = loss.item(), loss_validate = loss_validate_history[-1])
        if it % score_freq == 0:
            full_log[it] = log 
            
    plt.plot(loss_history, label='train', color='blue')
    plt.plot(loss_validate_history, label='validate', color='red')
    plt.yscale('log')
    plt.legend()
    plt.show()

    # Transport X_train_fair and X_test_fair
    X_train_fair_transp = D.push(
        torch.tensor(X_train_fair.to_numpy()).float().to(device).requires_grad_(True),
        encoder.encode(torch.tensor(S_train_fair.to_numpy()).float().to(device).flatten())
    ).detach().cpu().numpy()

    X_test_fair_transp = D.push(
        torch.tensor(X_test_fair.to_numpy()).float().to(device).requires_grad_(True),
        encoder.encode(torch.tensor(S_test_fair.to_numpy()).float().to(device).flatten())
    ).detach().cpu().numpy()

    # # Ridge regression on unfair data
    # ridge_unfair = Ridge(alpha=alpha_ridge)
    # ridge_unfair.fit(X_train_fair, Y_train_fair)
    # Y_pred_unfair = ridge_unfair.predict(X_test_fair)
    # MSE_ridge_unfair = mean_squared_error(Y_test_fair, Y_pred_unfair)
    # KS_ridge_unfair = compute_KS(Y_pred_unfair.squeeze(), S_test_fair)
    # print("MSE on unfair data:", MSE_ridge_unfair)
    # print("KS on unfair data:", KS_ridge_unfair)

    # # Ridge regression on fair data
    # ridge_fair = Ridge(alpha=alpha_ridge)
    # ridge_fair.fit(X_train_fair_transp, Y_train_fair)
    # Y_pred_fair = ridge_fair.predict(X_test_fair_transp)
    # MSE_ridge_fair = mean_squared_error(Y_test_fair, Y_pred_fair)
    # KS_ridge_fair = compute_KS(Y_pred_fair.squeeze(), S_test_fair)
    # print("MSE on fair data:", MSE_ridge_fair)
    # print("KS on fair data:", KS_ridge_fair)

    # Random forest on unfair data
    rf_unfair = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    rf_unfair.fit(X_train_fair, Y_train_fair.squeeze())
    Y_pred_unfair = rf_unfair.predict(X_test_fair)
    MSE_rf_unfair = mean_squared_error(Y_test_fair, Y_pred_unfair)
    KS_rf_unfair = compute_KS(Y_pred_unfair.squeeze(), S_test_fair)
    print("MSE on unfair data:", MSE_rf_unfair)
    print("KS on unfair data:", KS_rf_unfair)

    # Random forest on fair data
    rf_fair = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    rf_fair.fit(X_train_fair_transp, Y_train_fair.squeeze())
    Y_pred_fair = rf_fair.predict(X_test_fair_transp)
    MSE_rf_fair = mean_squared_error(Y_test_fair, Y_pred_fair)
    KS_rf_fair = compute_KS(Y_pred_fair.squeeze(), S_test_fair)
    print("MSE on fair data:", MSE_rf_fair)
    print("KS on fair data:", KS_rf_fair)

    # # KRLS on unfair data
    # KRLS_unfair = KernelRidge(kernel='rbf', gamma=gamma_RKLS)
    # KRLS_unfair.fit(X_train_fair, Y_train_fair.squeeze())
    # Y_pred_unfair = KRLS_unfair.predict(X_test_fair)
    # MSE_KRLS_unfair = mean_squared_error(Y_test_fair, Y_pred_unfair)
    # KS_KRLS_unfair = compute_KS(Y_pred_unfair.squeeze(), S_test_fair)
    # print("MSE on unfair data:", MSE_KRLS_unfair)
    # print("KS on unfair data:", KS_KRLS_unfair)

    # # KRLS on fair data
    # KRLS_fair = KernelRidge(kernel='rbf', gamma=gamma_RKLS)
    # KRLS_fair.fit(X_train_fair_transp, Y_train_fair.squeeze())
    # Y_pred_fair = KRLS_fair.predict(X_test_fair_transp)
    # MSE_KRLS_fair = mean_squared_error(Y_test_fair, Y_pred_fair)
    # KS_KRLS_fair = compute_KS(Y_pred_fair.squeeze(), S_test_fair)
    # print("MSE on fair data:", mean_squared_error(Y_test_fair, Y_pred_fair))
    # print("KS on fair data:", compute_KS(Y_pred_fair.squeeze(), S_test_fair))

    # # save log 
    # if OUTPUT_FOLDER is not None:
    #     pd.DataFrame.from_dict(full_log).T.to_csv(os.path.join(OUTPUT_FOLDER, NAME + '.csv'))

    return full_log, MSE_rf_unfair, KS_rf_unfair, MSE_rf_fair, KS_rf_fair