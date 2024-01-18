def mvmd(signal, alpha, tau, K, DC, init, tol, max_N):

    import torch

    # Period and sampling frequency of input signal
    C, T = signal.shape # T:length of signal C:  channel number
    fs = 1 / float(T)

    # extend the signal by mirroring
    # T = save_T
    # print(T)
    f_mirror = torch.zeros(C, 2*T).to(signal.device)
    #print(f_mirror)
    f_mirror[:,0:T//2] = torch.flip(signal[:,0:T//2], dims=[-1]) 
    # print(f_mirror)
    f_mirror[:,T//2:3*T//2] = signal
    # print(f_mirror)
    f_mirror[:,3*T//2:2*T] = torch.flip(signal[:,T//2:], dims=[-1])
    # print(f_mirror)
    f = f_mirror


    # Time Domain 0 to T (of mirrored signal)
    T = float(f.shape[1])
    # print(T)
    t = torch.linspace(1/float(T), 1, int(T)).to(signal.device)
    # print(t)

    # Spectral Domain discretization
    freqs = t - 0.5 - 1/T

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    N = max_N

    # For future generalizations: individual alpha for each mode
    Alpha = alpha * torch.ones(K, dtype=torch.cfloat).to(signal.device)

    # Construct and center f_hat
    f_hat = torch.fft.fftshift(torch.fft.fft(f))
    f_hat_plus = f_hat
    f_hat_plus[:, 0:int(int(T)/2)] = 0

    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = torch.zeros((N, len(freqs), K, C), dtype=torch.cfloat).to(signal.device)

    # Initialization of omega_k
    omega_plus = torch.zeros((N, K), dtype=torch.cfloat).to(signal.device)
                        
    if (init == 1):
        for i in range(1, K+1):
            omega_plus[0,i-1] = (0.5/K)*(i-1)
    elif (init==2):
        omega_plus[0,:] = torch.sort(torch.exp(torch.log(fs)) +
        (torch.log(0.5) - torch.log(fs)) * torch.random.rand(1, K))
    else:
        omega_plus[0,:] = 0

    if (DC):
        omega_plus[0,0] = 0


    # start with empty dual variables
    lamda_hat = torch.zeros((N, len(freqs), C), dtype=torch.cfloat).to(signal.device)

    # other inits
    uDiff = tol+2.2204e-16 #updata step
    n = 1 #loop counter
    sum_uk = torch.zeros((len(freqs), C)).to(signal.device) #accumulator

    T = int(T)

    # ----------- Main loop for iterative updates

    while uDiff > tol and n < N:
        # update first mode accumulator
        k = 1
        sum_uk = u_hat_plus[n-1,:,K-1,:] + sum_uk - u_hat_plus[n-1,:,0,:]

        #update spectrum of first mode through Wiener filter of residuals
        f_hat_plus_transposed = f_hat_plus.T.to(signal.device)
        numerator = f_hat_plus_transposed - sum_uk - lamda_hat[n - 1, :, :] / 2
        denominator = 1 + Alpha[k - 1] * torch.square(freqs.unsqueeze(1) - omega_plus[n - 1, k - 1])
        u_hat_plus[n, :, k - 1, :] = numerator / denominator
   
        #update first omega if not held at 0
        if DC == False:
            omega_plus[n,k-1] = torch.sum(torch.mm(freqs[T//2:T].unsqueeze(0), 
                            torch.square(torch.abs(u_hat_plus[n,T//2:T,k-1,:])))) \
            / torch.sum(torch.square(torch.abs(u_hat_plus[n,T//2:T,k-1,:])))


        for k in range(2, K+1):

            #accumulator
            sum_uk = u_hat_plus[n,:,k-2,:] + sum_uk - u_hat_plus[n-1,:,k-1,:]

            #mode spectrum
            f_hat_plus_transposed = f_hat_plus.T
            numerator = f_hat_plus_transposed - sum_uk - lamda_hat[n - 1, :, :] / 2
            denominator = 1 + Alpha[k - 1] * torch.square(freqs.unsqueeze(1) - omega_plus[n - 1, k - 1])
            u_hat_plus[n, :, k - 1, :] = numerator / denominator

            
            #center frequencies
            omega_plus[n,k-1] = torch.sum(torch.mm(freqs[T//2:T].unsqueeze(0),
                torch.square(torch.abs(u_hat_plus[n,T//2:T,k-1,:])))) \
                /  torch.sum(torch.square(torch.abs(u_hat_plus[n,T//2:T:,k-1,:])))

        #Dual ascent
        lamda_hat[n,:,:] = lamda_hat[n-1,:,:]

        #loop counter
        n = n + 1

        #converged yet?
        uDiff = 2.2204e-16

        for i in range(1, K+1):
            uDiff = uDiff+1 / float(T) * torch.mm(u_hat_plus[n-1,:,i-1,:] - u_hat_plus[n-2,:,i-1,:],
                                                ((u_hat_plus[n-1,:,i-1,:]-u_hat_plus[n-2,:,i-1,:]).conj()).conj().T)

        uDiff = torch.sum(torch.abs(uDiff))

        
    # ------ Postprocessing and cleanup

    # discard empty space if converged early

    N = min(N, n)
    # omega = omega_plus[0:N,:]

    # Signal reconstruction
    u_hat = torch.zeros((T, K, C), dtype=torch.cfloat).to(signal.device)

    # 批量处理：直接对整个张量操作
    u_hat[T // 2:T, :, :] = (u_hat_plus[N - 1, T // 2:T, :, :])

    # 创建逆序索引并应用
    second_index = list(range(1, T // 2 + 1))
    second_index.reverse()
    u_hat[second_index, :, :] = (torch.conj(u_hat_plus[N - 1, T // 2:T, :, :]))

    # 对第0个元素进行特殊处理
    u_hat[0, :, :] = torch.conj(u_hat[-1, :, :])


    u = torch.fft.ifft(torch.fft.ifftshift(u_hat, dim=0), dim=0).real
    u = u.permute(1, 0, 2)
    u = u[:, T // 4:3 * T // 4, :]
    u = torch.fft.ifftshift(u, dim=-1)
    return u