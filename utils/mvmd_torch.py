import torch

def mvmd(signal, alpha, tau, K, DC, init, tol, max_N):

    # T:length of signal C:  channel number
    C, T = signal.shape
    fs = 1 / float(T)

    # extend the signal by mirroring
    f_mirror = torch.zeros(C, 2*T).to(signal.device)
    f_mirror[:,0:T//2] = torch.flip(signal[:,0:T//2], dims=[-1])
    f_mirror[:,T//2:3*T//2] = signal
    f_mirror[:,3*T//2:2*T] = torch.flip(signal[:,T//2:], dims=[-1])
    f = f_mirror


    # Time Domain 0 to T
    T = float(f.shape[1])
    t = torch.linspace(1/float(T), 1, int(T)).to(signal.device)


    # Spectral Domain discretization
    freqs = t - 0.5 - 1/T

    # Maximum number of iterations
    N = max_N

    Alpha = alpha * torch.ones(K, dtype=torch.cfloat).to(signal.device)

    # Construct and center f_hat
    f_hat = torch.fft.fftshift(torch.fft.fft(f))
    f_hat_plus = f_hat
    f_hat_plus[:, 0:int(int(T)/2)] = 0

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

    # update step
    uDiff = tol+2.2204e-16
    # counter
    n = 1
    # accumulator
    sum_uk = torch.zeros((len(freqs), C)).to(signal.device)

    T = int(T)

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

        lamda_hat[n,:,:] = lamda_hat[n-1,:,:]

        #counter
        n = n + 1

        #converged yet?
        uDiff = 2.2204e-16

        for i in range(1, K+1):
            uDiff = uDiff+1 / float(T) * torch.mm(u_hat_plus[n-1,:,i-1,:] - u_hat_plus[n-2,:,i-1,:],
                                                ((u_hat_plus[n-1,:,i-1,:]-u_hat_plus[n-2,:,i-1,:]).conj()).conj().T)

        uDiff = torch.sum(torch.abs(uDiff))

    N = min(N, n)

    # Signal reconstruction
    u_hat = torch.zeros((T, K, C), dtype=torch.cfloat).to(signal.device)

    u_hat[T // 2:T, :, :] = (u_hat_plus[N - 1, T // 2:T, :, :])

    second_index = list(range(1, T // 2 + 1))
    second_index.reverse()
    u_hat[second_index, :, :] = (torch.conj(u_hat_plus[N - 1, T // 2:T, :, :]))

    u_hat[0, :, :] = torch.conj(u_hat[-1, :, :])


    u = torch.fft.ifft(torch.fft.ifftshift(u_hat, dim=0), dim=0).real
    u = u.permute(1, 0, 2)
    u = u[:, T // 4:3 * T // 4, :]
    u = torch.fft.ifftshift(u, dim=-1)

    return u