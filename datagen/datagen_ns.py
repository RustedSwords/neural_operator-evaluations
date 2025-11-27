"""
datdagen_ns.py

Produces pseudo-spectral 2D Navier-Stokes vorticity datasets in .mat format
suitable for the FNO3D.py loader (field name 'u', shape (N, 64, 64, 50)).

Usage: edit parameters in main() and run with Python 3.8+.

Dependencies:
    numpy, scipy

Outputs:
    ns_data_V{viscid_str}_N{N}_T50_{idx}.mat  (one or multiple files, each contains 'u')
"""

import numpy as np
import scipy.io
import os

# ---------------------------
# Solver utilities
# ---------------------------

def build_wavenumbers(Nx, Ny, Lx=1.0, Ly=1.0):
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    kxv = kx.reshape(Nx, 1)
    kyv = ky.reshape(1, Ny)
    kx2 = kxv * 1.0
    ky2 = kyv * 1.0
    k2 = kx2**2 + ky2**2
    return kx2, ky2, k2

def dealias_mask(Nx, Ny, pad_factor=3/2):
    # 2/3-rule: keep modes with |k| < k_max * 2/3
    kx = np.fft.fftfreq(Nx) * Nx
    ky = np.fft.fftfreq(Ny) * Ny
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    kabs = np.sqrt(kx_grid**2 + ky_grid**2)
    kmax = max(Nx, Ny) / 2.0
    mask = (kabs <= (2.0/3.0) * kmax).astype(float)
    return mask

def spectral_poisson_solve(omega_hat, k2):
    # Solve -Δ ψ = ω  =>  in Fourier space: psi_hat = - omega_hat / k2 , handle k2=0
    psi_hat = np.zeros_like(omega_hat, dtype=np.complex128)
    mask = (k2 != 0)
    psi_hat[mask] = - omega_hat[mask] / k2[mask]
    psi_hat[~mask] = 0.0 + 0.0j
    return psi_hat

def compute_velocity_from_psi_hat(psi_hat, kx, ky):
    # u = dψ/dy, v = -dψ/dx  (remember spectral derivative: d/dx -> i kx)
    ux_hat = 1j * ky * psi_hat    # d/dy psi -> multiply by i*ky
    uy_hat = -1j * kx * psi_hat   # -d/dx psi -> multiply by -i*kx
    u = np.fft.ifft2(ux_hat).real
    v = np.fft.ifft2(uy_hat).real
    return u, v

def spectral_laplacian_hat(omega_hat, k2):
    return -k2 * omega_hat

# ---------------------------
# Initial condition generator
# ---------------------------

def random_truncated_fourier_vorticity(Nx, Ny, modes=8, amplitude=1.0, seed=None):
    """
    Generate a smooth, mean-zero random vorticity field by sampling
    random Fourier coefficients on low modes (|kx|,|ky| <= modes).
    Returns real-space vorticity on Nx x Ny grid.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    hat = np.zeros((Nx, Ny), dtype=np.complex128)

    # center frequencies (k=0..Nx-1 map) using numpy fft ordering
    for kx in range(-modes, modes+1):
        for ky in range(-modes, modes+1):
            # skip the (0,0) mean mode to enforce mean-zero vorticity
            if kx == 0 and ky == 0:
                continue
            amp = amplitude * np.exp(-0.5 * (kx**2 + ky**2) / (modes**2))  # taper high modes
            phase = rng.uniform(0, 2*np.pi)
            # map negative freq indices to fft layout
            ix = kx % Nx
            iy = ky % Ny
            coeff = amp * (rng.normal() + 1j * rng.normal()) * np.exp(1j * phase)
            hat[ix, iy] = coeff

    # ensure reality: set conjugate symmetric entries
    # (but above we filled symmetric positions implicitly with modulo)
    omega = np.fft.ifft2(hat).real
    return omega

# ---------------------------
# Time-stepping: RK4 for vorticity eqn
# ---------------------------

def rhs_vorticity(omega, kx, ky, k2, nu, dealias_mask_arr):
    """
    Compute time derivative of vorticity: ω_t = - u·∇ω + ν Δ ω
    omega: real-space vorticity (Nx,Ny)
    returns rhs in real space (Nx,Ny)
    """
    # Fourier transform of omega
    omega_hat = np.fft.fft2(omega)

    # streamfunction
    psi_hat = spectral_poisson_solve(omega_hat, k2)

    # get velocity in physical space
    u, v = compute_velocity_from_psi_hat(psi_hat, kx, ky)

    # compute gradients of omega in spectral space (dealiased)
    # grad_x omega:
    domega_dx_hat = 1j * kx * omega_hat
    domega_dy_hat = 1j * ky * omega_hat

    # apply dealiasing to nonlinear convective term by truncating high modes
    domega_dx = np.fft.ifft2(domega_dx_hat * dealias_mask_arr).real
    domega_dy = np.fft.ifft2(domega_dy_hat * dealias_mask_arr).real

    conv = u * domega_dx + v * domega_dy

    # diffusion term (real-space form via spectral multiply)
    lap_omega_hat = spectral_laplacian_hat(omega_hat, k2)
    lap_omega = np.fft.ifft2(lap_omega_hat).real

    rhs = - conv + nu * lap_omega
    return rhs

def rk4_step(omega, dt, kx, ky, k2, nu, dealias_mask_arr):
    k1 = rhs_vorticity(omega, kx, ky, k2, nu, dealias_mask_arr)
    k2r = rhs_vorticity(omega + 0.5*dt*k1, kx, ky, k2, nu, dealias_mask_arr)
    k3 = rhs_vorticity(omega + 0.5*dt*k2r, kx, ky, k2, nu, dealias_mask_arr)
    k4 = rhs_vorticity(omega + dt*k3, kx, ky, k2, nu, dealias_mask_arr)
    return omega + (dt/6.0) * (k1 + 2*k2r + 2*k3 + k4)

# ---------------------------
# Dataset generator
# ---------------------------

def generate_dataset(N, Nx=64, Ny=64, T_total=50, nu=1e-4, dt=0.01,
                     modes_ic=8, ic_amplitude=2.0, seed_base=0, out_path='.', name_prefix=None):
    """
    Generate N trajectories of vorticity fields (Nx x Ny grid, T_total timesteps)
    and save into a .mat under key 'u' with shape (N, Nx, Ny, T_total).
    """
    assert T_total >= 1
    if name_prefix is None:
        name_prefix = f'ns_data_V{int(1/nu) if nu<1 else int(nu)}_N{N}_T{T_total}'

    # prepare wavenumbers (spectral grid)
    kx, ky, k2 = build_wavenumbers(Nx, Ny)
    dealias_mask_arr = dealias_mask(Nx, Ny)

    data = np.zeros((N, Nx, Ny, T_total), dtype=np.float32)

    rng = np.random.RandomState(seed_base)

    for n in range(N):
        seed = seed_base + n
        omega0 = random_truncated_fourier_vorticity(Nx, Ny, modes=modes_ic, amplitude=ic_amplitude, seed=seed)
        # normalize initial vorticity amplitude a bit (makes datasets similar scale)
        sigma0 = np.std(omega0)
        if sigma0 > 0:
            omega0 = omega0 / sigma0 * ic_amplitude * 0.5

        omega = omega0.copy()
        data[n, :, :, 0] = omega.astype(np.float32)

        for t in range(1, T_total):
            omega = rk4_step(omega, dt, kx, ky, k2, nu, dealias_mask_arr)
            data[n, :, :, t] = omega.astype(np.float32)

        if (n+1) % 50 == 0 or (n+1) == N:
            print(f'Generated {n+1}/{N} trajectories')

    # ensure output directory exists
    os.makedirs(out_path, exist_ok=True)
    vis_str = f'{nu:.0e}'.replace('-', 'm').replace('+','p')
    filename = os.path.join(out_path, f'ns_data_V{vis_str}_N{N}_T{T_total}.mat')
    scipy.io.savemat(filename, {'u': data})
    print(f'wrote {filename}  (u shape = {data.shape})')
    return filename

# ---------------------------
# Example main: generate train/test splits / multiple viscosities
# ---------------------------

def main():
    out_dir = './data'
    Nx = Ny = 64
    T_total = 50
    dt = 0.01            # time-step for RK4; you can reduce for more accuracy
    modes_ic = 8
    ic_amplitude = 2.0

    # choose viscosities and dataset sizes (matches your earlier message options)
    configs = [
        # (viscosity, num_samples)
        (1e-3, 1000),
        (1e-4, 10000),   # large dataset authors reported good results with this
        # (1e-4, 1000),  # small dataset configuration (insufficient)
        # (1e-5, 1000),
    ]

    # Generate each as a separate file
    seed_base = 0
    for nu, N in configs:
        print(f'Generating nu={nu}, N={N}')
        generate_dataset(N, Nx=Nx, Ny=Ny, T_total=T_total, nu=nu, dt=dt,
                         modes_ic=modes_ic, ic_amplitude=ic_amplitude,
                         seed_base=seed_base, out_path=out_dir)
        seed_base += N + 7  # bump base seed to vary between files

if __name__ == '__main__':
    main()