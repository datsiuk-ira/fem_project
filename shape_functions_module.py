# shape_functions_module.py
import numpy as np
from constants import (NODE_LOCAL_COORDS_20_HEX, NODE_LOCAL_COORDS_8_QUAD_FACE,
                        GAUSS_POINTS_1D_3, GAUSS_WEIGHTS_1D_3,
                        NUM_GAUSS_POINTS_1D, NUM_GAUSS_POINTS_2D, NUM_GAUSS_POINTS_3D)


# --- 3D Форм-функції для гексаедра (без змін) ---
def get_shape_functions_20_node_hex(alpha: float, beta: float, gamma: float) -> np.ndarray:
    # ... (код з попереднього кроку без змін) ...
    phi = np.zeros(20)
    coords = NODE_LOCAL_COORDS_20_HEX
    for i in range(8):  # Кутові
        ai, bi, gi = coords[i, 0], coords[i, 1], coords[i, 2]
        term_sum = alpha * ai + beta * bi + gamma * gi
        phi[i] = 0.125 * (1 + alpha * ai) * (1 + beta * bi) * (1 + gamma * gi) * (term_sum - 2)
    for i in range(8, 20):  # Серединні
        ai, bi, gi = coords[i, 0], coords[i, 1], coords[i, 2]
        phi_val_temp = 0.25 * (1 + alpha * ai) * (1 + beta * bi) * (1 + gamma * gi)
        if abs(ai) < 1e-9:
            phi_val_temp *= (1 - alpha ** 2 * bi ** 2 * gi ** 2)
        elif abs(bi) < 1e-9:
            phi_val_temp *= (1 - beta ** 2 * ai ** 2 * gi ** 2)
        elif abs(gi) < 1e-9:
            phi_val_temp *= (1 - gamma ** 2 * ai ** 2 * bi ** 2)
        phi[i] = phi_val_temp
    return phi


def get_shape_function_derivatives_20_node_hex(alpha: float, beta: float, gamma: float) -> np.ndarray:
    # ... (код з попереднього кроку без змін) ...
    d_phi_d_local = np.zeros((20, 3))
    coords = NODE_LOCAL_COORDS_20_HEX
    for i in range(8):  # Кутові
        ai, bi, gi = coords[i, 0], coords[i, 1], coords[i, 2]
        d_phi_d_local[i, 0] = 0.125 * ai * (1 + beta * bi) * (1 + gamma * gi) * (
                    2 * alpha * ai + beta * bi + gamma * gi - 1)
        d_phi_d_local[i, 1] = 0.125 * bi * (1 + alpha * ai) * (1 + gamma * gi) * (
                    alpha * ai + 2 * beta * bi + gamma * gi - 1)
        d_phi_d_local[i, 2] = 0.125 * gi * (1 + alpha * ai) * (1 + beta * bi) * (
                    alpha * ai + beta * bi + 2 * gamma * gi - 1)
    for i in range(8, 20):  # Серединні
        ai, bi, gi = coords[i, 0], coords[i, 1], coords[i, 2]
        term_alpha_deriv = ai * (
                    1 - (alpha * bi * gi) ** 2 - (beta * ai * gi) ** 2 - (gamma * ai * bi) ** 2) - 2 * alpha * (
                                       1 + alpha * ai) * (bi ** 2 * gi ** 2)
        d_phi_d_local[i, 0] = 0.25 * (1 + beta * bi) * (1 + gamma * gi) * term_alpha_deriv
        term_beta_deriv = bi * (
                    1 - (alpha * bi * gi) ** 2 - (beta * ai * gi) ** 2 - (gamma * ai * bi) ** 2) - 2 * beta * (
                                      1 + beta * bi) * (ai ** 2 * gi ** 2)
        d_phi_d_local[i, 1] = 0.25 * (1 + alpha * ai) * (1 + gamma * gi) * term_beta_deriv
        term_gamma_deriv = gi * (
                    1 - (alpha * bi * gi) ** 2 - (beta * ai * gi) ** 2 - (gamma * ai * bi) ** 2) - 2 * gamma * (
                                       1 + gamma * gi) * (ai ** 2 * bi ** 2)
        d_phi_d_local[i, 2] = 0.25 * (1 + alpha * ai) * (1 + beta * bi) * term_gamma_deriv
    return d_phi_d_local


def get_gauss_points_3d():
    gp_coords = np.zeros((NUM_GAUSS_POINTS_3D, 3))
    gp_weights = np.zeros(NUM_GAUSS_POINTS_3D)
    idx = 0
    for i in range(NUM_GAUSS_POINTS_1D):
        for j in range(NUM_GAUSS_POINTS_1D):
            for k in range(NUM_GAUSS_POINTS_1D):
                gp_coords[idx, :] = [GAUSS_POINTS_1D_3[k], GAUSS_POINTS_1D_3[j], GAUSS_POINTS_1D_3[i]]
                gp_weights[idx] = GAUSS_WEIGHTS_1D_3[k] * GAUSS_WEIGHTS_1D_3[j] * GAUSS_WEIGHTS_1D_3[i]
                idx += 1
    return gp_coords, gp_weights


def precompute_dphi_at_gauss_points():
    gp_coords, _ = get_gauss_points_3d()
    DFIABG = np.zeros((NUM_GAUSS_POINTS_3D, 3, 20))
    for gp_idx, (alpha, beta, gamma) in enumerate(gp_coords):
        d_phi_matrix = get_shape_function_derivatives_20_node_hex(alpha, beta, gamma)
        DFIABG[gp_idx, 0, :] = d_phi_matrix[:, 0]
        DFIABG[gp_idx, 1, :] = d_phi_matrix[:, 1]
        DFIABG[gp_idx, 2, :] = d_phi_matrix[:, 2]
    return DFIABG


DFIABG_PRECOMPUTED = precompute_dphi_at_gauss_points()


# --- 2D Форм-функції для граней (8-вузловий квадрат) ---
def get_shape_functions_8_node_quad(eta: float, tau: float) -> np.ndarray:
    """
    Обчислює значення 8 форм-функцій (psi_i) для квадратного елемента
    в точці з локальними координатами (eta, tau).
    Формули з Лабораторного практикуму, стор. 19, рівняння (33).
    """
    psi = np.zeros(8)
    coords_face = NODE_LOCAL_COORDS_8_QUAD_FACE  # (etai, taui) для 8 вузлів

    # Кутові вузли (i = 0..3, що відповідає вузлам 1..4 в PDF)
    for i in range(4):
        ei, ti = coords_face[i, 0], coords_face[i, 1]
        psi[i] = 0.25 * (1 + eta * ei) * (1 + tau * ti) * (eta * ei + tau * ti - 1)

    # Середини ребер (i = 4..7, що відповідає вузлам 5..8 в PDF)
    # Вузол 4 (PDF 5): ei=0, ti=-1 (між 0-1)
    ei, ti = coords_face[4, 0], coords_face[4, 1]  # (0, -1)
    psi[4] = 0.5 * (1 - eta ** 2) * (1 + tau * ti)
    # Вузол 5 (PDF 6): ei=1, ti=0 (між 1-2)
    ei, ti = coords_face[5, 0], coords_face[5, 1]  # (1, 0)
    psi[5] = 0.5 * (1 - tau ** 2) * (1 + eta * ei)
    # Вузол 6 (PDF 7): ei=0, ti=1 (між 2-3)
    ei, ti = coords_face[6, 0], coords_face[6, 1]  # (0, 1)
    psi[6] = 0.5 * (1 - eta ** 2) * (1 + tau * ti)
    # Вузол 7 (PDF 8): ei=-1, ti=0 (між 3-0)
    ei, ti = coords_face[7, 0], coords_face[7, 1]  # (-1, 0)
    psi[7] = 0.5 * (1 - tau ** 2) * (1 + eta * ei)

    return psi


def get_shape_function_derivatives_8_node_quad(eta: float, tau: float) -> np.ndarray:
    """
    Обчислює похідні 8 форм-функцій (dpsi/d_eta, dpsi/d_tau) для квадратного елемента.
    Повертає масив розміром (8, 2).
    """
    d_psi_d_local = np.zeros((8, 2))
    coords_face = NODE_LOCAL_COORDS_8_QUAD_FACE

    # Кутові вузли (i = 0..3)
    for i in range(4):
        ei, ti = coords_face[i, 0], coords_face[i, 1]
        # dpsi_i / d_eta
        d_psi_d_local[i, 0] = 0.25 * ei * (1 + tau * ti) * (2 * eta * ei + tau * ti)
        # dpsi_i / d_tau
        d_psi_d_local[i, 1] = 0.25 * ti * (1 + eta * ei) * (eta * ei + 2 * tau * ti)

    # Середини ребер (i = 4..7)
    # Вузол 4 (PDF 5): ei=0, ti=-1
    ei4, ti4 = coords_face[4, 0], coords_face[4, 1]
    d_psi_d_local[4, 0] = 0.5 * (-2 * eta) * (1 + tau * ti4)  # d/d_eta
    d_psi_d_local[4, 1] = 0.5 * (1 - eta ** 2) * ti4  # d/d_tau

    # Вузол 5 (PDF 6): ei=1, ti=0
    ei5, ti5 = coords_face[5, 0], coords_face[5, 1]
    d_psi_d_local[5, 0] = 0.5 * (1 - tau ** 2) * ei5  # d/d_eta
    d_psi_d_local[5, 1] = 0.5 * (-2 * tau) * (1 + eta * ei5)  # d/d_tau

    # Вузол 6 (PDF 7): ei=0, ti=1
    ei6, ti6 = coords_face[6, 0], coords_face[6, 1]
    d_psi_d_local[6, 0] = 0.5 * (-2 * eta) * (1 + tau * ti6)  # d/d_eta
    d_psi_d_local[6, 1] = 0.5 * (1 - eta ** 2) * ti6  # d/d_tau

    # Вузол 7 (PDF 8): ei=-1, ti=0
    ei7, ti7 = coords_face[7, 0], coords_face[7, 1]
    d_psi_d_local[7, 0] = 0.5 * (1 - tau ** 2) * ei7  # d/d_eta
    d_psi_d_local[7, 1] = 0.5 * (-2 * tau) * (1 + eta * ei7)  # d/d_tau

    return d_psi_d_local


def get_gauss_points_2d():
    """Генерує 2D точки Гаусса та їх ваги (напр. 3x3=9 точок)."""
    gp_coords_2d = np.zeros((NUM_GAUSS_POINTS_2D, 2))  # (eta, tau)
    gp_weights_2d = np.zeros(NUM_GAUSS_POINTS_2D)
    idx = 0
    for i in range(NUM_GAUSS_POINTS_1D):  # tau
        for j in range(NUM_GAUSS_POINTS_1D):  # eta
            gp_coords_2d[idx, :] = [GAUSS_POINTS_1D_3[j], GAUSS_POINTS_1D_3[i]]
            gp_weights_2d[idx] = GAUSS_WEIGHTS_1D_3[j] * GAUSS_WEIGHTS_1D_3[i]
            idx += 1
    return gp_coords_2d, gp_weights_2d


def precompute_dpsi_at_gauss_points():
    """
    Попередньо обчислює похідні 2D форм-функцій (dpsi/d_eta, dpsi/d_tau)
    в усіх 2D точках Гаусса. DPSITE.
    PDF: DPSITE[9,2,8] -> [ТГ, по якій локальній координаті (eta/tau), номер функції psi]
    Повертає:
        DPSITE (np.ndarray): Масив (NUM_GAUSS_POINTS_2D, 2_local_coords, 8_shape_functions).
    """
    gp_coords_2d, _ = get_gauss_points_2d()
    DPSITE = np.zeros((NUM_GAUSS_POINTS_2D, 2, 8))

    for gp_idx, (eta, tau) in enumerate(gp_coords_2d):
        d_psi_matrix = get_shape_function_derivatives_8_node_quad(eta, tau)  # Розмір (8, 2)
        DPSITE[gp_idx, 0, :] = d_psi_matrix[:, 0]  # dpsi/d_eta для всіх 8 функцій
        DPSITE[gp_idx, 1, :] = d_psi_matrix[:, 1]  # dpsi/d_tau для всіх 8 функцій

    return DPSITE


DPSITE_PRECOMPUTED = precompute_dpsi_at_gauss_points()