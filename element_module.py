# element_module.py
import numpy as np
from constants import (LAMBDA_DOC, MU_DOC, POISSON_RATIO, NUM_GAUSS_POINTS_3D,
                        HEX_FACE_DEFINITIONS, DEFAULT_PRESSURE, DEFAULT_LOAD_FACE_ID)
from shape_functions_module import (DFIABG_PRECOMPUTED, DPSITE_PRECOMPUTED,
                                     get_gauss_points_3d, get_gauss_points_2d,
                                     get_shape_functions_8_node_quad)


class Hexa20ElementCalculator:
    """Клас для обчислень на рівні одного 20-вузлового гексаедрального елемента."""

    def __init__(self, element_node_coords_global: np.ndarray):
        self.element_node_coords_global = element_node_coords_global  # (20, 3) - X, Y, Z глоб. координати вузлів елемента
        self.dphi_d_local_at_gp = DFIABG_PRECOMPUTED
        self.gauss_points_3d, self.gauss_weights_3d = get_gauss_points_3d()

        self.dpsi_d_local_at_gp_face = DPSITE_PRECOMPUTED  # (NUM_GP_2D, 2, 8)
        self.gauss_points_2d_face, self.gauss_weights_2d_face = get_gauss_points_2d()

        self.DXYZABG = np.zeros((NUM_GAUSS_POINTS_3D, 3, 3))
        self.DJ = np.zeros(NUM_GAUSS_POINTS_3D)
        self.DFIXYZ = np.zeros((NUM_GAUSS_POINTS_3D, 20, 3))
        self.MGE = np.zeros((60, 60))
        self.FE = np.zeros(60)  # Елементний вектор навантаження

        self._calculate_jacobians_and_derivatives()
        self._calculate_stiffness_matrix()
        # За замовчуванням вектор навантаження обчислюється для стандартної грані та тиску
        self.calculate_load_vector(face_id=DEFAULT_LOAD_FACE_ID, pressure=DEFAULT_PRESSURE)

    def _calculate_jacobians_and_derivatives(self):
        # ... (код з попереднього кроку без змін) ...
        for gp_idx in range(NUM_GAUSS_POINTS_3D):
            dphi_d_local_at_this_gp = self.dphi_d_local_at_gp[gp_idx, :, :]
            jacobian_matrix = np.zeros((3, 3))
            for i_global in range(3):
                for j_local in range(3):
                    sum_val = 0
                    for k_node in range(20):
                        sum_val += dphi_d_local_at_this_gp[j_local, k_node] * self.element_node_coords_global[
                            k_node, i_global]
                    jacobian_matrix[i_global, j_local] = sum_val
            self.DXYZABG[gp_idx, :, :] = jacobian_matrix
            det_J = np.linalg.det(jacobian_matrix)
            if det_J <= 1e-9:
                # print(f"Попередження: Визначник Якобіана <= 0 ({det_J:.2e}) в ТГ {gp_idx}.") # Забагато логів
                pass
            self.DJ[gp_idx] = det_J
            try:
                inv_jacobian_matrix = np.linalg.inv(jacobian_matrix)
            except np.linalg.LinAlgError:
                self.DFIXYZ[gp_idx, :, :] = np.nan
                continue
            dphi_d_local_at_this_gp_T = dphi_d_local_at_this_gp.T
            self.DFIXYZ[gp_idx, :, :] = dphi_d_local_at_this_gp_T @ inv_jacobian_matrix.T

    def _calculate_stiffness_matrix(self):
        # ... (код з попереднього кроку без змін) ...
        lambda_val, mu_val, nu_val = LAMBDA_DOC, MU_DOC, POISSON_RATIO
        for gp_idx in range(NUM_GAUSS_POINTS_3D):
            dphi_dx = self.DFIXYZ[gp_idx, :, :]
            det_J = self.DJ[gp_idx]
            weight = self.gauss_weights_3d[gp_idx]
            if np.isnan(det_J) or det_J <= 1e-9 or np.isnan(dphi_dx).any(): continue
            for i_node in range(20):
                for j_node in range(20):
                    a11_ij = lambda_val * (1 - nu_val) * (dphi_dx[i_node, 0] * dphi_dx[j_node, 0]) + mu_val * (
                                dphi_dx[i_node, 1] * dphi_dx[j_node, 1] + dphi_dx[i_node, 2] * dphi_dx[j_node, 2])
                    a22_ij = lambda_val * (1 - nu_val) * (dphi_dx[i_node, 1] * dphi_dx[j_node, 1]) + mu_val * (
                                dphi_dx[i_node, 0] * dphi_dx[j_node, 0] + dphi_dx[i_node, 2] * dphi_dx[j_node, 2])
                    a33_ij = lambda_val * (1 - nu_val) * (dphi_dx[i_node, 2] * dphi_dx[j_node, 2]) + mu_val * (
                                dphi_dx[i_node, 0] * dphi_dx[j_node, 0] + dphi_dx[i_node, 1] * dphi_dx[j_node, 1])
                    a12_ij = lambda_val * nu_val * (dphi_dx[i_node, 0] * dphi_dx[j_node, 1]) + mu_val * (
                                dphi_dx[i_node, 1] * dphi_dx[j_node, 0])
                    a13_ij = lambda_val * nu_val * (dphi_dx[i_node, 0] * dphi_dx[j_node, 2]) + mu_val * (
                                dphi_dx[i_node, 2] * dphi_dx[j_node, 0])
                    a23_ij = lambda_val * nu_val * (dphi_dx[i_node, 1] * dphi_dx[j_node, 2]) + mu_val * (
                                dphi_dx[i_node, 2] * dphi_dx[j_node, 1])
                    term_common = weight * det_J
                    self.MGE[i_node, j_node] += a11_ij * term_common
                    self.MGE[i_node + 20, j_node + 20] += a22_ij * term_common
                    self.MGE[i_node + 40, j_node + 40] += a33_ij * term_common
                    self.MGE[i_node, j_node + 20] += a12_ij * term_common
                    self.MGE[
                        i_node + 20, j_node] += a12_ij * term_common  # a21_ij(i,j) = a12_ij(j,i) -> M_yx = M_xy^T -> symmetric term usage
                    self.MGE[i_node, j_node + 40] += a13_ij * term_common
                    self.MGE[i_node + 40, j_node] += a13_ij * term_common
                    self.MGE[i_node + 20, j_node + 40] += a23_ij * term_common
                    self.MGE[i_node + 40, j_node + 20] += a23_ij * term_common

    def calculate_load_vector(self, face_id: int, pressure: float):
        """
        Обчислює елементний вектор навантаження FE для заданої грані та тиску.
        Args:
            face_id (int): ID грані (1-6) з HEX_FACE_DEFINITIONS.
            pressure (float): Значення тиску (позитивне = назовні).
        """
        self.FE.fill(0)  # Очищення попереднього вектора навантаження
        if face_id not in HEX_FACE_DEFINITIONS:
            print(f"Помилка: Неправильний ID грані {face_id}")
            return

        face_def = HEX_FACE_DEFINITIONS[face_id]
        hex_face_node_indices = face_def["hex_node_indices"]  # Гексагональні індекси 8 вузлів на цій грані

        # Глобальні координати 8 вузлів на цій грані елемента
        face_nodes_global_coords = self.element_node_coords_global[hex_face_node_indices, :]  # (8, 3)

        # Інтегрування по точках Гаусса на поверхні
        for gp_idx_face in range(len(self.gauss_points_2d_face)):
            eta, tau = self.gauss_points_2d_face[gp_idx_face]
            weight_face = self.gauss_weights_2d_face[gp_idx_face]

            # Значення 2D форм-функцій psi_i в поточній ТГ на грані
            psi_values_at_gp = get_shape_functions_8_node_quad(eta, tau)  # (8,)

            # Похідні dpsi/d_eta, dpsi/d_tau в поточній ТГ на грані
            # dpsi_d_local_face_at_gp має розмір (2_local_face_coords, 8_shape_funcs)
            dpsi_d_local_face_at_gp = self.dpsi_d_local_at_gp_face[gp_idx_face, :, :]

            # Обчислення Якобіана поверхні J_surf = [dx/d_eta, dy/d_eta, dz/d_eta; dx/d_tau, dy/d_tau, dz/d_tau]
            # J_surf[global_coord_idx (x,y,z), face_local_coord_idx (eta,tau)]
            jacobian_surface = np.zeros((3, 2))  # (x,y,z) x (eta,tau)
            for i_global in range(3):  # x,y,z
                for j_face_local in range(2):  # eta, tau
                    sum_val = 0
                    for k_face_node in range(8):  # по 8 вузлах грані
                        # dpsi_d_local_face_at_gp[j_face_local, k_face_node] - це d(psi_k)/d(local_face_j)
                        # face_nodes_global_coords[k_face_node, i_global] - це global_i координата k-го вузла грані
                        sum_val += dpsi_d_local_face_at_gp[j_face_local, k_face_node] * \
                                   face_nodes_global_coords[k_face_node, i_global]
                    jacobian_surface[i_global, j_face_local] = sum_val

            # Вектори дотичні до поверхні: dR/d_eta, dR/d_tau
            dr_d_eta = jacobian_surface[:, 0]  # [dx/d_eta, dy/d_eta, dz/d_eta]
            dr_d_tau = jacobian_surface[:, 1]  # [dx/d_tau, dy/d_tau, dz/d_tau]

            # Вектор нормалі до поверхні (не нормований)
            normal_vector = np.cross(dr_d_eta, dr_d_tau)

            # Модуль Якобіана поверхні (площа елемента поверхні dA)
            # Lambda_dA = || normal_vector ||
            lambda_dA = np.linalg.norm(normal_vector)

            if lambda_dA < 1e-9:  # Перевірка на виродженість
                # print(f"Попередження: Нульовий Якобіан поверхні Lambda_dA в ТГ {gp_idx_face} грані {face_id}")
                continue

            # Одиничний вектор нормалі
            unit_normal = normal_vector / lambda_dA

            # Розподіл навантаження по вузлах грані
            for i_face_node in range(8):  # по 8 вузлах грані
                psi_i = psi_values_at_gp[i_face_node]

                # Глобальний індекс i-го вузла грані серед 20 вузлів елемента
                hex_node_idx = hex_face_node_indices[i_face_node]

                # P_n * n_x * psi_i * dA -> F_x
                # P_n * n_y * psi_i * dA -> F_y
                # P_n * n_z * psi_i * dA -> F_z
                # Тиск P діє протилежно до нормалі, якщо P - скалярний тиск.
                # Або Pn - це вже проекція сили на нормаль.
                # "нормальне до неї навантаження інтенсивності Р" (стор. 4)
                # Якщо P - тиск, то сила F = -P * n.
                # Або, якщо P - це вже величина сили на одиницю площі в напрямку нормалі, то F = P * n.
                # Знак тиску в constants.py (-10e6) вказує на стиснення (вздовж -Z для верхньої грані).
                # Тож, якщо normal_vector вказує назовні, а тиск "тисне" на поверхню,
                # то сила = pressure * unit_normal (де pressure < 0 для стиснення).

                force_components = pressure * unit_normal  # Вектор сили на одиницю площі

                self.FE[hex_node_idx] += force_components[0] * psi_i * lambda_dA * weight_face
                self.FE[hex_node_idx + 20] += force_components[1] * psi_i * lambda_dA * weight_face
                self.FE[hex_node_idx + 40] += force_components[2] * psi_i * lambda_dA * weight_face