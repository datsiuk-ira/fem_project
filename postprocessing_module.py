# postprocessing_module.py
import numpy as np
from constants import NODE_LOCAL_COORDS_20_HEX, LAMBDA_DOC, MU_DOC, POISSON_RATIO
from shape_functions_module import get_shape_function_derivatives_20_node_hex
from element_module import Hexa20ElementCalculator  # Для доступу до методів обчислення J та DFIXYZ
from mesh_module import Mesh


class StressStrainCalculator:
    """Клас для обчислення напружень та деформацій."""

    def __init__(self, mesh: Mesh, U_global: np.ndarray):
        self.mesh = mesh
        self.U_global = U_global  # Глобальний вектор переміщень (num_dof,)

        # Зберігатимемо напруження та деформації для кожного вузла
        # Для кожного вузла: [eps_xx, eps_yy, eps_zz, gam_xy, gam_yz, gam_zx]
        # та [sig_xx, sig_yy, sig_zz, sig_xy, sig_yz, sig_xz]
        self.nodal_strains = np.zeros((mesh.nqp, 6))
        self.nodal_stresses = np.zeros((mesh.nqp, 6))
        self.nodal_principal_stresses = np.zeros((mesh.nqp, 3))  # (sig1, sig2, sig3)

        # Лічильник для усереднення значень у вузлах, що належать кільком елементам
        self._nodal_contribution_count = np.zeros(mesh.nqp, dtype=int)

        self._calculate_all_nodal_values()

    def _get_element_dof_values(self, element_idx: int) -> np.ndarray:
        """Витягує значення DOF (переміщень) для вузлів заданого елемента."""
        element_node_ids = self.mesh.NT[element_idx, :]
        element_dofs = np.zeros(60)
        for i_local_node in range(20):
            global_node_id = element_node_ids[i_local_node]
            element_dofs[i_local_node] = self.U_global[global_node_id * 3 + 0]  # Ux
            element_dofs[i_local_node + 20] = self.U_global[global_node_id * 3 + 1]  # Uy
            element_dofs[i_local_node + 40] = self.U_global[global_node_id * 3 + 2]  # Uz
        return element_dofs

    def _calculate_all_nodal_values(self):
        """Обчислює деформації та напруження в усіх вузлах, усереднюючи по елементах."""

        # Ітеруємо по кожному елементу
        for elem_idx in range(self.mesh.nel):
            elem_global_node_ids = self.mesh.NT[elem_idx, :]
            elem_node_coords_global = self.mesh.get_element_nodes_coords(elem_idx)
            elem_dof_values = self._get_element_dof_values(elem_idx)

            # Створюємо тимчасовий калькулятор для цього елемента, щоб отримати J та DFIXYZ
            # Це може бути неефективно, якщо DFIXYZ для вузлів не кешується.
            # Поки що, для простоти, будемо переобчислювати.
            temp_elem_calc = Hexa20ElementCalculator(elem_node_coords_global)

            # Ітеруємо по кожному з 20 локальних вузлів цього елемента
            for local_node_idx_in_elem in range(20):
                alpha, beta, gamma = NODE_LOCAL_COORDS_20_HEX[local_node_idx_in_elem]

                # Отримуємо похідні d(phi_i)/d(glob_coord) в цій локальній точці (alpha,beta,gamma)
                # Потрібно обчислити J(alpha,beta,gamma) та J_inv(alpha,beta,gamma)
                dphi_d_local_at_node = get_shape_function_derivatives_20_node_hex(alpha, beta, gamma).T  # (3, 20)

                jacobian_at_node = np.zeros((3, 3))
                for i_global in range(3):
                    for j_local in range(3):
                        sum_val = 0
                        for k_node_sf in range(20):  # по форм-функціях
                            sum_val += dphi_d_local_at_node[j_local, k_node_sf] * elem_node_coords_global[
                                k_node_sf, i_global]
                        jacobian_at_node[i_global, j_local] = sum_val

                try:
                    inv_jacobian_at_node = np.linalg.inv(jacobian_at_node)
                except np.linalg.LinAlgError:
                    # print(f"Сингулярний Якобіан при обчисленні напружень для вузла елемента {elem_idx}")
                    continue  # Пропустити цей вузол/елемент

                # DFIXYZ_at_node має бути (20, 3) - d(phi_i)/d(x), d(phi_i)/d(y), d(phi_i)/d(z)
                DFIXYZ_at_node = (dphi_d_local_at_node.T @ inv_jacobian_at_node.T)  # (20,3)

                # Матриця B для деформацій (6, 60)
                # epsilon = B * u_element
                # B_i = [dphi_i/dx  0          0       ]
                #       [0          dphi_i/dy  0       ]
                #       [0          0          dphi_i/dz]
                #       [dphi_i/dy  dphi_i/dx  0       ]
                #       [0          dphi_i/dz  dphi_i/dy]
                #       [dphi_i/dz  0          dphi_i/dx]
                B_matrix = np.zeros((6, 60))
                for i_sf in range(20):  # для кожної форм-функції
                    dphi_i_dx = DFIXYZ_at_node[i_sf, 0]
                    dphi_i_dy = DFIXYZ_at_node[i_sf, 1]
                    dphi_i_dz = DFIXYZ_at_node[i_sf, 2]

                    B_matrix[0, i_sf] = dphi_i_dx
                    B_matrix[1, i_sf + 20] = dphi_i_dy
                    B_matrix[2, i_sf + 40] = dphi_i_dz

                    B_matrix[3, i_sf] = dphi_i_dy
                    B_matrix[3, i_sf + 20] = dphi_i_dx

                    B_matrix[4, i_sf + 20] = dphi_i_dz
                    B_matrix[4, i_sf + 40] = dphi_i_dy

                    B_matrix[5, i_sf] = dphi_i_dz
                    B_matrix[5, i_sf + 40] = dphi_i_dx

                strains_vector = B_matrix @ elem_dof_values  # (6,) [eps_xx, eps_yy, eps_zz, gam_xy, gam_yz, gam_zx]

                # Обчислення напружень за законом Гука (стор. 5, рівняння (2))
                # sigma_xx = lambda_true * (eps_xx+eps_yy+eps_zz) + 2*mu_true*eps_xx
                # lambda_true = E*nu / ((1+nu)*(1-2*nu))
                # mu_true = E / (2*(1+nu)) (це MU_DOC)
                lambda_true = LAMBDA_DOC * POISSON_RATIO / (1 - POISSON_RATIO) if abs(
                    1 - POISSON_RATIO) > 1e-9 else LAMBDA_DOC * POISSON_RATIO
                # Або простіше: sigma_xx = LAMBDA_DOC * ( (1-POISSON_RATIO)*eps_xx + POISSON_RATIO*(eps_yy+eps_zz) ) - це з PDF стор.5
                # Ця формула виглядає не зовсім стандартно.
                # Стандартна матриця D для 3D:
                # C = E / ((1+nu)(1-2nu))
                # D_mat = C * np.array([
                #     [1-nu, nu,   nu,   0, 0, 0],
                #     [nu,   1-nu, nu,   0, 0, 0],
                #     [nu,   nu,   1-nu, 0, 0, 0],
                #     [0,0,0, (1-2*nu)/2, 0, 0],
                #     [0,0,0, 0, (1-2*nu)/2, 0],
                #     [0,0,0, 0, 0, (1-2*nu)/2]
                # ])
                # Або sigma_xx = LAMBDA_DOC * (eps_xx + eps_yy + eps_zz) + 2 * MU_DOC * eps_xx (якщо LAMBDA_DOC = lambda_true)
                # Ні, формула з стор. 5 документа:
                # sig_xx = lambda_val * ( (1-nu)*eps_x + nu*(eps_y+eps_z) )
                # де lambda_val = E / ((1+nu)(1-2*nu)) (це наш LAMBDA_DOC)
                # sig_xy = mu_val * gam_xy (де mu_val = E / (2*(1+nu)) (це наш MU_DOC)

                stresses_vector = np.zeros(6)
                eps_x, eps_y, eps_z = strains_vector[0], strains_vector[1], strains_vector[2]
                gam_xy, gam_yz, gam_zx = strains_vector[3], strains_vector[4], strains_vector[5]

                stresses_vector[0] = LAMBDA_DOC * (
                            (1 - POISSON_RATIO) * eps_x + POISSON_RATIO * (eps_y + eps_z))  # sig_xx
                stresses_vector[1] = LAMBDA_DOC * (
                            (1 - POISSON_RATIO) * eps_y + POISSON_RATIO * (eps_x + eps_z))  # sig_yy
                stresses_vector[2] = LAMBDA_DOC * (
                            (1 - POISSON_RATIO) * eps_z + POISSON_RATIO * (eps_x + eps_y))  # sig_zz
                stresses_vector[3] = MU_DOC * gam_xy  # tau_xy
                stresses_vector[4] = MU_DOC * gam_yz  # tau_yz
                stresses_vector[5] = MU_DOC * gam_zx  # tau_zx

                # Додавання до глобальних масивів вузлових значень
                global_node_id_of_local = elem_global_node_ids[local_node_idx_in_elem]
                self.nodal_strains[global_node_id_of_local, :] += strains_vector
                self.nodal_stresses[global_node_id_of_local, :] += stresses_vector
                self._nodal_contribution_count[global_node_id_of_local] += 1

        # Усереднення значень
        for i in range(self.mesh.nqp):
            if self._nodal_contribution_count[i] > 0:
                self.nodal_strains[i, :] /= self._nodal_contribution_count[i]
                self.nodal_stresses[i, :] /= self._nodal_contribution_count[i]

        # Обчислення головних напружень
        self._calculate_principal_stresses_for_all_nodes()

    def _calculate_principal_stresses_for_all_nodes(self):
        for node_idx in range(self.mesh.nqp):
            if self._nodal_contribution_count[node_idx] == 0: continue  # Якщо для вузла не було внесків

            s_xx, s_yy, s_zz = self.nodal_stresses[node_idx, 0], self.nodal_stresses[node_idx, 1], self.nodal_stresses[
                node_idx, 2]
            s_xy, s_yz, s_zx = self.nodal_stresses[node_idx, 3], self.nodal_stresses[node_idx, 4], self.nodal_stresses[
                node_idx, 5]

            # Інваріанти тензора напружень (стор. 29, "Адаптивні методи аналізу.pdf")
            J1 = s_xx + s_yy + s_zz
            J2 = (s_xx * s_yy + s_xx * s_zz + s_yy * s_zz) - (s_xy ** 2 + s_yz ** 2 + s_zx ** 2)
            J3 = (s_xx * s_yy * s_zz + 2 * s_xy * s_yz * s_zx) - \
                 (s_xx * s_yz ** 2 + s_yy * s_zx ** 2 + s_zz * s_xy ** 2)

            # Розв'язування кубічного рівняння: s^3 - J1*s^2 + J2*s - J3 = 0
            coeffs = [1, -J1, J2, -J3]
            roots = np.roots(coeffs)
            # Головні напруження сортуються зазвичай sig1 >= sig2 >= sig3
            self.nodal_principal_stresses[node_idx, :] = np.sort(roots)[::-1]