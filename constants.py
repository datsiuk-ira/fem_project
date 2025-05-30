# constants.py
import numpy as np

# Локальні координати вузлів стандартного 20-вузлового гексаедра
NODE_LOCAL_COORDS_20_HEX = np.array([
    # Кутові вузли (0-7)
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    # Середини ребер (8-19)
    [ 0, -1, -1], [ 1,  0, -1], [ 0,  1, -1], [-1,  0, -1],
    [ 0, -1,  1], [ 1,  0,  1], [ 0,  1,  1], [-1,  0,  1],
    [-1, -1,  0], [ 1, -1,  0], [ 1,  1,  0], [-1,  1,  0]
])

# Локальні координати (eta, tau) для 8-вузлового стандартного квадратного елемента (грані)
# Нумерація відповідає Рис. 4, стор. 19, Лабораторний практикум.pdf (але 0-індексована)
# Кутові: 0-3; Серединні: 4-7
NODE_LOCAL_COORDS_8_QUAD_FACE = np.array([
    [-1, -1], # 0 (Локальний вузол 1 на рис.4)
    [ 1, -1], # 1 (Локальний вузол 2 на рис.4)
    [ 1,  1], # 2 (Локальний вузол 3 на рис.4)
    [-1,  1], # 3 (Локальний вузол 4 на рис.4)
    [ 0, -1], # 4 (Локальний вузол 5 на рис.4, між 0-1)
    [ 1,  0], # 5 (Локальний вузол 6 на рис.4, між 1-2)
    [ 0,  1], # 6 (Локальний вузол 7 на рис.4, між 2-3)
    [-1,  0]  # 7 (Локальний вузол 8 на рис.4, між 3-0)
])

# Визначення граней гексаедра: (вісь, значення_координати, відображення_локальних_осей_грані)
# (axis_index (0:alpha, 1:beta, 2:gamma), coord_value (-1 or 1),
#  map_to_face_eta_axis (0:alpha, 1:beta, 2:gamma), map_to_face_tau_axis)
# та відповідні 8 вузлів з 20-вузлового гексаедра (0-індексовані)
# Нумерація граней згідно "Лабораторний практикум.pdf", стор.18, ZP i,2
# Припустимо наступну нумерацію граней:
# 1: alpha = -1 (задня X)
# 2: alpha =  1 (передня X)
# 3: beta  = -1 (нижня Y)
# 4: beta  =  1 (верхня Y)
# 5: gamma = -1 (нижня Z, "дно")
# 6: gamma =  1 (верхня Z, "кришка")

HEX_FACE_DEFINITIONS = {
    1: {"fixed_coord_idx": 0, "fixed_coord_val": -1, "eta_maps_to_axis": 1, "tau_maps_to_axis": 2, # beta, gamma
        "hex_node_indices": [0, 3, 7, 4, 11, 19, 15, 16]}, # Вузли на грані alpha=-1 (локальні індекси 0-19)
                                                           # 1,4,8,5 кутові; 12,20,16,17 серединні (з PDF рис.2)
                                                           # Наші індекси: 0,3,7,4 кутові; 11,19,15,16 серединні
    2: {"fixed_coord_idx": 0, "fixed_coord_val":  1, "eta_maps_to_axis": 1, "tau_maps_to_axis": 2, # beta, gamma
        "hex_node_indices": [1, 2, 6, 5, 9, 18, 13, 17]}, # Вузли на грані alpha=1
                                                           # 2,3,7,6 кутові; 10,19,14,18 серединні
                                                           # Наші індекси: 1,2,6,5 кутові; 9,18,13,17 серединні
    3: {"fixed_coord_idx": 1, "fixed_coord_val": -1, "eta_maps_to_axis": 0, "tau_maps_to_axis": 2, # alpha, gamma
        "hex_node_indices": [0, 1, 5, 4, 8, 17, 13, 12]}, # Вузли на грані beta=-1
                                                           # 1,2,6,5 кутові; 9,18,14,13 серединні
                                                           # Наші індекси: 0,1,5,4 кутові; 8,17,13,12 (для eta=alpha, tau=gamma)
    4: {"fixed_coord_idx": 1, "fixed_coord_val":  1, "eta_maps_to_axis": 0, "tau_maps_to_axis": 2, # alpha, gamma
        "hex_node_indices": [3, 2, 6, 7, 10, 18, 14, 15]}, # Вузли на грані beta=1
                                                            # 4,3,7,8 кутові; 11,19,15,16 серединні
                                                            # Наші індекси: 3,2,6,7 кутові; 10,18,14,15
    5: {"fixed_coord_idx": 2, "fixed_coord_val": -1, "eta_maps_to_axis": 0, "tau_maps_to_axis": 1, # alpha, beta
        "hex_node_indices": [0, 1, 2, 3, 8, 9, 10, 11]},  # Вузли на грані gamma=-1
                                                            # 1,2,3,4 кутові; 9,10,11,12 серединні
                                                            # Наші індекси: 0,1,2,3 кутові; 8,9,10,11
    6: {"fixed_coord_idx": 2, "fixed_coord_val":  1, "eta_maps_to_axis": 0, "tau_maps_to_axis": 1, # alpha, beta
        "hex_node_indices": [4, 5, 6, 7, 12, 13, 14, 15]}  # Вузли на грані gamma=1
                                                            # 5,6,7,8 кутові; 13,14,15,16 серединні
                                                            # Наші індекси: 4,5,6,7 кутові; 12,13,14,15
}
# Порядок вузлів у hex_node_indices відповідає обходу грані: (-1,-1), (1,-1), (1,1), (-1,1) в локальних (eta,tau)
# та потім серединні ребра між ними.


# Стандартні параметри для генерації сітки
DEFAULT_DIMENSIONS = {
    "ax": 2.0, "ay": 1.0, "az": 2.0
}
DEFAULT_DIVISIONS = {
    "nx": 1, "ny": 1, "nz": 1
}

# Константи Гаусса для 1D інтегрування (3 точки)
GAUSS_POINTS_1D_3 = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
GAUSS_WEIGHTS_1D_3 = np.array([5/9, 8/9, 5/9])

# Кількість вузлів Гаусса
NUM_GAUSS_POINTS_1D = len(GAUSS_POINTS_1D_3)
NUM_GAUSS_POINTS_2D = NUM_GAUSS_POINTS_1D**2 # Для поверхонь (3*3=9)
NUM_GAUSS_POINTS_3D = NUM_GAUSS_POINTS_1D**3 # Для об'ємів (3*3*3=27)


# Матеріальні властивості
YOUNGS_MODULUS = 210e9
POISSON_RATIO = 0.3
LAMBDA_DOC = YOUNGS_MODULUS / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO))
MU_DOC = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO))

# Параметри навантаження (приклад: тиск на верхню грань gamma=1)
DEFAULT_LOAD_FACE_ID = 6 # Верхня грань (gamma=1)
DEFAULT_PRESSURE = -10e6 # Па, негативний тиск (вниз по Z, якщо нормаль Z+)

# Параметри закріплення (приклад: нижня грань gamma=-1 повністю закріплена)
DEFAULT_FIXED_FACE_ID = 5 # Нижня грань (gamma=-1)
# DOF_TO_FIX: 0=x, 1=y, 2=z. (True, True, True) означає закріплення по X, Y, Z.
DEFAULT_FIXED_DOFS = (True, True, True)
BOUNDARY_CONDITION_MAGNITUDE = 1e20 # Велике число для методу штрафів/великих діагоналей