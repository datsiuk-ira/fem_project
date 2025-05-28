# visualization_module.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable


def plot_mesh_3d(fig, AKT: np.ndarray, NT: np.ndarray, plot_nodes=True, plot_edges=True, title_suffix=""):
    """
    Відображає 3D сітку скінченних елементів.
    Args:
        fig: Фігура matplotlib.
        AKT (np.ndarray): Координати вузлів (nqp, 3).
        NT (np.ndarray): Зв'язність елементів (nel, 20).
        plot_nodes (bool): Чи відображати вузли.
        plot_edges (bool): Чи відображати ребра елементів.
        title_suffix (str): Додатковий текст для заголовка.
    """
    ax = fig.add_subplot(111, projection='3d')
    ax.clear()

    if AKT.size == 0:
        ax.text(0.5, 0.5, 0.5, "Сітка порожня", ha='center', va='center', transform=ax.transAxes)
        fig.canvas.draw_idle()
        return

    if plot_nodes:
        ax.scatter(AKT[:, 0], AKT[:, 1], AKT[:, 2], c='blue', marker='o', s=10, label="Вузли")

    if plot_edges and NT.size > 0:
        edges_def = [
            (0, 1, 8), (1, 2, 9), (2, 3, 10), (3, 0, 11),
            (4, 5, 12), (5, 6, 13), (6, 7, 14), (7, 4, 15),
            (0, 4, 16), (1, 5, 17), (2, 6, 18), (3, 7, 19)
        ]
        edge_lines = []
        for i in range(NT.shape[0]):
            element_node_ids = NT[i, :]
            element_nodes_coords = AKT[element_node_ids]
            for n1_idx_local, n2_idx_local, mid_idx_local in edges_def:
                p1 = element_nodes_coords[n1_idx_local]
                p_mid = element_nodes_coords[mid_idx_local]
                p2 = element_nodes_coords[n2_idx_local]
                edge_lines.append([p1, p_mid])
                edge_lines.append([p_mid, p2])

        lc = Line3DCollection(edge_lines, colors='k', linewidths=0.5, label="Ребра")
        ax.add_collection(lc)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Скінченно-елементна сітка {title_suffix}")

    if AKT.size > 0:
        min_coords = np.min(AKT, axis=0)
        max_coords = np.max(AKT, axis=0)
        # Встановлюємо межі, щоб забезпечити кубічну форму, якщо це можливо
        all_coords = np.vstack((min_coords, max_coords))
        center = np.mean(all_coords, axis=0)
        plot_radius = 0.5 * np.max(max_coords - min_coords)
        if plot_radius == 0: plot_radius = 1  # У випадку одного вузла

        ax.set_xlim([center[0] - plot_radius, center[0] + plot_radius])
        ax.set_ylim([center[1] - plot_radius, center[1] + plot_radius])
        ax.set_zlim([center[2] - plot_radius, center[2] + plot_radius])

    # ax.legend() # Може заважати, якщо багато елементів
    plt.tight_layout()
    fig.canvas.draw_idle()


def plot_deformed_mesh(fig, AKT_initial: np.ndarray, U_solution: np.ndarray, NT: np.ndarray, scale_factor: float = 1.0):
    """
    Відображає деформовану сітку.
    Args:
        fig: Фігура matplotlib.
        AKT_initial (np.ndarray): Початкові координати вузлів (nqp, 3).
        U_solution (np.ndarray): Вектор глобальних переміщень (nqp*3,).
        NT (np.ndarray): Зв'язність елементів (nel, 20).
        scale_factor (float): Коефіцієнт масштабування переміщень для візуалізації.
    """
    if U_solution is None or AKT_initial.size == 0:
        ax = fig.add_subplot(111, projection='3d')
        ax.clear()
        ax.text(0.5, 0.5, 0.5, "Немає даних для деформованої сітки", ha='center', va='center', transform=ax.transAxes)
        fig.canvas.draw_idle()
        return

    nqp = AKT_initial.shape[0]
    displacements = U_solution.reshape((nqp, 3))
    AKT_deformed = AKT_initial + displacements * scale_factor

    # Використовуємо існуючу функцію для відображення, але з деформованими координатами
    plot_mesh_3d(fig, AKT_deformed, NT, plot_nodes=False, plot_edges=True,
                 title_suffix=f"(Деформована, масштаб x{scale_factor})")


def plot_stress_contour(fig, AKT: np.ndarray, NT: np.ndarray, nodal_stress_component: np.ndarray,
                        component_name: str = "Напруження"):
    """
    Відображає поле напружень (одну компоненту) на вузлах сітки.
    Args:
        fig: Фігура matplotlib.
        AKT (np.ndarray): Координати вузлів (nqp, 3).
        NT (np.ndarray): Зв'язність елементів (nel, 20).
        nodal_stress_component (np.ndarray): Значення компоненти напружень у кожному вузлі (nqp,).
        component_name (str): Назва компоненти напружень для заголовка та колірної шкали.
    """
    ax = fig.add_subplot(111, projection='3d')
    ax.clear()

    if AKT.size == 0 or nodal_stress_component.size == 0:
        ax.text(0.5, 0.5, 0.5, "Немає даних для полів напружень", ha='center', va='center', transform=ax.transAxes)
        fig.canvas.draw_idle()
        return

    # Нормалізація значень напружень для колірної карти
    norm = mcolors.Normalize(vmin=np.min(nodal_stress_component), vmax=np.max(nodal_stress_component))
    cmap = plt.cm.jet  # Вибір колірної карти

    # Відображення вузлів з кольорами відповідно до напружень
    # sctt = ax.scatter(AKT[:, 0], AKT[:, 1], AKT[:, 2], c=nodal_stress_component, cmap=cmap, norm=norm, s=20, edgecolor='k', linewidth=0.2)

    # Спроба відобразити грані елементів з усередненим кольором
    # Це складно зробити ефективно та правильно в matplotlib для 3D з вузловими значеннями.
    # Простіший варіант - відобразити вузли з кольорами.
    # Або, для кращої візуалізації, використовувати бібліотеки типу PyVista/VTK.
    # Поки що залишимо вузлове відображення з кольором.

    sctt = ax.scatter(AKT[:, 0], AKT[:, 1], AKT[:, 2], c=nodal_stress_component, cmap=cmap, norm=norm, s=35)

    # Додавання колірної шкали
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, aspect=10)
    cbar.set_label(f"{component_name} (Па)")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Поле напружень: {component_name}")

    if AKT.size > 0:
        min_coords = np.min(AKT, axis=0)
        max_coords = np.max(AKT, axis=0)
        center = np.mean(np.vstack((min_coords, max_coords)), axis=0)
        plot_radius = 0.5 * np.max(max_coords - min_coords)
        if plot_radius == 0: plot_radius = 1

        ax.set_xlim([center[0] - plot_radius, center[0] + plot_radius])
        ax.set_ylim([center[1] - plot_radius, center[1] + plot_radius])
        ax.set_zlim([center[2] - plot_radius, center[2] + plot_radius])

    plt.tight_layout()
    fig.canvas.draw_idle()