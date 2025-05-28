# main_app.py
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time

from mesh_module import Mesh
from constants import (DEFAULT_DIMENSIONS, DEFAULT_DIVISIONS, DEFAULT_LOAD_FACE_ID, DEFAULT_PRESSURE,
                       DEFAULT_FIXED_FACE_ID, DEFAULT_FIXED_DOFS, HEX_FACE_DEFINITIONS,
                       BOUNDARY_CONDITION_MAGNITUDE)
from visualization_module import plot_mesh_3d, plot_deformed_mesh, plot_stress_contour  # Оновлені імпорти
from element_module import Hexa20ElementCalculator
from assembly_module import GlobalSystem
from solver_module import solve_system
from postprocessing_module import StressStrainCalculator


class FEMApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FEM Аналіз - 20-вузловий Гексаедр")
        self.geometry("1250x850")  # Трохи ширше для нових кнопок
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.mesh_instance: Mesh | None = None
        self.element_calculators: dict[int, Hexa20ElementCalculator] = {}
        self.global_system: GlobalSystem | None = None
        self.U_solution: np.ndarray | None = None
        self.stress_calculator: StressStrainCalculator | None = None

        # ----- Головний фрейм -----
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ----- Ліва панель: Керування -----
        controls_panel = ctk.CTkScrollableFrame(main_frame, width=380)  # Збільшено ширину
        controls_panel.pack(side="left", fill="y", padx=(0, 5), pady=0)

        # ... (Код для параметрів сітки, навантаження, закріплення - без змін) ...
        ctk.CTkLabel(controls_panel, text="Параметри сітки:", font=("Arial", 16, "bold")).pack(pady=5)
        self.entries = {}
        ctk.CTkLabel(controls_panel, text="Розміри (ax, ay, az):").pack(anchor="w", padx=10)
        for dim, val in DEFAULT_DIMENSIONS.items():
            frame = ctk.CTkFrame(controls_panel);
            frame.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(frame, text=f"{dim}:", width=30).pack(side="left")
            entry = ctk.CTkEntry(frame);
            entry.insert(0, str(val));
            entry.pack(side="left", fill="x", expand=True)
            self.entries[dim] = entry
        ctk.CTkLabel(controls_panel, text="Кількість елементів (nx, ny, nz):").pack(anchor="w", padx=10, pady=(5, 0))
        for div_param, val in DEFAULT_DIVISIONS.items():
            frame = ctk.CTkFrame(controls_panel);
            frame.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(frame, text=f"{div_param}:", width=30).pack(side="left")
            entry = ctk.CTkEntry(frame);
            entry.insert(0, str(val));
            entry.pack(side="left", fill="x", expand=True)
            self.entries[div_param] = entry

        generate_button = ctk.CTkButton(controls_panel, text="1. Згенерувати сітку",
                                        command=self.generate_and_display_mesh)
        generate_button.pack(pady=(10, 3), padx=10, fill="x")

        ctk.CTkLabel(controls_panel, text="Навантаження (тиск):", font=("Arial", 14, "bold")).pack(pady=(5, 0),
                                                                                                   anchor="w", padx=10)
        self.load_face_id_entry = self._create_param_entry(controls_panel, "ID грані (1-6):", DEFAULT_LOAD_FACE_ID)
        self.pressure_entry = self._create_param_entry(controls_panel, "Тиск (Па):", DEFAULT_PRESSURE)

        ctk.CTkLabel(controls_panel, text="Закріплення:", font=("Arial", 14, "bold")).pack(pady=(5, 0), anchor="w",
                                                                                           padx=10)
        self.fixed_face_id_entry = self._create_param_entry(controls_panel, "ID фікс. грані (1-6):",
                                                            DEFAULT_FIXED_FACE_ID)
        self.fixed_dof_vars = {}
        dof_labels = ["Fix X", "Fix Y", "Fix Z"]
        for i, label in enumerate(dof_labels):
            var = ctk.StringVar(value="1" if DEFAULT_FIXED_DOFS[i] else "0")
            chk = ctk.CTkCheckBox(controls_panel, text=label, variable=var);
            chk.pack(anchor="w", padx=20, pady=1)
            self.fixed_dof_vars[i] = var

        assemble_button = ctk.CTkButton(controls_panel, text="2. Зібрати глобальну систему",
                                        command=self.assemble_global_system_fem)
        assemble_button.pack(pady=(10, 3), padx=10, fill="x")
        apply_bc_button = ctk.CTkButton(controls_panel, text="3. Застосувати ГУ",
                                        command=self.apply_boundary_conditions_fem)
        apply_bc_button.pack(pady=3, padx=10, fill="x")
        solve_button = ctk.CTkButton(controls_panel, text="4. Розв'язати систему", command=self.solve_fem_system)
        solve_button.pack(pady=3, padx=10, fill="x")
        stress_button = ctk.CTkButton(controls_panel, text="5. Обчислити напруження",
                                      command=self.calculate_stresses_fem)
        stress_button.pack(pady=3, padx=10, fill="x")

        # Нові елементи для вибору візуалізації
        ctk.CTkLabel(controls_panel, text="Візуалізація:", font=("Arial", 14, "bold")).pack(pady=(10, 0), anchor="w",
                                                                                            padx=10)

        self.plot_type_var = ctk.StringVar(value="Початкова сітка")
        plot_options = ["Початкова сітка", "Деформована сітка", "Напруження S1", "Напруження S2", "Напруження S3",
                        "Напруження Von Mises"]  # Додамо Von Mises пізніше
        plot_type_menu = ctk.CTkOptionMenu(controls_panel, variable=self.plot_type_var, values=plot_options,
                                           command=self.update_plot_view)
        plot_type_menu.pack(pady=5, padx=10, fill="x")

        self.disp_scale_entry = self._create_param_entry(controls_panel, "Масштаб деформації:", "100")

        # Вікно для виводу інформації
        ctk.CTkLabel(controls_panel, text="Інформація:", font=("Arial", 14, "bold")).pack(pady=(5, 0), anchor="w",
                                                                                          padx=10)
        self.info_textbox = ctk.CTkTextbox(controls_panel, height=200, wrap="word")  # Трохи менше, бо панель довша
        self.info_textbox.pack(pady=(5, 10), padx=10, fill="both", expand=True)
        self.info_textbox.configure(state="disabled")

        # ----- Права панель: Візуалізація -----
        plot_panel = ctk.CTkFrame(main_frame);
        plot_panel.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=0)
        self.fig = plt.figure(figsize=(7, 6));
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_panel)
        self.canvas_widget = self.canvas.get_tk_widget();
        self.canvas_widget.pack(fill="both", expand=True)
        self._update_plot_placeholder()  # Замінено на update_plot_view
        self.update_plot_view()  # Початкове відображення

    def _create_param_entry(self, parent, label_text, default_value):
        # ... (код з попереднього кроку) ...
        frame = ctk.CTkFrame(parent);
        frame.pack(fill="x", padx=10, pady=1)
        ctk.CTkLabel(frame, text=label_text, width=140, anchor="w").pack(side="left")  # Збільшено ширину мітки
        entry = ctk.CTkEntry(frame);
        entry.insert(0, str(default_value));
        entry.pack(side="left", fill="x", expand=True)
        return entry

    def _update_plot_placeholder(self, message="Оберіть дію або згенеруйте сітку"):  # Оновлено повідомлення
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='3d')
        ax.text(0.5, 0.5, 0.5, message, ha="center", va="center", fontsize=12, wrap=True, transform=ax.transAxes)
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.set_zticks([])
        self.canvas.draw_idle()

    def _log_message(self, message: str, clear_previous: bool = False):
        # ... (код з попереднього кроку) ...
        self.info_textbox.configure(state="normal")
        if clear_previous: self.info_textbox.delete("1.0", "end")
        self.info_textbox.insert("end", message + "\n")
        self.info_textbox.configure(state="disabled");
        self.info_textbox.see("end")

    def generate_and_display_mesh(self):
        # ... (код з попереднього кроку, але викликає update_plot_view наприкінці) ...
        try:
            dims = {k: float(self.entries[k].get()) for k in DEFAULT_DIMENSIONS}
            divs = {k: int(self.entries[k].get()) for k in DEFAULT_DIVISIONS}
        except ValueError:
            self._log_message("Помилка: Некоректні значення параметрів.", clear_previous=True)
            self.update_plot_view()
            return
        self._log_message("Генерація сітки...", clear_previous=True)
        try:
            self.mesh_instance = Mesh(ax=dims["ax"], ay=dims["ay"], az=dims["az"], nx=divs["nx"], ny=divs["ny"],
                                      nz=divs["nz"])
        except Exception as e:
            self._log_message(f"Помилка генерації сітки: {e}");
            self.update_plot_view();
            return  # Оновлено
        self._log_message(
            f"Вузлів (nqp): {self.mesh_instance.nqp}, Елементів (nel): {self.mesh_instance.nel}, Півширина (ng): {self.mesh_instance.ng}")

        self.element_calculators = {};
        self.global_system = None;
        self.U_solution = None;
        self.stress_calculator = None
        self.plot_type_var.set("Початкова сітка")  # Скидання на початкову сітку
        self.update_plot_view()  # Оновлення візуалізації
        if self.mesh_instance.nel > 0: self._log_message("Сітку згенеровано.")

    def assemble_global_system_fem(self):
        # ... (код з попереднього кроку) ...
        if not self.mesh_instance or self.mesh_instance.nel == 0:
            self._log_message("Спочатку згенеруйте сітку.", clear_previous=True);
            return
        self._log_message("\n--- Збирання глобальної системи ---", clear_previous=True)
        start_time = time.time();
        self.global_system = GlobalSystem(self.mesh_instance)
        try:
            load_face_id = int(self.load_face_id_entry.get());
            pressure = float(self.pressure_entry.get())
        except ValueError:
            self._log_message("Помилка: Некоректні параметри навантаження."); return
        self.element_calculators = {}
        for elem_idx in range(self.mesh_instance.nel):
            elem_node_coords = self.mesh_instance.get_element_nodes_coords(elem_idx)
            calculator = Hexa20ElementCalculator(elem_node_coords)
            calculator.calculate_load_vector(face_id=load_face_id, pressure=pressure)
            self.element_calculators[elem_idx] = calculator
            self.global_system.assemble_element(elem_idx, calculator.MGE, calculator.FE)
        end_time = time.time()
        self._log_message(f"Глобальну систему зібрано за {end_time - start_time:.3f} сек.")
        self._log_message(f"Розмір MG: {self.global_system.MG.shape}, F: {self.global_system.F.shape}")
        self._log_message(
            f"Норма MG: {np.linalg.norm(self.global_system.MG):.2e}, Норма F: {np.linalg.norm(self.global_system.F):.2e}")
        self.U_solution = None;
        self.stress_calculator = None

    def apply_boundary_conditions_fem(self):
        # ... (використовуємо оновлений код з попереднього кроку) ...
        if not self.global_system: self._log_message("Спочатку зберіть глобальну систему.", clear_previous=True); return
        if not self.mesh_instance: self._log_message("Сітка не згенерована.", clear_previous=True); return
        self._log_message("\n--- Застосування граничних умов ---")
        try:
            fixed_face_id = int(self.fixed_face_id_entry.get())
            dofs_to_fix_flags = [bool(int(self.fixed_dof_vars[i].get())) for i in range(3)]
        except ValueError:
            self._log_message("Помилка: Некоректні параметри закріплення."); return
        if fixed_face_id not in HEX_FACE_DEFINITIONS: self._log_message(
            f"Помилка: Неправильний ID фіксованої грані {fixed_face_id}."); return
        face_def_for_bc = HEX_FACE_DEFINITIONS[fixed_face_id]
        fixed_coord_idx = face_def_for_bc["fixed_coord_idx"]
        fixed_coord_val_local = face_def_for_bc["fixed_coord_val"]
        target_global_coord_val = 0.0
        if fixed_coord_val_local == 1:
            if fixed_coord_idx == 0:
                target_global_coord_val = self.mesh_instance.ax
            elif fixed_coord_idx == 1:
                target_global_coord_val = self.mesh_instance.ay
            else:
                target_global_coord_val = self.mesh_instance.az
        fixed_global_node_indices = set()
        tolerance = 1e-6
        for node_idx in range(self.mesh_instance.nqp):
            node_coord = self.mesh_instance.AKT[node_idx, fixed_coord_idx]
            if abs(node_coord - target_global_coord_val) < tolerance: fixed_global_node_indices.add(node_idx)
        bc_info_list = []
        for global_node_idx in sorted(list(fixed_global_node_indices)):
            for dof_idx_in_node, should_fix in enumerate(dofs_to_fix_flags):
                if should_fix: bc_info_list.append((global_node_idx, dof_idx_in_node, 0.0))
        if not bc_info_list: self._log_message("Не знайдено вузлів для закріплення за поточними критеріями."); return
        self.global_system.apply_boundary_conditions(bc_info_list)
        self._log_message(f"Застосовано ГУ для {len(bc_info_list)} ступенів свободи.")
        self._log_message(f"Норма MG після ГУ: {np.linalg.norm(self.global_system.MG):.2e}")
        self._log_message(f"Норма F після ГУ: {np.linalg.norm(self.global_system.F):.2e}")
        self.U_solution = None;
        self.stress_calculator = None

    def solve_fem_system(self):
        # ... (код з попереднього кроку) ...
        if not self.global_system or not hasattr(self.global_system, 'MG'):
            self._log_message("Спочатку зберіть систему та застосуйте ГУ.", clear_previous=True);
            return
        self._log_message("\n--- Розв'язання системи ---")
        start_time = time.time()
        self.U_solution = solve_system(self.global_system.MG, self.global_system.F)
        end_time = time.time()
        if self.U_solution is not None:
            self._log_message(f"Систему розв'язано за {end_time - start_time:.3f} сек.")
            self._log_message(f"Норма вектора переміщень U: {np.linalg.norm(self.U_solution):.3e}")
            min_disp = np.min(self.U_solution);
            max_disp = np.max(self.U_solution)
            self._log_message(f"Мін. переміщення: {min_disp:.3e}, Макс. переміщення: {max_disp:.3e}")
            self.plot_type_var.set("Деформована сітка")  # Автоматично показати деформовану
            self.update_plot_view()
        else:
            self._log_message("Не вдалося розв'язати систему.");
            self.update_plot_view(message="Не вдалося розв'язати систему.")  # Оновлено
        self.stress_calculator = None

    def calculate_stresses_fem(self):
        # ... (код з попереднього кроку) ...
        if self.U_solution is None: self._log_message("Спочатку розв'яжіть систему.", clear_previous=True); return
        if not self.mesh_instance: self._log_message("Сітка не згенерована.", clear_previous=True); return
        self._log_message("\n--- Обчислення напружень та деформацій ---")
        start_time = time.time()
        try:
            self.stress_calculator = StressStrainCalculator(self.mesh_instance, self.U_solution)
        except Exception as e:
            self._log_message(f"Помилка обчислення напружень: {e}")
            import traceback;
            self._log_message(traceback.format_exc());
            return
        end_time = time.time()
        self._log_message(f"Напруження обчислено за {end_time - start_time:.3f} сек.")
        if self.stress_calculator.nodal_principal_stresses.size > 0:
            max_s1 = np.max(self.stress_calculator.nodal_principal_stresses[:, 0])
            min_s3 = np.min(self.stress_calculator.nodal_principal_stresses[:, 2])
            self._log_message(f"Макс. головне напруження (S1): {max_s1:.3e} Па")
            self._log_message(f"Мін. головне напруження (S3): {min_s3:.3e} Па")
            self.plot_type_var.set("Напруження S1")  # Автоматично показати S1
            self.update_plot_view()
        else:
            self._log_message("Не вдалося обчислити вузлові напруження.")

    def update_plot_view(self, selected_option=None):  # selected_option передається з OptionMenu
        """Оновлює візуалізацію відповідно до вибраного типу."""
        plot_type = self.plot_type_var.get()

        if plot_type == "Початкова сітка":
            if self.mesh_instance:
                plot_mesh_3d(self.fig, self.mesh_instance.AKT, self.mesh_instance.NT, title_suffix="(Початкова)")
            else:
                self._update_plot_placeholder("Сітка не згенерована.")
        elif plot_type == "Деформована сітка":
            if self.mesh_instance and self.U_solution is not None:
                try:
                    scale = float(self.disp_scale_entry.get())
                    plot_deformed_mesh(self.fig, self.mesh_instance.AKT, self.U_solution, self.mesh_instance.NT,
                                       scale_factor=scale)
                except ValueError:
                    self._log_message("Помилка: некоректний масштаб деформації.")
                    self._update_plot_placeholder("Введіть числовий масштаб деформації.")
            else:
                self._update_plot_placeholder("Спочатку розв'яжіть систему для переміщень.")
        elif "Напруження S" in plot_type:
            if self.stress_calculator and self.stress_calculator.nodal_principal_stresses.size > 0:
                stress_idx = int(plot_type.split("S")[1]) - 1  # 0 для S1, 1 для S2, 2 для S3
                if 0 <= stress_idx < 3:
                    plot_stress_contour(self.fig, self.mesh_instance.AKT, self.mesh_instance.NT,
                                        self.stress_calculator.nodal_principal_stresses[:, stress_idx],
                                        component_name=f"Головне напруження S{stress_idx + 1}")
                else:
                    self._update_plot_placeholder(f"Неправильний індекс головного напруження: S{stress_idx + 1}")
            else:
                self._update_plot_placeholder("Спочатку обчисліть напруження.")
        else:
            self._update_plot_placeholder()  # За замовчуванням або для невідомих опцій


if __name__ == "__main__":
    app = FEMApp()
    app.mainloop()