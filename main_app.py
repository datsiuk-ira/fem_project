# main_app.py
import customtkinter as ctk
import numpy as np
import time
import threading
import queue
import pyvista as pv

from mesh_module import Mesh
from constants import (DEFAULT_DIMENSIONS, DEFAULT_DIVISIONS, DEFAULT_LOAD_FACE_ID, DEFAULT_PRESSURE,
                       DEFAULT_FIXED_FACE_ID, DEFAULT_FIXED_DOFS, HEX_FACE_DEFINITIONS,
                       BOUNDARY_CONDITION_MAGNITUDE)
from visualization_module import PyVistaVisualizer
from element_module import Hexa20ElementCalculator
from assembly_module import GlobalSystem
from solver_module import solve_system
from postprocessing_module import StressStrainCalculator


class FEMApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FEM Аналіз (PyVista) - 20-вузловий Гексаедр")
        self.geometry("500x850")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.mesh_instance: Mesh | None = None
        self.element_calculators: dict[int, Hexa20ElementCalculator] = {}
        self.global_system: GlobalSystem | None = None
        self.U_solution: np.ndarray | None = None
        self.stress_calculator: StressStrainCalculator | None = None

        self.pv_visualizer = PyVistaVisualizer(app_logger_callback=self._log_message_from_thread)
        self.pv_visualizer.enable_picking(self.gui_picking_callback_from_thread)

        self._pv_thread: threading.Thread | None = None
        self._pv_command_queue = queue.Queue()

        controls_panel = ctk.CTkScrollableFrame(self, width=480)
        controls_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)

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
        self.load_face_id_entry = self._create_param_entry(controls_panel, "ID грані (1-6):", DEFAULT_LOAD_FACE_ID,
                                                           width=100)
        self.pressure_entry = self._create_param_entry(controls_panel, "Тиск (Па):", DEFAULT_PRESSURE, width=100)

        ctk.CTkLabel(controls_panel, text="Закріплення:", font=("Arial", 14, "bold")).pack(pady=(5, 0), anchor="w",
                                                                                           padx=10)
        self.fixed_face_id_entry = self._create_param_entry(controls_panel, "ID фікс. грані (1-6):",
                                                            DEFAULT_FIXED_FACE_ID, width=100)
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

        ctk.CTkLabel(controls_panel, text="Візуалізація (PyVista):", font=("Arial", 14, "bold")).pack(pady=(10, 0),
                                                                                                      anchor="w",
                                                                                                      padx=10)
        self.plot_type_var = ctk.StringVar(value="Початкова сітка")
        plot_options = ["Початкова сітка", "Деформована сітка", "Напруження S1", "Напруження S2", "Напруження S3"]
        plot_type_menu = ctk.CTkOptionMenu(controls_panel, variable=self.plot_type_var, values=plot_options,
                                           command=self.update_plot_view_command)
        plot_type_menu.pack(pady=5, padx=10, fill="x")
        self.disp_scale_entry = self._create_param_entry(controls_panel, "Масштаб деформації:", "100", width=100)

        update_plot_button = ctk.CTkButton(controls_panel, text="Оновити/Показати Візуалізацію",
                                           command=self.update_plot_view_command)
        update_plot_button.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(controls_panel, text="Інформація:", font=("Arial", 14, "bold")).pack(pady=(5, 0), anchor="w",
                                                                                          padx=10)
        self.info_textbox = ctk.CTkTextbox(controls_panel, height=250, wrap="word")
        self.info_textbox.pack(pady=(5, 10), padx=10, fill="both", expand=True)
        self.info_textbox.configure(state="disabled")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.launch_pyvista_thread()

    def launch_pyvista_thread(self):
        if self._pv_thread is None or not self._pv_thread.is_alive():
            if self.pv_visualizer.plotter:
                self.pv_visualizer.close_plotter()

            self._pv_thread = threading.Thread(target=self._pyvista_worker_loop, daemon=True)
            self._pv_thread.start()
            self._log_message("Потік PyVista запущено.", clear_previous=True)
        else:
            self._log_message("Потік PyVista вже активний.")

    def _pyvista_worker_loop(self):
        plotter_title = "3D FEM Візуалізація"
        try:
            self.pv_visualizer.plotter = pv.Plotter(
                window_size=[800, 600],
                title=plotter_title,
            )
            if self.pv_visualizer._picking_enabled and self.pv_visualizer._picking_callback_gui:
                self.pv_visualizer._register_picking_callback_pv()

            self._log_message_from_thread("Плоттер PyVista створено, запускаю вікно...")
            self.pv_visualizer.plotter.show(interactive_update=True, auto_close=False)
            self.pv_visualizer.is_active = True
            self._log_message_from_thread("Вікно PyVista активне та очікує команд.")

        except Exception as e:
            self._log_message_from_thread(f"Не вдалося створити або показати плоттер PyVista: {e}")
            self.pv_visualizer.is_active = False
            return

        while self.pv_visualizer.is_active:
            try:
                command, args, kwargs = self._pv_command_queue.get(timeout=0.1)

                if command == "stop":
                    self._log_message_from_thread("Команда stop для потоку PyVista.")
                    self.pv_visualizer.is_active = False
                    break

                if self.pv_visualizer.plotter and self.pv_visualizer.plotter.renderer:
                    new_title = kwargs.get("title",
                                           self.pv_visualizer.plotter.title if self.pv_visualizer.plotter.title else plotter_title)
                    if self.pv_visualizer.plotter.title != new_title:
                        self.pv_visualizer.plotter.title = new_title

                    # Очищаємо акторів перед відображенням нової сцени
                    self.pv_visualizer.plotter.clear_actors()

                    if command == "display_mesh":
                        self.pv_visualizer.display_mesh_to_plotter(*args, **kwargs)
                    elif command == "display_deformed_mesh":
                        self.pv_visualizer.display_deformed_mesh_to_plotter(*args, **kwargs)
                    elif command == "display_scalar_field":
                        self.pv_visualizer.display_scalar_field_to_plotter(*args, **kwargs)
                    elif command == "display_message":
                        self.pv_visualizer.display_message_on_plotter(*args, **kwargs)
                else:
                    if command == "stop": break
                    if self.pv_visualizer.plotter is None and self.pv_visualizer.is_active:
                        self._log_message_from_thread("Плоттер PyVista більше не існує. Потік зупиняється.")
                        self.pv_visualizer.is_active = False

                self._pv_command_queue.task_done()
            except queue.Empty:
                # Перевіряємо, чи вікно PyVista було закрито користувачем
                # Атрибут 'closed' може бути відсутнім у деяких версіях/конфігураціях PyVista.
                # Більш надійний спосіб - перевірити, чи рендерер ще існує.
                plotter_still_open = False
                if self.pv_visualizer.plotter and self.pv_visualizer.plotter.renderer:
                    # PyVista вікна зазвичай встановлюють renderer в None при закритті,
                    # або сам об'єкт plotter стає None після close_plotter.
                    # self.pv_visualizer.plotter.window_ಬೇಕು() # Цей метод може не існувати або блокувати
                    plotter_still_open = True  # Припускаємо, що відкритий, якщо об'єкт існує

                if not plotter_still_open and self.pv_visualizer.is_active:
                    # Якщо is_active все ще True, але плоттер виглядає закритим, оновлюємо стан
                    self._log_message_from_thread("Вікно PyVista, схоже, було закрито користувачем.")
                    self.pv_visualizer.is_active = False

                if not self.pv_visualizer.is_active:
                    break
                continue
            except AttributeError as ae:
                if "'Plotter' object has no attribute 'closed'" in str(ae):
                    # Обробляємо помилку, якщо атрибут 'closed' не існує
                    self._log_message_from_thread(
                        "Атрибут 'plotter.closed' не знайдено. Перевірка закриття вікна може бути неповною.")
                    # Продовжуємо роботу, але закриття вікна користувачем може не оброблятися коректно
                else:
                    self._log_message_from_thread(f"AttributeError в _pyvista_worker_loop: {ae}")
                # Можна додати інші обробки помилок тут
            except Exception as e:
                self._log_message_from_thread(f"Помилка в _pyvista_worker_loop: {e}")
                import traceback
                self._log_message_from_thread(traceback.format_exc())

        if self.pv_visualizer.plotter:
            self.pv_visualizer.close_plotter()
        self._log_message_from_thread("_pyvista_worker_loop завершено.")

    def on_closing(self):
        self._log_message("Закриття додатку...")
        self.pv_visualizer.is_active = False  # Встановлюємо прапорець для зупинки потоку PyVista
        if self._pv_thread and self._pv_thread.is_alive():
            self._pv_command_queue.put(("stop", [], {}))  # Надсилаємо команду зупинки
            self._pv_thread.join(timeout=2)  # Чекаємо завершення потоку
            if self._pv_thread.is_alive():
                self._log_message("Потік PyVista не завершився коректно вчасно.")

        if self.pv_visualizer and self.pv_visualizer.plotter:  # Додаткова перевірка
            self.pv_visualizer.close_plotter()

        self.destroy()

    def _create_param_entry(self, parent, label_text, default_value, width=140):
        frame = ctk.CTkFrame(parent);
        frame.pack(fill="x", padx=10, pady=1)
        ctk.CTkLabel(frame, text=label_text, width=width, anchor="w").pack(side="left")
        entry = ctk.CTkEntry(frame);
        entry.insert(0, str(default_value));
        entry.pack(side="left", fill="x", expand=True)
        return entry

    def _log_message_from_thread(self, message: str, clear_previous: bool = False):
        self.after(0, lambda: self._log_message(message, clear_previous))

    def _log_message(self, message: str, clear_previous: bool = False):
        self.info_textbox.configure(state="normal")
        if clear_previous: self.info_textbox.delete("1.0", "end")
        self.info_textbox.insert("end", message + "\n")
        self.info_textbox.configure(state="disabled");
        self.info_textbox.see("end")

    def generate_and_display_mesh(self):
        try:
            dims = {k: float(self.entries[k].get()) for k in DEFAULT_DIMENSIONS}
            divs = {k: int(self.entries[k].get()) for k in DEFAULT_DIVISIONS}
        except ValueError:
            self._log_message("Помилка: Некоректні значення параметрів.", clear_previous=True);
            return
        self._log_message("Генерація сітки...", clear_previous=True)
        try:
            self.mesh_instance = Mesh(ax=dims["ax"], ay=dims["ay"], az=dims["az"], nx=divs["nx"], ny=divs["ny"],
                                      nz=divs["nz"])
        except Exception as e:
            self._log_message(f"Помилка генерації сітки: {e}");
            return
        self._log_message(
            f"Вузлів (nqp): {self.mesh_instance.nqp}, Елементів (nel): {self.mesh_instance.nel}, Півширина (ng): {self.mesh_instance.ng}")
        self.element_calculators = {};
        self.global_system = None;
        self.U_solution = None;
        self.stress_calculator = None
        self.plot_type_var.set("Початкова сітка")
        # Запускаємо оновлення візуалізації після невеликої затримки, щоб GUI встиг обробити події
        self.after(100, self.update_plot_view_command)
        if self.mesh_instance.nel > 0: self._log_message("Сітку згенеровано.")

    def assemble_global_system_fem(self):
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
            self._log_message("Помилка: Некоректні параметри навантаження.");
            return
        self.element_calculators = {}
        for elem_idx in range(self.mesh_instance.nel):
            elem_node_coords = self.mesh_instance.get_element_nodes_coords(elem_idx)
            calculator = Hexa20ElementCalculator(elem_node_coords)
            calculator.calculate_load_vector(face_id=load_face_id, pressure=pressure)
            self.element_calculators[elem_idx] = calculator
            self.global_system.assemble_element(elem_idx, calculator.MGE, calculator.FE)
        end_time = time.time()
        self._log_message(f"Глобальну систему зібрано за {end_time - start_time:.3f} сек.")
        self.U_solution = None;
        self.stress_calculator = None

    def apply_boundary_conditions_fem(self):
        if not self.global_system: self._log_message("Спочатку зберіть глобальну систему.", clear_previous=True); return
        if not self.mesh_instance: self._log_message("Сітка не згенерована.", clear_previous=True); return
        self._log_message("\n--- Застосування граничних умов ---")
        try:
            fixed_face_id = int(self.fixed_face_id_entry.get())
            dofs_to_fix_flags = [bool(int(self.fixed_dof_vars[i].get())) for i in range(3)]
        except ValueError:
            self._log_message("Помилка: Некоректні параметри закріплення.");
            return
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

        if not bc_info_list: self._log_message("Не знайдено вузлів для закріплення."); return

        self.global_system.apply_boundary_conditions(bc_info_list)
        self._log_message(f"Застосовано ГУ для {len(bc_info_list)} ступенів свободи.")
        self.U_solution = None;
        self.stress_calculator = None

    def _reconstruct_full_mg_from_banded(self, banded_mg: np.ndarray, num_dof: int, bandwidth_ng: int) -> np.ndarray:
        """Допоміжна функція для реконструкції повної матриці MG зі стрічкового формату."""
        full_mg = np.zeros((num_dof, num_dof))
        # banded_mg[k, j] зберігає A_full[j-k, j] де k = j-i (індекс діагоналі)
        # k - це band_row (0 для головної діагоналі)
        for j_col in range(num_dof):  # Ітерація по стовпцях повної матриці
            for band_row in range(bandwidth_ng):  # Ітерація по діагоналях, що зберігаються
                i_row_full = j_col - band_row  # Рядок у повній матриці
                if 0 <= i_row_full < num_dof:  # Перевіряємо, чи рядок в межах матриці
                    value = banded_mg[band_row, j_col]
                    full_mg[i_row_full, j_col] = value
                    if i_row_full != j_col:  # Симетричне заповнення
                        full_mg[j_col, i_row_full] = value
        return full_mg

    def solve_fem_system(self):
        if not self.global_system or not hasattr(self.global_system, 'MG'):
            self._log_message("Спочатку зберіть систему та застосуйте ГУ.", clear_previous=True);
            return
        self._log_message("\n--- Розв'язання системи ---")
        start_time = time.time()

        if self.global_system.is_banded:
            self._log_message("Реконструкція повної матриці MG зі стрічкової для розв'язання...")
            # Переконуємося, що mesh_instance існує перед доступом до ng
            if not self.mesh_instance:
                self._log_message("Помилка: екземпляр сітки не створено, неможливо отримати ng.")
                return
            full_MG_reconstructed = self._reconstruct_full_mg_from_banded(
                self.global_system.MG,
                self.global_system.num_dof,
                self.mesh_instance.ng
            )
            self.U_solution = solve_system(full_MG_reconstructed, self.global_system.F)
        else:
            self.U_solution = solve_system(self.global_system.MG, self.global_system.F)

        end_time = time.time()
        if self.U_solution is not None:
            self._log_message(f"Систему розв'язано за {end_time - start_time:.3f} сек.")
            min_disp = np.min(self.U_solution);
            max_disp = np.max(self.U_solution)
            self._log_message(f"Мін. переміщення: {min_disp:.3e}, Макс. переміщення: {max_disp:.3e}")
            self.plot_type_var.set("Деформована сітка")
            self.update_plot_view_command()
        else:
            self._log_message("Не вдалося розв'язати систему.")
        self.stress_calculator = None

    def calculate_stresses_fem(self):
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
            self.plot_type_var.set("Напруження S1")
            self.update_plot_view_command()
        else:
            self._log_message("Не вдалося обчислити вузлові напруження.")

    def update_plot_view_command(self, selected_option=None):
        plot_type = self.plot_type_var.get()

        if not self.pv_visualizer.is_active and (self._pv_thread is None or not self._pv_thread.is_alive()):
            self._log_message("Вікно PyVista не активне або потік не запущено. Запускаю потік...")
            self.launch_pyvista_thread()
            self.after(500, lambda pt=plot_type: self._send_pv_command(pt))
            return
        self._send_pv_command(plot_type)

    def _send_pv_command(self, plot_type: str):
        args, kwargs = [], {}
        command_str = "display_message"
        default_message_args = ["Оберіть тип візуалізації або виконайте попередні кроки."]

        if plot_type == "Початкова сітка":
            if self.mesh_instance and self.mesh_instance.AKT.size > 0:
                command_str = "display_mesh"
                args = [self.mesh_instance.AKT, self.mesh_instance.NT]
                kwargs = {"title": "Початкова сітка"}
            else:
                args = ["Сітка не згенерована."]
        elif plot_type == "Деформована сітка":
            if self.mesh_instance and self.U_solution is not None:
                try:
                    scale = float(self.disp_scale_entry.get())
                    command_str = "display_deformed_mesh"
                    args = [self.mesh_instance.AKT, self.U_solution, self.mesh_instance.NT, scale]
                    kwargs = {"title": f"Деформована сітка (x{scale})"}
                except ValueError:
                    self._log_message("Помилка: некоректний масштаб деформації.")
                    args = ["Некоректний масштаб."]
            else:
                args = ["Спочатку розв'яжіть систему."]
        elif "Напруження S" in plot_type:
            if self.stress_calculator and self.stress_calculator.nodal_principal_stresses.size > 0 and self.mesh_instance:
                stress_idx_map = {"Напруження S1": 0, "Напруження S2": 1, "Напруження S3": 2}
                stress_idx = stress_idx_map.get(plot_type, -1)
                if 0 <= stress_idx < 3:
                    akt_to_display = self.mesh_instance.AKT
                    if self.U_solution is not None:
                        try:
                            scale = float(self.disp_scale_entry.get())
                            nqp = self.mesh_instance.AKT.shape[0]
                            displacements = self.U_solution.reshape((nqp, 3))
                            akt_to_display = self.mesh_instance.AKT + displacements * scale
                        except ValueError:
                            pass

                    command_str = "display_scalar_field"
                    args = [akt_to_display, self.mesh_instance.NT,
                            self.stress_calculator.nodal_principal_stresses[:, stress_idx],
                            f"S{stress_idx + 1}"]
                    kwargs = {"title": f"Головне напруження S{stress_idx + 1}"}
                else:
                    args = [f"Неправильний індекс напруження: {plot_type}"]
            else:
                args = ["Спочатку обчисліть напруження."]
        else:
            args = default_message_args

        if self._pv_thread and self._pv_thread.is_alive() and self.pv_visualizer.is_active:
            self._pv_command_queue.put((command_str, args, kwargs))
        elif not (self._pv_thread and self._pv_thread.is_alive()):
            self._log_message("Потік PyVista не активний. Спробуйте перезапустити додаток.")

    def gui_picking_callback_from_thread(self, node_id: int, picked_coords: np.ndarray,
                                         active_pv_mesh: pv.UnstructuredGrid):
        log_lines = []
        log_lines.append(f"\n--- Інформація про вибраний вузол ID: {node_id} ---")
        log_lines.append(
            f"Координати на графіку: [{picked_coords[0]:.3f}, {picked_coords[1]:.3f}, {picked_coords[2]:.3f}]")

        if self.mesh_instance and 0 <= node_id < self.mesh_instance.nqp:
            initial_coords = self.mesh_instance.AKT[node_id]
            log_lines.append(
                f"Початкові координати: [{initial_coords[0]:.3f}, {initial_coords[1]:.3f}, {initial_coords[2]:.3f}]")

        if self.U_solution is not None and self.mesh_instance and 0 <= node_id < self.mesh_instance.nqp:
            disp_x = self.U_solution[node_id * 3 + 0];
            disp_y = self.U_solution[node_id * 3 + 1];
            disp_z = self.U_solution[node_id * 3 + 2]
            log_lines.append(f"Переміщення (Ux,Uy,Uz): [{disp_x:.3e}, {disp_y:.3e}, {disp_z:.3e}]")

        if self.stress_calculator and self.stress_calculator.nodal_stresses.size > 0 and \
                self.mesh_instance and 0 <= node_id < self.mesh_instance.nqp:
            stresses = self.stress_calculator.nodal_stresses[node_id, :];
            p_stresses = self.stress_calculator.nodal_principal_stresses[node_id, :]
            log_lines.append(f"Напруження (xx,yy,zz,xy,yz,zx):")
            log_lines.append(f"  [{stresses[0]:.3e}, {stresses[1]:.3e}, {stresses[2]:.3e},")
            log_lines.append(f"   {stresses[3]:.3e}, {stresses[4]:.3e}, {stresses[5]:.3e}]")
            log_lines.append(f"Головні напруження (S1,S2,S3):")
            log_lines.append(f"  [{p_stresses[0]:.3e}, {p_stresses[1]:.3e}, {p_stresses[2]:.3e}]")

        self.after(0, lambda: self._log_message("\n".join(log_lines), clear_previous=True))


if __name__ == "__main__":
    app = FEMApp()
    app.mainloop()