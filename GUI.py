import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText
from tkinter import messagebox
import tempfile
import webbrowser
import folium
from PIL import Image, ImageTk

from Route_Finder import a_star_search_with_recovery, coords

locations = sorted(coords.keys())
BACKGROUND_PATH = "D:/Shaheer/IBA/Sem 4/AI/Project/Img.jpg"

def create_map(path, transitions):
    if not path or path[0] not in coords:
        return None
    m = folium.Map(location=coords[path[0]], zoom_start=13)
    for i, stop in enumerate(path):
        coord = coords.get(stop)
        if not coord:
            continue
        color = "green" if i == 0 else "red" if i == len(path) - 1 else "blue"
        folium.Marker(location=coord, tooltip=stop, icon=folium.Icon(color=color)).add_to(m)
    for frm, to, bus in transitions:
        if frm in coords and to in coords:
            tooltip = f"{bus}" if bus and bus != "RECOVERY_JUMP" else "Recovery Jump"
            color = "blue" if bus != "RECOVERY_JUMP" else "gray"
            dash = "5, 5" if bus == "RECOVERY_JUMP" else None
            folium.PolyLine([coords[frm], coords[to]], color=color, weight=5, dash_array=dash, tooltip=tooltip).add_to(m)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    m.save(temp_file.name)
    return temp_file.name


class RouteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Karachi Bus Route Finder")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)

        self.start_var = ttkb.StringVar()
        self.end_var = ttkb.StringVar()

        self.set_background_and_ui()

    def set_background_and_ui(self):
        try:
            image = Image.open(BACKGROUND_PATH).resize((1000, 700))
            self.bg_image = ImageTk.PhotoImage(image)
            bg_label = ttkb.Label(self.root, image=self.bg_image)
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print(f"Failed to load background image: {e}")

        card = ttkb.Frame(
            self.root,
            bootstyle="light",
            padding=(30, 20),
            relief="ridge",
            borderwidth=3
        )
        card.place(relx=0.5, rely=0.4, anchor="center")

        ttkb.Label(card, text="Karachi Bus Route Finder", font=("Helvetica", 18, "bold"), bootstyle="dark").grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # From
        ttkb.Label(card, text="From:", font=("Segoe UI", 12)).grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.start_combo = ttkb.Combobox(card, textvariable=self.start_var, width=40, bootstyle="info")
        self.start_combo['postcommand'] = lambda: self.filter_combobox_on_click(self.start_combo, self.start_var)
        self.start_combo.grid(row=1, column=1, padx=10, pady=5)

        # To
        ttkb.Label(card, text="To:", font=("Segoe UI", 12)).grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.end_combo = ttkb.Combobox(card, textvariable=self.end_var, width=40, bootstyle="info")
        self.end_combo['postcommand'] = lambda: self.filter_combobox_on_click(self.end_combo, self.end_var)
        self.end_combo.grid(row=2, column=1, padx=10, pady=5)
        


        

        # Find button
        self.find_button = ttkb.Button(
            card,
            text="🔍 Find Route",
            command=self.find_route,
            bootstyle="success-outline",
            width=20
        )
        self.find_button.grid(row=3, column=0, columnspan=2, pady=(20, 0))

    def filter_combobox_on_click(self, combobox, var):
        typed = var.get().strip().lower()
        filtered = [loc for loc in locations if typed in loc.lower()]
        combobox['values'] = filtered


    def find_route(self):
        start = self.start_var.get().strip()
        end = self.end_var.get().strip()

        if not start or not end:
            messagebox.showerror("Error", "Please select both start and end locations")
            return

        if start == end:
            messagebox.showerror("Error", "Start and end cannot be the same")
            return

        try:
            path, transitions = a_star_search_with_recovery(start, end, 30.0)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while finding route:\n{e}")
            return

        if not path:
            messagebox.showinfo("Result", "No route found.")
            return

        self.show_result_window(start, end, path, transitions)

    def show_result_window(self, start, end, path, transitions):
        result_win = ttkb.Toplevel(self.root)
        result_win.title("Route Result")
        result_win.geometry("900x600")

        ScrolledText(result_win, height=30, width=110).pack(padx=10, pady=10)
        result_text = result_win.winfo_children()[0]

        result_text.insert("end", f"Route from {start} to {end}:\n\n")
        result_text.insert("end", " → ".join(path) + "\n\n")

        result_text.insert("end", "Transitions:\n")
        for i, (frm, to, bus) in enumerate(transitions, 1):
            if bus == "RECOVERY_JUMP":
                result_text.insert("end", f"{i}. {frm} → {to} (Recovery Jump)\n")
            elif bus:
                result_text.insert("end", f"{i}. {frm} → {to} via Bus {bus}\n")
            else:
                result_text.insert("end", f"{i}. {frm} → {to} (Walk)\n")

        html_path = create_map(path, transitions)
        if html_path:
            webbrowser.open(f"file://{html_path}")


# Run the app
if __name__ == "__main__":
    app = RouteApp(ttkb.Window(themename="morph"))  # Try "solar", "vapor", "cosmo" themes too
    app.root.mainloop()
