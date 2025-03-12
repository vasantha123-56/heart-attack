import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tkinter import Tk, Entry, Button, StringVar, messagebox, Canvas, Label, Frame
from PIL import Image, ImageTk

# Load and preprocess dataset
file_path = r"C:\\Users\\bvasa\\Downloads\\updatedheatattack3.csv"
data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()
data.rename(columns={'Sex': 'Gender (M/F)'}, inplace=True)
data['Gender (M/F)'] = data['Gender (M/F)'].map({'Male': 0, 'Female': 1})

if 'Heart Attack Risk' in data.columns:
    X = data.drop(['Heart Attack Risk'], axis=1)
    y = data['Heart Attack Risk']
else:
    raise KeyError("Column 'Heart Attack Risk' not found in the DataFrame.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

root = Tk()
root.title("Heart Attack Prediction")
root.geometry("1424x768")

image_path = r"C:\\Users\\bvasa\\Downloads\\pic5.jpg"
bg_image = Image.open(image_path)
window_width = root.winfo_screenwidth()
window_height = root.winfo_screenheight()
bg_image = bg_image.resize((window_width, window_height), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

canvas = Canvas(root, width=1424, height=768)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

canvas.create_text(500, 40, text="Heart Attack Prediction ❤", font=("Arial", 20, "bold"), fill="white")

labels = [
    "Age:", "Gender (M/F):", "Cholesterol:", "Physical Activity Level:", "Sleep Duration:", "Diet Quality:",
    "Heart Rate:", "Diabetes:", "Family History:", "Smoking:", "Obesity:", "Stress Level:"
]

entries = []
for i, label_text in enumerate(labels):
    canvas.create_text(500, 100 + i * 40, text=label_text, font=("Arial", 18, "bold"), fill="white", anchor="e")
    entry_var = StringVar()
    entry = Entry(root, textvariable=entry_var, font=("Arial", 18), bg=root.cget('bg'), width=10, bd=2, relief="flat")
    canvas.create_window(600, 100 + i * 40, window=entry)
    entries.append(entry_var)

(age_var, gender_var, cholesterol_var, physical_activity_var, sleep_duration_var, diet_quality_var,
 heart_rate_var, diabetes_var, family_history_var, smoking_var, obesity_var, stress_level_var) = entries

current_display_data = None  # Track the current data to persist display
def predict_heart_attack():
    try:
        age = int(age_var.get())
        gender = 0 if gender_var.get().strip().lower() == 'm' else 1
        cholesterol = int(cholesterol_var.get())
        physical_activity = int(physical_activity_var.get())
        sleep_duration = int(sleep_duration_var.get())
        diet_quality = int(diet_quality_var.get())
        heart_rate = int(heart_rate_var.get())
        diabetes = int(diabetes_var.get())
        family_history = int(family_history_var.get())
        smoking = int(smoking_var.get())
        obesity = int(obesity_var.get())
        stress_level = int(stress_level_var.get())

        input_data = pd.DataFrame([[age, gender, cholesterol, physical_activity, sleep_duration, diet_quality,
                                    heart_rate, diabetes, family_history, smoking, obesity, stress_level]],
                                  columns=X.columns)

        prediction = model.predict(input_data)[0]
        result = "Heart Attack Detected!" if prediction == 1 else "No Heart Attack Detected 😊!"
        messagebox.showinfo("Prediction Result", result)

        canvas.create_text(650, 700, text="Thank you for trusting us with your care, Stay healthy! 😊", 
                           font=("Arial", 14, "bold"), fill="white")

    except ValueError as ve:
        messagebox.showerror("Input Error", f"Please enter valid numeric values: {ve}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def display_data(sample):
    global current_display_data
    current_display_data = sample
    clear_table()
    table_frame.place(x=1000, y=100)

    for i, (feature, value) in enumerate(sample.items()):
        if feature != "Heart Attack Risk":
            Label(table_frame, text=f"{feature}:", font=("Arial", 14, "bold"), bg="lightblue", width=20).grid(row=i, column=0, padx=10, pady=2)
            Label(table_frame, text=str(value), font=("Arial", 14), bg="white", width=15).grid(row=i, column=1, padx=10, pady=2)

    close_button = Button(table_frame, text="X", font=("Arial", 12, "bold"), bg="red", fg="white", command=close_data_window)
    close_button.grid(row=i+1, column=1, padx=10, pady=10)

def show_healthy_data():
    if current_display_data and current_display_data.get("Heart Attack Risk") == 0:
        table_frame.place(x=1000, y=100)
    else:
        healthy_sample = data[data["Heart Attack Risk"] == 0].sample(n=1).iloc[0].to_dict()
        display_data(healthy_sample)

def show_unhealthy_data():
    if current_display_data and current_display_data.get("Heart Attack Risk") == 1:
        table_frame.place(x=1000, y=100)
    else:
        unhealthy_sample = data[data["Heart Attack Risk"] == 1].sample(n=1).iloc[0].to_dict()
        display_data(unhealthy_sample)

def clear_table():
    for widget in table_frame.winfo_children():
        widget.destroy()

def close_data_window():
    table_frame.place_forget()

table_frame = Frame(root, bg="lightblue", padx=10, pady=10)

predict_button = Button(root, text="Predict", bg="lightgrey", font=("Arial", 18), command=predict_heart_attack)
canvas.create_window(600, 610, window=predict_button)

healthy_button = Button(root, text="Healthy Data", bg="green", fg="white", font=("Arial", 14, "bold"), command=show_healthy_data)
canvas.create_window(1150, 50, window=healthy_button)

unhealthy_button = Button(root, text="Unhealthy Data", bg="red", fg="white", font=("Arial", 14, "bold"), command=show_unhealthy_data)
canvas.create_window(1400, 50, window=unhealthy_button)

root.mainloop()
